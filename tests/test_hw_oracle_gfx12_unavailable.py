"""The hardware-oracle tools that cannot run on GFX12 say so and exit 77.

    python -I tests/test_hw_oracle_gfx12_unavailable.py

docs/testing.md: a tool that cannot use the hardware oracle prints `hardware oracle unavailable: ...` and exits 77.
On an RDNA4 card (measured on gfx1200) eleven tools did something else: an uncaught RuntimeError or ValueError, a
"harness error" exit 2, or rows printed as MISMATCH whose hardware value was the reason text. Their kernels cannot
run there for one of three reasons, each known before a kernel is assembled:

- every kernel sets the MODE fields IEEE and DX10_CLAMP, which gfx1170+ fixes (asm.modes_skip_reason);
- the kernel text or raw encoding is written for an older family (asm.family_skip_reason);
- the T# layout of the family is not built (tsharp.check_arch).

Each tool is run here with a fake oracle that reports gfx1200 and fails the test if a kernel is run. Standard library
only; no GPU, no AMD tools, no game files.
"""
import contextlib
import importlib
import io
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for path in (TOOLS, os.path.join(TOOLS, "hw_oracle")):
    if path not in sys.path:
        sys.path.insert(0, path)

import runner as flat_runner  # noqa: E402  (tools that put tools/hw_oracle on sys.path import it this way)
from hw_oracle import asm, runner as package_runner  # noqa: E402

# (module, arguments, why its kernels cannot run on GFX12)
TOOLS_UNAVAILABLE_ON_GFX12 = [
    ("f32_fuzz", ["--oracle", "hw", "--programs", "4"], "MODE"),
    ("f16_trans_sweep", [], "MODE"),
    ("fmisc_diff", ["--oracle", "hw", "--programs", "4"], "MODE"),
    ("pknorm_probe", ["--oracle", "hw"], "MODE"),
    ("fma_mix_probe", ["--oracle", "hw"], "MODE"),
    ("vinterp_hw_probe", ["--oracle", "hw"], "MODE"),
    ("vcmp16_opsel_probe", ["--oracle", "hw"], "is written for gfx10"),
    ("sampler_lod_probe", ["--oracle", "hw"], "is written for gfx10"),
    ("buffer_swizzle_probe", ["--oracle", "hw"], "is written for gfx10 and gfx11"),
    ("smem_neg_probe", ["--oracle", "hw"], "is written for gfx10 and gfx11"),
    ("texoff_probe", ["--oracle", "hw"], "no T# layout for gfx12"),
]


class KernelRan(Exception):
    """The fake oracle was asked to run a kernel."""


class FakeOracle:
    """An oracle that reports an architecture and must not be asked to run anything."""
    device = "fake device"
    true16 = False
    chunk = 64

    def __init__(self, arch):
        self.arch = arch
        self.ran = False
        self.closed = False

    def run(self, programs, words=4):
        self.ran = True
        raise KernelRan(f"{len(programs)} kernels on {self.arch}")

    def close(self):
        self.closed = True


def run_tool(module, argv, arch):
    """(exit code or the exception main() raised, everything printed, the fake oracle)."""
    oracle = FakeOracle(arch)
    tool = importlib.import_module(module)
    out = io.StringIO()
    with mock.patch.object(flat_runner, "open_oracle", lambda **kw: oracle), \
            mock.patch.object(package_runner, "open_oracle", lambda **kw: oracle), \
            mock.patch.object(sys, "argv", [module + ".py", *argv]), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            rc = tool.main()
        except SystemExit as e:
            rc = e.code
        except Exception as e:  # noqa: BLE001 - reported to the caller, which decides whether it is expected
            rc = e
    return rc, out.getvalue(), oracle


class SkipReasons(unittest.TestCase):
    GAME = {"ieee": 0, "dx10_clamp": 1, "denorm32": 0}

    def test_modes_skip_reason_needs_every_mode_to_be_unsettable(self):
        for arch in ("gfx1200", "gfx1201", "gfx1170"):
            why = asm.modes_skip_reason([self.GAME, dict(self.GAME, ieee=1)], arch)
            self.assertIn("cannot be set on " + arch, why)
            self.assertEqual(why, asm.mode_skip_reason(self.GAME, arch))
            # one kernel that leaves the fixed fields alone can still run, so the tool has something to measure
            self.assertIsNone(asm.modes_skip_reason([self.GAME, {"denorm32": 3}], arch))
            self.assertIsNone(asm.modes_skip_reason([self.GAME, None], arch))
        for arch in ("gfx1030", "gfx1036", "gfx1100", "gfx1150"):
            self.assertIsNone(asm.modes_skip_reason([self.GAME, dict(self.GAME, ieee=1)], arch), arch)
        self.assertIsNone(asm.modes_skip_reason([], "gfx1200"))

    def test_family_skip_reason_names_the_families_and_the_architecture(self):
        self.assertIsNone(asm.family_skip_reason("gfx1030", ("gfx10",), "this kernel"))
        self.assertIsNone(asm.family_skip_reason("gfx1150", ("gfx10", "gfx11"), "this kernel"))
        self.assertEqual(asm.family_skip_reason("gfx1150", ("gfx10",), "this kernel"),
                         "this kernel is written for gfx10; gfx1150 is gfx11")
        self.assertEqual(asm.family_skip_reason("gfx1200", ("gfx10", "gfx11"), "this kernel"),
                         "this kernel is written for gfx10 and gfx11; gfx1200 is gfx12")


class ToolsOnGfx12(unittest.TestCase):
    def test_each_tool_reports_unavailable_and_runs_nothing(self):
        for module, argv, reason in TOOLS_UNAVAILABLE_ON_GFX12:
            with self.subTest(tool=module):
                rc, out, oracle = run_tool(module, argv, "gfx1200")
                self.assertEqual(rc, 77, out[-600:])
                self.assertIn("hardware oracle unavailable: ", out)
                self.assertIn(reason, out.split("hardware oracle unavailable: ", 1)[1])
                self.assertNotIn("MISMATCH", out)
                self.assertNotIn("Traceback", out)
                self.assertFalse(oracle.ran, "a kernel was run on an architecture that cannot run it")
                self.assertTrue(oracle.closed, "the oracle was left open")

    def test_the_same_tools_still_run_their_kernels_on_rdna2(self):
        # The gate must not shut out the family the tools were written for: on gfx1030 each one reaches Oracle.run.
        for module, argv, _reason in TOOLS_UNAVAILABLE_ON_GFX12:
            with self.subTest(tool=module):
                rc, out, oracle = run_tool(module, argv, "gfx1030")
                self.assertNotEqual(rc, 77, out[-600:])
                self.assertNotIn("hardware oracle unavailable", out)
                self.assertTrue(oracle.ran, f"no kernel was run on gfx1030 (main returned {rc!r})")

    def test_gfx11_keeps_what_its_assembler_takes(self):
        # gfx1150 has the same fixed-MODE-free descriptor as RDNA2 and still assembles the MUBUF text, but the raw
        # 16-bit compare words and the GFX10 mip fields are not its own.
        still_run = ("f32_fuzz", "fmisc_diff", "pknorm_probe", "fma_mix_probe", "buffer_swizzle_probe",
                     "smem_neg_probe")
        gfx10_only = ("vcmp16_opsel_probe", "sampler_lod_probe")
        for module, argv, _reason in TOOLS_UNAVAILABLE_ON_GFX12:
            if module in still_run:
                with self.subTest(tool=module):
                    rc, out, oracle = run_tool(module, argv, "gfx1150")
                    self.assertTrue(oracle.ran, f"no kernel was run on gfx1150 (main returned {rc!r}): {out[-300:]}")
            elif module in gfx10_only:
                with self.subTest(tool=module):
                    rc, out, oracle = run_tool(module, argv, "gfx1150")
                    self.assertEqual(rc, 77, out[-600:])
                    self.assertIn("is written for gfx10; gfx1150 is gfx11", out)
                    self.assertFalse(oracle.ran)


if __name__ == "__main__":
    unittest.main()

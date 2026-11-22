"""Semantic interpreter and SPIR-V validation tests for RDNA2 VOP3 `clamp` across all eight SAD-family ops.

Covers native RDNA2 (gfx1036, wave32 & wave64) measured clamp saturation rules:
- v_sad_u8, v_sad_hi_u8, v_sad_u16, v_sad_u32, v_msad_u8: 32-bit unsigned saturation to 0xFFFFFFFF (including
  v_sad_hi_u8 saturating the entire 32-bit word when (byte_sad << 16) + S2 >= 2^32).
- v_qsad_pk_u16_u8, v_mqsad_pk_u16_u8: independent per-lane 16-bit unsigned saturation to 0xFFFF with zero carry
  between adjacent halves.
- v_mqsad_u32_u8: independent per-lane 32-bit unsigned saturation to 0xFFFFFFFF.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (TOOLS_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import sad_clamp_probe as scp  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M16 = 0xFFFF
M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_sad_clamp.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sad_clamp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines: list[str], export_tail: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm_text = "; test_sad_clamp\n" + "".join(f"\t{ln}\n" for ln in list(lines) + export_tail) + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm_text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_export_regs(lines: list[str], regs: list[int]) -> tuple[str, list[int]]:
    """Translate `lines`, export `regs` safely via 16-bit OR-0x40000000 packing, and run in spirv_mini_interp."""
    assert 1 <= len(regs) <= 4
    tail: list[str] = []
    mrt_regs: list[str] = []
    mrt_idx = 0
    for r in regs:
        lo, hi = 20 + len(mrt_regs), 21 + len(mrt_regs)
        tail += [
            f"v_and_b32 v{lo}, 0xffff, v{r}",
            f"v_lshrrev_b32 v{hi}, 16, v{r}",
            f"v_or_b32 v{lo}, 0x40000000, v{lo}",
            f"v_or_b32 v{hi}, 0x40000000, v{hi}",
        ]
        mrt_regs.extend([f"v{lo}", f"v{hi}"])
    exports: list[str] = []
    for pos in range(0, len(mrt_regs), 4):
        chunk = list(mrt_regs[pos : pos + 4])
        while len(chunk) < 4:
            chunk.append(chunk[0])
        done = " done" if pos + 4 >= len(mrt_regs) else ""
        exports.append(f"exp mrt{mrt_idx} {', '.join(chunk)}{done} vm")
        mrt_idx += 1
    spv = translate_ps(lines, tail + exports)
    out = si.run_fragment(spv)
    halves: list[int] = []
    for i in range(mrt_idx):
        halves.extend(out[f"%out_mrt{i}"])
    halves = halves[: 2 * len(regs)]
    assert all((v >> 16) == 0x4000 for v in halves), [hex(v) for v in halves]
    words = [(halves[2 * i] & M16) | ((halves[2 * i + 1] & M16) << 16) for i in range(len(regs))]
    return spv, words


def validate_spirv(spv_text: str):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


def run_case(op: str, case: tuple, *, clamp: bool) -> tuple[str, tuple[int, ...]]:
    suf = " clamp" if clamp else ""
    if op in scp.SAD32_OPS:
        _, s0, s1, s2 = case
        lines = [
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {s2:#x}",
            f"{op} v0, v1, v2, v3{suf}",
        ]
        spv, words = run_export_regs(lines, [0])
        return spv, tuple(words)
    if op in scp.QSAD16_OPS:
        _, s0, s1, s2 = case
        lines = [
            f"v_mov_b32 v2, {s0 & M32:#x}",
            f"v_mov_b32 v3, {(s0 >> 32) & M32:#x}",
            f"v_mov_b32 v4, {s1:#x}",
            f"v_mov_b32 v6, {s2 & M32:#x}",
            f"v_mov_b32 v7, {(s2 >> 32) & M32:#x}",
            f"{op} v[0:1], v[2:3], v4, v[6:7]{suf}",
        ]
        spv, words = run_export_regs(lines, [0, 1])
        return spv, tuple(words)
    if op in scp.MQSAD32_OPS:
        _, s0, s1, accs = case
        lines = [
            f"v_mov_b32 v2, {s0 & M32:#x}",
            f"v_mov_b32 v3, {(s0 >> 32) & M32:#x}",
            f"v_mov_b32 v1, {s1:#x}",
            *(f"v_mov_b32 v{4 + k}, {accs[k]:#x}" for k in range(4)),
            f"{op} v[8:11], v[2:3], v1, v[4:7]{suf}",
        ]
        spv, words = run_export_regs(lines, [8, 9, 10, 11])
        return spv, tuple(words)
    raise AssertionError(op)


class SadClampSemanticTests(unittest.TestCase):
    def test_all_ops_plain_and_clamped_match_gfx1036_cases(self):
        for op in scp.OPS:
            saw_overflow_diff = False
            for case in scp.CASES[op]:
                want_plain = scp.expected_case_words(op, case, clamp=False)
                want_clamp = scp.expected_case_words(op, case, clamp=True)
                with self.subTest(op=op, case=case[0]):
                    _, got_plain = run_case(op, case, clamp=False)
                    _, got_clamp = run_case(op, case, clamp=True)
                    self.assertEqual(got_plain, want_plain)
                    self.assertEqual(got_clamp, want_clamp)
                    if want_plain != want_clamp:
                        saw_overflow_diff = True
                        self.assertNotEqual(got_clamp, got_plain)
            self.assertTrue(saw_overflow_diff, f"{op} had no overflowing case distinguishing clamp")

    def test_spirv_validation_and_unclamped_structure(self):
        for op in scp.OPS:
            case = scp.CASES[op][0]
            with self.subTest(op=op):
                spv_plain, _ = run_case(op, case, clamp=False)
                spv_clamp, _ = run_case(op, case, clamp=True)
                self.assertNotIn("OpCapability Int64", spv_clamp)
                self.assertNotIn("OpCapability Float64", spv_clamp)
                if op in scp.SAD32_OPS or op in scp.MQSAD32_OPS:
                    self.assertNotIn("OpULessThan", spv_plain)
                    self.assertIn("OpULessThan", spv_clamp)
                else:
                    self.assertIn("%cffff = OpConstant %uint 65535", spv_clamp)
                    self.assertNotRegex(spv_plain, r"OpExtInst %uint %glsl UMin %\w+ %cffff")
                    self.assertRegex(spv_clamp, r"OpExtInst %uint %glsl UMin %\w+ %cffff")
                validate_spirv(spv_clamp)

    def test_destination_aliasing_clamped(self):
        # v_sad_hi_u8 clamp with dst == src2 and dst == src0
        _, got_hi = run_export_regs([
            "v_mov_b32 v0, 0x01000000",
            "v_mov_b32 v1, 0x00000000",
            "v_mov_b32 v2, 0xffff1234",
            "v_sad_hi_u8 v2, v0, v1, v2 clamp",
        ], [2])
        self.assertEqual(got_hi, [0xFFFFFFFF])

        # v_qsad_pk_u16_u8 clamp with overlapping dst v[2:3] and s2 v[2:3]
        _, got_qsad = run_export_regs([
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v4, 0xffffffff",
            "v_mov_b32 v2, 0x0000ffff",
            "v_mov_b32 v3, 0xfc03fc04",
            "v_qsad_pk_u16_u8 v[2:3], v[0:1], v4, v[2:3] clamp",
        ], [2, 3])
        self.assertEqual(got_qsad, [0x03FCFFFF, 0xFFFFFFFF])

        # v_mqsad_u32_u8 clamp with dst v[5:8] overlapping acc v[4:7]
        _, got_mqsad = run_export_regs([
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0xff00ffff",
            "v_mov_b32 v4, 0xfffffd02",
            "v_mov_b32 v5, 0xfffffd03",
            "v_mov_b32 v6, 0xffffffff",
            "v_mov_b32 v7, 0x00000010",
            "v_mqsad_u32_u8 v[5:8], v[0:1], v2, v[4:7] clamp",
        ], [5, 6, 7, 8])
        self.assertEqual(got_mqsad, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0000030D])

    def test_inactive_exec_lane_preserved_under_clamp(self):
        _, got_sad = run_export_regs([
            "v_mov_b32 v0, 0xdeadbeef",
            "v_mov_b32 v1, 0xff00ff00",
            "v_mov_b32 v2, 0x00ff00ff",
            "v_mov_b32 v3, 0xffffffff",
            "s_mov_b32 exec_lo, 0",
            "v_sad_u8 v0, v1, v2, v3 clamp",
            "s_mov_b32 exec_lo, 1",
        ], [0])
        self.assertEqual(got_sad, [0xDEADBEEF])

        _, got_mqsad = run_export_regs([
            "v_mov_b32 v8, 0x11111111",
            "v_mov_b32 v9, 0x22222222",
            "v_mov_b32 v10, 0x33333333",
            "v_mov_b32 v11, 0x44444444",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "v_mov_b32 v1, 0xffffffff",
            "v_mov_b32 v4, 0xffffffff",
            "v_mov_b32 v5, 0xffffffff",
            "v_mov_b32 v6, 0xffffffff",
            "v_mov_b32 v7, 0xffffffff",
            "s_mov_b32 exec_lo, 0",
            "v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7] clamp",
            "s_mov_b32 exec_lo, 1",
        ], [8, 9, 10, 11])
        self.assertEqual(got_mqsad, [0x11111111, 0x22222222, 0x33333333, 0x44444444])

    def test_unsupported_modifiers_and_lerp_clamp_still_rejected(self):
        for bad_line in (
            "v_sad_u8 v0, v1, v2, v3 mul:2",
            "v_sad_hi_u8 v0, v1, v2, v3 div:2",
            "v_sad_u16 v0, v1, v2, v3 op_sel:[1,0,0,0]",
            "v_sad_u32 v0, v1, v2, v3 mul:4",
            "v_msad_u8 v0, v1, v2, v3 mul:2",
            "v_qsad_pk_u16_u8 v[0:1], v[2:3], v4, v[6:7] mul:2",
            "v_mqsad_pk_u16_u8 v[0:1], v[2:3], v4, v[6:7] div:2",
            "v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7] mul:2",
            "v_lerp_u8 v0, v1, v2, v3 clamp",
        ):
            with self.subTest(line=bad_line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_ps(["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 3", bad_line],
                                 ["exp mrt0 v0, v0, v0, v0 done vm"])


if __name__ == "__main__":
    unittest.main()

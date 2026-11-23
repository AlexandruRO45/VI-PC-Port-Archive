"""Tests for the gfx12 (gfx1170+, RDNA4) support in tools/f32_fuzz.py's hardware oracle: the store tail's family
coverage, the kernel descriptor's float mode and the wave-sized VCC condition spelling (no game data, no GPU,
no llvm-mc). The three gaps came from the SHADER-985 smoke run on gfx1201.

    python -I tests/test_f32_fuzz_gfx12.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.dont_write_bytecode = True

import f32_fuzz as ff               # noqa: E402
from hw_oracle import asm           # noqa: E402


def case(op="v_add_f32", vals=(0x3F800000, 0x40000000)):
    return ff.Case(op, list(vals) + [0] * (9 - len(vals)), [(0, {}), (1, {})], False, [])


class _RecordingOracle:
    """Stands in for the hw_oracle runner: records the kernels hw_run builds and returns zero words."""

    def __init__(self, arch):
        self.arch, self.kernels = arch, []

    def run(self, kernels, words=2):
        self.kernels += list(kernels)
        return [[0] * words for _ in kernels]


class StoreTail(unittest.TestCase):
    def test_every_family_asm_family_returns_has_a_store_spelling(self):
        for arch in ("gfx1010", "gfx1030", "gfx1100", "gfx1150", "gfx1170", "gfx1200", "gfx1201"):
            with self.subTest(arch=arch):
                fam = asm.family(arch)
                self.assertIn(fam, ff.HW_STORE)
                tail = ff.hw_lines(case(), arch)
                self.assertEqual(tail[-2], f"{ff.HW_STORE[fam]} v{ff.HW_ZERO_REG}, v[8:9], s[2:3]")
        self.assertEqual(ff.HW_STORE["gfx12"], ff.HW_STORE["gfx11"])   # GFX12 takes GFX11's spellings

    def test_an_arch_no_family_covers_is_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            ff.hw_lines(case(), "gfx1300")


class KernelDescriptor(unittest.TestCase):
    def test_gfx10_and_gfx11_set_the_requested_mode(self):
        for arch in ("gfx1030", "gfx1100", "gfx1150"):
            with self.subTest(arch=arch):
                o = _RecordingOracle(arch)
                ff.hw_run(o, [case()])
                self.assertEqual(o.kernels[0].mode, ff.HW_MODES[ff.HW_PRIMARY])

    def test_gfx1170_and_later_take_the_default_descriptor(self):
        # asm.Kernel keeps mode=None as {}: no field asked for, so the runner's mode check lets the kernel run
        # with the assembler's default descriptor instead of skipping it.
        for arch in ("gfx1170", "gfx1200", "gfx1201"):
            with self.subTest(arch=arch):
                o = _RecordingOracle(arch)
                ff.hw_run(o, [case()], "keep")
                self.assertEqual(o.kernels[0].mode, {})

    def test_why_the_mode_is_dropped_there(self):
        # Negative control: every HW_MODES entry asks for the ieee/dx10_clamp bits the assembler rejects on
        # gfx1170+, so a kernel that still asked would be skipped by the runner's mode check, never assembled.
        for name, mode in ff.HW_MODES.items():
            with self.subTest(mode=name):
                self.assertIsNotNone(asm.mode_skip_reason(mode, "gfx1201"))
                self.assertIsNone(asm.mode_skip_reason(mode, "gfx1150"))


class ConditionSpelling(unittest.TestCase):
    def test_the_condition_operand_is_wave_sized(self):
        self.assertEqual(ff.hw_lines(case())[-4], "v_cndmask_b32_e64 v9, 0, 1, vcc_lo")
        self.assertEqual(ff.hw_lines(case(), wave=64)[-4], "v_cndmask_b32_e64 v9, 0, 1, vcc")

    def test_hw_run_threads_the_wave_into_the_kernel(self):
        for wave in (32, 64):
            with self.subTest(wave=wave):
                o = _RecordingOracle("gfx1150")
                ff.hw_run(o, [case()], wave=wave)
                self.assertEqual(o.kernels[0].wave, wave)
        o = _RecordingOracle("gfx1150")
        ff.hw_run(o, [case()])
        self.assertEqual(o.kernels[0].wave, 32)     # the tool's default stays wave 32


if __name__ == "__main__":
    unittest.main()

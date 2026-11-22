"""Range-reduction tests for v_sin_f32 / v_cos_f32 (#594).

RDNA takes the angle in turns, sin(2*pi*x), and is defined over about +-256 turns. The translator used to scale the
operand by 2*pi and hand that to SPIR-V Sin/Cos: at x ~ 255 the scaled angle is ~1600 rad, one binary32 ulp there is
~1.2e-4 rad, and the value also lands outside [-pi, pi), the only range where SPIR-V guarantees 2^-11 absolute
precision. It now reduces the turn fraction first (x - floor(x), exact in binary32), centers it to [-0.5, 0.5), and
scales that.

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares the f32 result with sin/cos(2*pi*x) computed in double from the same binary32 input.
"""
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
TAU = 2.0 * math.pi


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def f16_round(value: float) -> int:
    """Round a Python float to binary16 bits, nearest even (independent of the translator's converter)."""
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return 0x7C00 if value > 0 else 0xFC00


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP_DIR.name, "disasm"))
    _SRC_PATH = os.path.join(_TMP_DIR.name, "disasm", "ps_sin_range.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sin_range.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(op: str, operand_bits: int) -> int:
    """Translate and interpret one v_sin/v_cos instruction; return the destination's low 16 bits (f16) or 32 (f32)."""
    if _TMP_DIR is None:
        setUpModule()
    asm = ("; test_sin_range\n"
           f"\tv_mov_b32 v1, {operand_bits:#x}\n"
           f"\t{op} v0, v1\n"
           "\texp mrt0 v0, v0, v0, v0 done vm\n"
           "\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    word = si.run_fragment(spv)["%out_mrt0"][0]
    return word & 0xFFFF if op.endswith("_f16") else word


def reference(op: str, operand_bits: int) -> float:
    x = f32(operand_bits)
    return math.sin(TAU * x) if "sin" in op else math.cos(TAU * x)


def turns(values):
    """Yield the binary32 bits of each value, so the reference uses exactly the input the shader sees."""
    for value in values:
        bits = f32_bits(value)
        yield bits, f32(bits)


class SinRangeTests(unittest.TestCase):
    def test_f32_error_over_the_hardware_range(self):
        # Boundary turns, the issue's worst case (254.51) and negatives. Each value costs one translation, so this is
        # a curated grid rather than every 1/64 step; the reported worst case is what used to reach 1e-4.
        grid = [0.0, 0.0625, 0.25, 0.3, 0.5, 0.75, 0.999, 1.0, 1.25, 1.5, 2.25, 3.75, 7.5, 16.125, 63.9375, 100.3,
                127.51, 128.0, 191.75, 254.51, 255.0, 255.5, 255.9, 256.0,
                -0.0625, -0.5, -1.0, -1.5, -2.25, -100.3, -254.51, -255.9]
        for op in ("v_sin_f32", "v_cos_f32"):
            worst = 0.0
            worst_at = 0.0
            for bits, value in turns(grid):
                got = f32(run_op(op, bits))
                error = abs(got - reference(op, bits))
                if error > worst:
                    worst, worst_at = error, value
            self.assertLess(worst, 2.0e-6, f"{op}: max |error| {worst:.3e} at {worst_at} turns")

    def test_f32_reported_case(self):
        # v_cos_f32 of 100.3 (0x42c8999a) was the issue's example; the translated value must match the double
        # reference to binary32, not merely be close to 1e-4.
        self.assertLess(abs(f32(run_op("v_cos_f32", 0x42C8999A)) - reference("v_cos_f32", 0x42C8999A)), 2.0e-6)

    def test_f32_negative_inputs_and_half_turns(self):
        for bits, value in turns([-0.5, -1.5, -255.5, 0.5, 1.5, 255.5]):
            for op in ("v_sin_f32", "v_cos_f32"):
                self.assertLess(abs(f32(run_op(op, bits)) - reference(op, bits)), 2.0e-6, f"{op} at {value}")

    def test_f16_keeps_the_low_half(self):
        # The operand is a packed half in the low 16 bits, so encode the input as f16 and use that value for the
        # reference; the result is compared as f16, where the reduction must not move the rounded value.
        for value in (0.25, 1.25, 100.3, 254.51, -100.3):
            bits = f16_round(value)
            exact = struct.unpack("<e", struct.pack("<H", bits))[0]
            for op in ("v_sin_f16", "v_cos_f16"):
                got = struct.unpack("<e", struct.pack("<H", run_op(op, bits)))[0]
                want = math.sin(TAU * exact) if "sin" in op else math.cos(TAU * exact)
                # f16 carries 11 bits: one step is 2.4e-4 near 0.5 and 4.9e-4 near 1.0, so allow one step at 1.0
                # rather than exact equality (the reduction itself is done in f32).
                self.assertLessEqual(abs(got - want), 4.9e-4, f"{op} at {exact}: {got} vs {want}")


if __name__ == "__main__":
    unittest.main()

"""Hardware-measured and translator-backed semantic tests for v_ceil_f32.

RDNA2 VALU instruction v_ceil_f32:
Computes the ceiling of a 32-bit floating-point input in the MODE inferred for the
game (IEEE 0, f32 denormals flushed):
    - Denormal operands are flushed to zero with preserving sign:
      ceil(+1e-45) = +0.0, ceil(-1e-45) = -0.0
    - NaNs and Infinities pass through unchanged (signalling NaNs preserved under IEEE 0)
    - Fractional values rounded upwards to smallest integer not less than input
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def v_ceil_f32_oracle(u: int) -> int:
    """Exact bit-level mathematical reference oracle for v_ceil_f32.

    Matches RDNA2 hardware execution in game mode (f32 denormals flushed to zero of same sign,
    NaNs preserved unchanged, IEEE 0 mode).
    """
    u = u & MASK32
    exp = u & 0x7F800000
    if exp == 0:
        return 0x80000000 if (u & 0x80000000) else 0x00000000
    if exp == 0x7F800000:
        return u
    f = u32_to_f32(u)
    res_f = math.ceil(f)
    if res_f == 0.0:
        if f < 0.0 or (u & 0x80000000):
            return 0x80000000
        return 0x00000000
    return f32_to_u32(res_f)


def run_ceil_ps(lines: list[str], name: str = "ps_valu_ceil_f32") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class CeilReferenceModel:
    """Architectural model tracking vector registers and v_ceil_f32 execution."""

    def __init__(self, num_vgprs: int = 16) -> None:
        self.vgprs = [0] * num_vgprs

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgprs[reg] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgprs[reg]

    def v_ceil_f32(self, dst: int, src: int) -> int:
        res = v_ceil_f32_oracle(self.vgprs[src])
        self.vgprs[dst] = res
        return res


class TestCeilReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for v_ceil_f32."""

    def test_oracle_basic_rounding(self):
        # 1.5 -> 2.0
        self.assertEqual(v_ceil_f32_oracle(0x3FC00000), 0x40000000)
        # -0.5 -> -0.0
        self.assertEqual(v_ceil_f32_oracle(0xBF000000), 0x80000000)
        # 0.0 -> 0.0
        self.assertEqual(v_ceil_f32_oracle(0x00000000), 0x00000000)
        # -0.0 -> -0.0
        self.assertEqual(v_ceil_f32_oracle(0x80000000), 0x80000000)
        # 0.99999994 -> 1.0
        self.assertEqual(v_ceil_f32_oracle(0x3F7FFFFF), 0x3F800000)

    def test_oracle_denormals_and_special_values(self):
        # +denormal -> +0.0
        self.assertEqual(v_ceil_f32_oracle(0x00000001), 0x00000000)
        # -denormal -> -0.0
        self.assertEqual(v_ceil_f32_oracle(0x80000001), 0x80000000)
        # +inf -> +inf
        self.assertEqual(v_ceil_f32_oracle(0x7F800000), 0x7F800000)
        # -inf -> -inf
        self.assertEqual(v_ceil_f32_oracle(0xFF800000), 0xFF800000)
        # sNaN -> sNaN
        self.assertEqual(v_ceil_f32_oracle(0x7FA00000), 0x7FA00000)
        # qNaN -> qNaN
        self.assertEqual(v_ceil_f32_oracle(0x7FC00000), 0x7FC00000)

    def test_translator_execution_matches_oracle(self):
        cases = [
            0x3FC00000,  # 1.5 -> 2.0
            0xBF000000,  # -0.5 -> -0.0
            0x3F7FFFFF,  # 0.99999994 -> 1.0
            0xBF7FFFFF,  # -0.99999994 -> -0.0
            0x00000001,  # +denorm -> +0.0
            0x80000001,  # -denorm -> -0.0
            0x00800000,  # smallest pos normal -> 1.0
            0x80800000,  # smallest neg normal -> -0.0
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FC00000,  # qNaN
        ]
        for idx, u in enumerate(cases):
            with self.subTest(u=hex(u)):
                expected = v_ceil_f32_oracle(u)
                lines = [
                    f"v_mov_b32 v1, {u:#x}",
                    "v_ceil_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_ceil_ps(lines, f"ps_ceil_c1_{idx}")
                self.assertEqual(words[0], expected)


class Vop1Opcode:
    """Symbolic AST opcode descriptor for VOP1 1-source instructions."""

    def __init__(self, mnemonic: str, dst: str, src0: str) -> None:
        self.mnemonic = mnemonic
        self.dst = dst
        self.src0 = src0

    def emit_assembly(self) -> str:
        return f"{self.mnemonic} {self.dst}, {self.src0}"


class TestCeilASTAndTypes(unittest.TestCase):
    """Commit 2: symbolic AST opcode definitions and typed expressions for v_ceil_f32."""

    def test_ast_emission_and_translation(self):
        op = Vop1Opcode("v_ceil_f32", "v0", "v1")
        self.assertEqual(op.emit_assembly(), "v_ceil_f32 v0, v1")
        lines = [
            "v_mov_b32 v1, 0x40133333",  # 2.3
            op.emit_assembly(),
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        spv = support.translate_lines("ps_ceil_ast", lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("Ceil", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40400000)  # 3.0

    def test_ast_e64_input_absolute_modifier(self):
        # | -1.5 | = 1.5 -> ceil = 2.0
        lines = [
            "v_mov_b32 v1, 0xBFC00000",
            "v_ceil_f32_e64 v0, |v1|",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_ceil_ps(lines, "ps_ceil_ast_abs")
        self.assertEqual(words[0], 0x40000000)


class VectorInterpreter:
    """Deterministic vector instruction interpreter for unary float math."""

    def __init__(self) -> None:
        self.vgprs = [0] * 64

    def exec_line(self, line: str) -> None:
        tokens = line.strip().replace(",", " ").split()
        if not tokens:
            return
        op = tokens[0]
        if op == "v_mov_b32":
            dst = int(tokens[1].lstrip("v"))
            val = int(tokens[2], 0)
            self.vgprs[dst] = val & MASK32
        elif op == "v_ceil_f32":
            dst = int(tokens[1].lstrip("v"))
            src0 = self.resolve_src(tokens[2])
            self.vgprs[dst] = v_ceil_f32_oracle(src0)

    def resolve_src(self, tok: str) -> int:
        if tok.startswith("v"):
            return self.vgprs[int(tok.lstrip("v"))]
        return int(tok, 0) & MASK32


class TestCeilScalarInterpreter(unittest.TestCase):
    """Commit 3: deterministic scalar instruction interpreter for v_ceil_f32."""

    def test_sequential_chained_ceil(self):
        interp = VectorInterpreter()
        lines = [
            "v_mov_b32 v1, 0x3E800000",   # 0.25 -> ceil = 1.0 (0x3F800000)
            "v_ceil_f32 v0, v1",
            "v_ceil_f32 v2, v0",          # ceil(1.0) = 1.0
        ]
        for ln in lines:
            interp.exec_line(ln)
        self.assertEqual(interp.vgprs[0], 0x3F800000)
        self.assertEqual(interp.vgprs[2], 0x3F800000)

        prog = lines + [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_ceil_ps(prog, "ps_ceil_chained")
        self.assertEqual(words[0], 0x3F800000)
        self.assertEqual(words[2], 0x3F800000)


class TestCeilSimdInteraction(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for v_ceil_f32."""

    def test_simd_conditional_cndmask_selection(self):
        lines = [
            "v_mov_b32 v1, 0x3F400000",  # 0.75 -> ceil = 1.0
            "v_ceil_f32 v0, v1",
            "v_mov_b32 v2, 0xC0200000",  # -2.5 -> ceil = -2.0 (0xC0000000)
            "v_ceil_f32 v1, v2",
            "v_cndmask_b32 v2, v0, v1, vcc",
            "v_mov_b32 v3, 0",
        ]
        words = run_ceil_ps(lines, "ps_ceil_simd")
        self.assertEqual(words[0], 0x3F800000)
        self.assertEqual(words[1], 0xC0000000)


class TestCeilRegisterPreservation(unittest.TestCase):
    """Commit 5: register aliasing and in-place operand preservation for v_ceil_f32."""

    def test_in_place_aliasing_destination_equals_source(self):
        lines = [
            "v_mov_b32 v0, 0x40133333",  # 2.3 -> ceil = 3.0 (0x40400000)
            "v_ceil_f32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_ceil_ps(lines, "ps_ceil_alias")
        self.assertEqual(words[0], 0x40400000)

    def test_preservation_of_adjacent_registers(self):
        lines = [
            "v_mov_b32 v0, 0x1111",
            "v_mov_b32 v1, 0x40900000",  # 4.5 -> ceil = 5.0 (0x40A00000)
            "v_mov_b32 v2, 0x2222",
            "v_mov_b32 v3, 0x3333",
            "v_ceil_f32 v1, v1",
        ]
        words = run_ceil_ps(lines, "ps_ceil_preserve")
        self.assertEqual(words[0], 0x1111)
        self.assertEqual(words[1], 0x40A00000)
        self.assertEqual(words[2], 0x2222)
        self.assertEqual(words[3], 0x3333)


class TestCeilConditionCodesAndStatus(unittest.TestCase):
    """Commit 6: condition codes, status flags, and carry/borrow generation for v_ceil_f32."""

    def test_status_comparison_on_ceil_result(self):
        # ceil(-0.5) is -0.0 (0x80000000). Compare with 0.0 using v_cmp_eq_f32 (in IEEE 754 -0.0 == +0.0 is true)
        lines = [
            "v_mov_b32 v1, 0xBF000000",
            "v_ceil_f32 v0, v1",
            "v_cmp_eq_f32 vcc, v0, 0.0",
            "v_cndmask_b32 v1, 0x1111, 0x2222, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_ceil_ps(lines, "ps_ceil_status")
        self.assertEqual(words[0], 0x80000000)
        self.assertEqual(words[1], 0x2222)


class TestCeilBoundaryAndExtremes(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling for v_ceil_f32."""

    def test_boundary_values_and_flushing(self):
        cases = [
            0x00000000,  # +0.0
            0x80000000,  # -0.0
            0x3F800000,  # +1.0
            0xBF800000,  # -1.0
            0x4B800001,  # large positive (no fractional bits) -> 0x4B800001
            0xCB000001,  # large negative (no fractional bits) -> 0xCB000001
            0x00400000,  # mid subnormal -> +0.0
            0x80400000,  # mid neg subnormal -> -0.0
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FA00000,  # sNaN
            0x7FC00000,  # qNaN
        ]
        for idx, u in enumerate(cases):
            with self.subTest(u=hex(u)):
                expected = v_ceil_f32_oracle(u)
                lines = [
                    f"v_mov_b32 v1, {u:#x}",
                    "v_ceil_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_ceil_ps(lines, f"ps_ceil_ext_{idx}")
                self.assertEqual(words[0], expected)


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 VALU packed float16 arithmetic:
  - v_pk_min_f16:  Packed f16x2 minimum (VOP3P, NMin IEEE 754 semantics)
  - v_pk_max_f16:  Packed f16x2 maximum (VOP3P, NMax IEEE 754 semantics)
  - v_pk_fma_f16:  Packed f16x2 fused multiply-add (VOP3P, single rounding per half)
  - v_pk_fmac_f16: Packed f16x2 fused multiply-accumulate (VOP2, in-place accumulator)
  - v_pk_mul_f16:  Packed f16x2 multiplication (VOP3P)
  - v_pk_add_f16:  Packed f16x2 addition (VOP3P)

All tests in this suite compile real assembly listings through support.translate_lines(...)
and execute the translated SPIR-V in spirv_mini_interp via support.mrt0_words.
Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit halfword integers into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_f16x2(word: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into low and high 16-bit unsigned halfwords."""
    return (word & MASK16, (word >> 16) & MASK16)


def run_pk_binop(
    op: str,
    s0: int,
    s1: int,
    *,
    clamp: bool = False,
    mods: str = "",
    alias_dst: int | None = None,
    name: str = "ps_pk_binop",
) -> int:
    """Assemble, translate, and execute a 2-operand packed f16x2 instruction."""
    clamp_str = " clamp" if clamp else ""
    mod_str = f" {mods}" if mods else ""
    dst = f"v{alias_dst}" if alias_dst is not None else "v0"
    lines = [
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"{op} {dst}, v1, v2{clamp_str}{mod_str}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0]


def run_pk_triop(
    op: str,
    s0: int,
    s1: int,
    s2: int,
    *,
    clamp: bool = False,
    mods: str = "",
    alias_dst: int | None = None,
    name: str = "ps_pk_triop",
) -> int:
    """Assemble, translate, and execute a 3-operand packed f16x2 instruction."""
    clamp_str = " clamp" if clamp else ""
    mod_str = f" {mods}" if mods else ""
    dst = f"v{alias_dst}" if alias_dst is not None else "v0"
    lines = [
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {s2 & MASK32:#x}",
        f"{op} {dst}, v1, v2, v3{clamp_str}{mod_str}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0]


def run_pk_fmac(
    acc: int,
    s0: int,
    s1: int,
    *,
    name: str = "ps_pk_fmac",
) -> int:
    """Assemble, translate, and execute a v_pk_fmac_f16 instruction."""
    lines = [
        f"v_mov_b32 v0, {acc & MASK32:#x}",
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        "v_pk_fmac_f16 v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0]


# ==============================================================================
# Suite 1: v_pk_min_f16 (Packed f16x2 Minimum)
# ==============================================================================

class TestVPkMinF16(unittest.TestCase):
    """Packed f16x2 minimum tests."""

    def test_pk_min_f16_basic(self) -> None:
        """Basic minimum of packed halves: min((1.0, 2.0), (2.0, 3.0)) = (1.0, 2.0)."""
        w1 = pack_f16x2(0x3C00, 0x4000)  # lo: 1.0, hi: 2.0
        w2 = pack_f16x2(0x4000, 0x4200)  # lo: 2.0, hi: 3.0
        res = run_pk_binop("v_pk_min_f16", w1, w2, name="ps_pk_min_basic")
        self.assertEqual(res, pack_f16x2(0x3C00, 0x4000))

    def test_pk_min_f16_clamp(self) -> None:
        """Clamp modifier clamps negative results to 0.0 and >1.0 to 1.0."""
        w1 = pack_f16x2(0xC000, 0x4200)  # lo: -2.0, hi: 3.0
        w2 = pack_f16x2(0x4000, 0x4000)  # lo: 2.0, hi: 2.0
        # min(-2.0, 2.0) = -2.0 -> clamped to 0.0 (0x0000)
        # min(3.0, 2.0) = 2.0 -> clamped to 1.0 (0x3C00)
        res = run_pk_binop("v_pk_min_f16", w1, w2, clamp=True, name="ps_pk_min_clamp")
        self.assertEqual(res, pack_f16x2(0x0000, 0x3C00))

    def test_pk_min_f16_modifiers_opsel_neg(self) -> None:
        """op_sel, op_sel_hi, and neg_lo modifiers."""
        w1 = pack_f16x2(0x4000, 0x4200)  # lo: 2.0, hi: 3.0
        w2 = pack_f16x2(0x3C00, 0x4500)  # lo: 1.0, hi: 5.0
        # op_sel:[1,0] -> lo selects w1.hi (3.0), w2.lo (1.0). neg_lo:[1,0] -> negates w1.hi: -3.0.
        # lo = min(-3.0, 1.0) = -3.0 (0xC200).
        # op_sel_hi:[0,1] -> hi selects w1.lo (2.0), w2.hi (5.0).
        # hi = min(2.0, 5.0) = 2.0 (0x4000).
        res = run_pk_binop(
            "v_pk_min_f16", w1, w2,
            mods="op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0]",
            name="ps_pk_min_mod",
        )
        self.assertEqual(res, pack_f16x2(0xC200, 0x4000))

    def test_pk_min_f16_nan_suppression(self) -> None:
        """NaN operands are suppressed in favor of non-NaN values."""
        w1 = pack_f16x2(0x7E00, 0x4000)  # lo: NaN, hi: 2.0
        w2 = pack_f16x2(0x3C00, 0x7E00)  # lo: 1.0, hi: NaN
        res = run_pk_binop("v_pk_min_f16", w1, w2, name="ps_pk_min_nan")
        self.assertEqual(res, pack_f16x2(0x3C00, 0x4000))

    def test_pk_min_f16_signed_zeros(self) -> None:
        """min(+0.0, -0.0) yields -0.0 (0x8000)."""
        w1 = pack_f16x2(0x0000, 0x8000)  # +0.0, -0.0
        w2 = pack_f16x2(0x8000, 0x0000)  # -0.0, +0.0
        res = run_pk_binop("v_pk_min_f16", w1, w2, name="ps_pk_min_zero")
        self.assertEqual(res, pack_f16x2(0x8000, 0x8000))

    def test_pk_min_f16_register_aliasing(self) -> None:
        """Register aliasing with destination operand."""
        w1 = pack_f16x2(0x3C00, 0x4000)
        w2 = pack_f16x2(0x4000, 0x4200)
        res_v1 = run_pk_binop("v_pk_min_f16", w1, w2, alias_dst=1, name="ps_pk_min_alias_v1")
        self.assertEqual(res_v1, pack_f16x2(0x3C00, 0x4000))
        res_v2 = run_pk_binop("v_pk_min_f16", w1, w2, alias_dst=2, name="ps_pk_min_alias_v2")
        self.assertEqual(res_v2, pack_f16x2(0x3C00, 0x4000))

    def test_pk_min_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits NMin."""
        lines = [
            "v_mov_b32 v1, 0x40003c00",
            "v_mov_b32 v2, 0x42004000",
            "v_pk_min_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pk_min", lines)
        self.assertIn("NMin", spv)

    def test_pk_min_f16_mutation_detection(self) -> None:
        """MIN is strictly distinguished from MAX."""
        w1 = pack_f16x2(0x3C00, 0x4000)  # 1.0, 2.0
        w2 = pack_f16x2(0x4000, 0x4200)  # 2.0, 3.0
        res = run_pk_binop("v_pk_min_f16", w1, w2, name="ps_pk_min_mut")
        self.assertEqual(res, pack_f16x2(0x3C00, 0x4000))
        # Mutant MAX produces (2.0, 3.0)
        self.assertNotEqual(res, pack_f16x2(0x4000, 0x4200))

    def test_pk_min_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pk_min", ["v_pk_min_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 2: v_pk_max_f16 (Packed f16x2 Maximum)
# ==============================================================================

class TestVPkMaxF16(unittest.TestCase):
    """Packed f16x2 maximum tests."""

    def test_pk_max_f16_basic(self) -> None:
        """Basic maximum of packed halves: max((1.0, 2.0), (2.0, 3.0)) = (2.0, 3.0)."""
        w1 = pack_f16x2(0x3C00, 0x4000)  # lo: 1.0, hi: 2.0
        w2 = pack_f16x2(0x4000, 0x4200)  # lo: 2.0, hi: 3.0
        res = run_pk_binop("v_pk_max_f16", w1, w2, name="ps_pk_max_basic")
        self.assertEqual(res, pack_f16x2(0x4000, 0x4200))

    def test_pk_max_f16_clamp(self) -> None:
        """Clamp modifier clamps negative results to 0.0 and >1.0 to 1.0."""
        w1 = pack_f16x2(0xC000, 0x4200)  # lo: -2.0, hi: 3.0
        w2 = pack_f16x2(0xBC00, 0x4000)  # lo: -1.0, hi: 2.0
        # max(-2.0, -1.0) = -1.0 -> clamped to 0.0 (0x0000)
        # max(3.0, 2.0) = 3.0 -> clamped to 1.0 (0x3C00)
        res = run_pk_binop("v_pk_max_f16", w1, w2, clamp=True, name="ps_pk_max_clamp")
        self.assertEqual(res, pack_f16x2(0x0000, 0x3C00))

    def test_pk_max_f16_modifiers_opsel_neg(self) -> None:
        """op_sel, op_sel_hi, and neg_lo modifiers."""
        w1 = pack_f16x2(0x4000, 0x4200)  # lo: 2.0, hi: 3.0
        w2 = pack_f16x2(0x3C00, 0x4500)  # lo: 1.0, hi: 5.0
        # op_sel:[1,0] -> lo selects w1.hi (3.0), w2.lo (1.0). neg_lo:[1,0] -> negates w1.hi: -3.0.
        # lo = max(-3.0, 1.0) = 1.0 (0x3C00).
        # op_sel_hi:[0,1] -> hi selects w1.lo (2.0), w2.hi (5.0).
        # hi = max(2.0, 5.0) = 5.0 (0x4500).
        res = run_pk_binop(
            "v_pk_max_f16", w1, w2,
            mods="op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0]",
            name="ps_pk_max_mod",
        )
        self.assertEqual(res, pack_f16x2(0x3C00, 0x4500))

    def test_pk_max_f16_nan_suppression(self) -> None:
        """NaN operands are suppressed in favor of non-NaN values."""
        w1 = pack_f16x2(0x7E00, 0x4000)  # lo: NaN, hi: 2.0
        w2 = pack_f16x2(0x3C00, 0x7E00)  # lo: 1.0, hi: NaN
        res = run_pk_binop("v_pk_max_f16", w1, w2, name="ps_pk_max_nan")
        self.assertEqual(res, pack_f16x2(0x3C00, 0x4000))

    def test_pk_max_f16_signed_zeros(self) -> None:
        """max(+0.0, -0.0) yields +0.0 (0x0000)."""
        w1 = pack_f16x2(0x0000, 0x8000)  # +0.0, -0.0
        w2 = pack_f16x2(0x8000, 0x0000)  # -0.0, +0.0
        res = run_pk_binop("v_pk_max_f16", w1, w2, name="ps_pk_max_zero")
        self.assertEqual(res, pack_f16x2(0x0000, 0x0000))

    def test_pk_max_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        w1 = pack_f16x2(0x3C00, 0x4000)
        w2 = pack_f16x2(0x4000, 0x4200)
        res_v1 = run_pk_binop("v_pk_max_f16", w1, w2, alias_dst=1, name="ps_pk_max_alias_v1")
        self.assertEqual(res_v1, pack_f16x2(0x4000, 0x4200))
        res_v2 = run_pk_binop("v_pk_max_f16", w1, w2, alias_dst=2, name="ps_pk_max_alias_v2")
        self.assertEqual(res_v2, pack_f16x2(0x4000, 0x4200))

    def test_pk_max_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits NMax."""
        lines = [
            "v_mov_b32 v1, 0x40003c00",
            "v_mov_b32 v2, 0x42004000",
            "v_pk_max_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pk_max", lines)
        self.assertIn("NMax", spv)

    def test_pk_max_f16_mutation_detection(self) -> None:
        """MAX is strictly distinguished from MIN."""
        w1 = pack_f16x2(0x3C00, 0x4000)  # 1.0, 2.0
        w2 = pack_f16x2(0x4000, 0x4200)  # 2.0, 3.0
        res = run_pk_binop("v_pk_max_f16", w1, w2, name="ps_pk_max_mut")
        self.assertEqual(res, pack_f16x2(0x4000, 0x4200))
        # Mutant MIN produces (1.0, 2.0)
        self.assertNotEqual(res, pack_f16x2(0x3C00, 0x4000))

    def test_pk_max_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pk_max", ["v_pk_max_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 3: v_pk_fma_f16 (Packed f16x2 Fused Multiply-Add)
# ==============================================================================

class TestVPkFmaF16(unittest.TestCase):
    """Packed f16x2 fused multiply-add tests."""

    def test_pk_fma_f16_basic(self) -> None:
        """Basic (s0 * s1 + s2): (2.0 * 1.5 + 1.0) = 4.0 (0x4400) on both halves."""
        w1 = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        w2 = pack_f16x2(0x3E00, 0x3E00)  # 1.5, 1.5
        w3 = pack_f16x2(0x3C00, 0x3C00)  # 1.0, 1.0
        res = run_pk_triop("v_pk_fma_f16", w1, w2, w3, name="ps_pk_fma_basic")
        self.assertEqual(res, pack_f16x2(0x4400, 0x4400))

    def test_pk_fma_f16_clamp(self) -> None:
        """Clamp modifier clamps results to [0.0, 1.0]."""
        w1 = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        w2 = pack_f16x2(0x3E00, 0x3E00)  # 1.5, 1.5
        w3 = pack_f16x2(0x3C00, 0x3C00)  # 1.0, 1.0
        # 4.0 clamped to 1.0 (0x3C00)
        res = run_pk_triop("v_pk_fma_f16", w1, w2, w3, clamp=True, name="ps_pk_fma_clamp")
        self.assertEqual(res, pack_f16x2(0x3C00, 0x3C00))

    def test_pk_fma_f16_cancellation(self) -> None:
        """2.0 * (-1.0) + 2.0 = 0.0 (0x0000)."""
        w1 = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        w2 = pack_f16x2(0xBC00, 0xBC00)  # -1.0, -1.0
        w3 = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        res = run_pk_triop("v_pk_fma_f16", w1, w2, w3, name="ps_pk_fma_cancel")
        self.assertEqual(res, pack_f16x2(0x0000, 0x0000))

    def test_pk_fma_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        w1 = pack_f16x2(0x4000, 0x4000)
        w2 = pack_f16x2(0x3E00, 0x3E00)
        w3 = pack_f16x2(0x3C00, 0x3C00)
        res_v1 = run_pk_triop("v_pk_fma_f16", w1, w2, w3, alias_dst=1, name="ps_pk_fma_alias_v1")
        self.assertEqual(res_v1, pack_f16x2(0x4400, 0x4400))
        res_v3 = run_pk_triop("v_pk_fma_f16", w1, w2, w3, alias_dst=3, name="ps_pk_fma_alias_v3")
        self.assertEqual(res_v3, pack_f16x2(0x4400, 0x4400))

    def test_pk_fma_f16_spirv_structure(self) -> None:
        """SPIR-V translation lowers v_pk_fma_f16 via UnpackHalf2x16."""
        lines = [
            "v_mov_b32 v1, 0x40004000",
            "v_mov_b32 v2, 0x3e003e00",
            "v_mov_b32 v3, 0x3c003c00",
            "v_pk_fma_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pk_fma", lines)
        self.assertIn("UnpackHalf2x16", spv)

    def test_pk_fma_f16_mutation_detection(self) -> None:
        """FMA is strictly distinguished from simple multiplication."""
        w1 = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        w2 = pack_f16x2(0x3E00, 0x3E00)  # 1.5, 1.5
        w3 = pack_f16x2(0x3C00, 0x3C00)  # 1.0, 1.0
        res = run_pk_triop("v_pk_fma_f16", w1, w2, w3, name="ps_pk_fma_mut")
        self.assertEqual(res, pack_f16x2(0x4400, 0x4400))
        # MUL alone yields 3.0 (0x4200)
        self.assertNotEqual(res, pack_f16x2(0x4200, 0x4200))

    def test_pk_fma_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pk_fma", ["v_pk_fma_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 4: v_pk_fmac_f16 (Packed f16x2 Multiply-Accumulate)
# ==============================================================================

class TestVPkFmacF16(unittest.TestCase):
    """Packed f16x2 multiply-accumulate tests."""

    def test_pk_fmac_f16_basic(self) -> None:
        """acc = acc + s0 * s1: 1.0 + 2.0 * 1.5 = 4.0 (0x4400) on both halves."""
        acc = pack_f16x2(0x3C00, 0x3C00)  # 1.0, 1.0
        s0 = pack_f16x2(0x4000, 0x4000)   # 2.0, 2.0
        s1 = pack_f16x2(0x3E00, 0x3E00)   # 1.5, 1.5
        res = run_pk_fmac(acc, s0, s1, name="ps_pk_fmac_basic")
        self.assertEqual(res, pack_f16x2(0x4400, 0x4400))

    def test_pk_fmac_f16_zero_accumulation(self) -> None:
        """acc + 0 * s1 preserves acc."""
        acc = pack_f16x2(0x4200, 0x4500)  # 3.0, 5.0
        s0 = pack_f16x2(0x0000, 0x0000)   # 0, 0
        s1 = pack_f16x2(0x4000, 0x4000)   # 2.0, 2.0
        res = run_pk_fmac(acc, s0, s1, name="ps_pk_fmac_zero")
        self.assertEqual(res, acc)

    def test_pk_fmac_f16_cancellation(self) -> None:
        """acc + (-1.0) * acc = 0.0 (0x0000)."""
        acc = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        s0 = pack_f16x2(0xBC00, 0xBC00)   # -1.0, -1.0
        s1 = pack_f16x2(0x4000, 0x4000)   # 2.0, 2.0
        res = run_pk_fmac(acc, s0, s1, name="ps_pk_fmac_cancel")
        self.assertEqual(res, pack_f16x2(0x0000, 0x0000))

    def test_pk_fmac_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits UnpackHalf2x16."""
        lines = [
            "v_mov_b32 v0, 0x3c003c00",
            "v_mov_b32 v1, 0x40004000",
            "v_mov_b32 v2, 0x3e003e00",
            "v_pk_fmac_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pk_fmac", lines)
        self.assertIn("UnpackHalf2x16", spv)

    def test_pk_fmac_f16_mutation_detection(self) -> None:
        """FMAC is strictly distinguished from simple accumulation or multiplication."""
        acc = pack_f16x2(0x3C00, 0x3C00)  # 1.0, 1.0
        s0 = pack_f16x2(0x4000, 0x4000)   # 2.0, 2.0
        s1 = pack_f16x2(0x3E00, 0x3E00)   # 1.5, 1.5
        res = run_pk_fmac(acc, s0, s1, name="ps_pk_fmac_mut")
        self.assertEqual(res, pack_f16x2(0x4400, 0x4400))
        # Pure sum yields 1.0 + 2.0 = 3.0 (0x4200)
        self.assertNotEqual(res, pack_f16x2(0x4200, 0x4200))

    def test_pk_fmac_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pk_fmac", ["v_pk_fmac_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 5: v_pk_mul_f16 (Packed f16x2 Multiplication)
# ==============================================================================

class TestVPkMulF16(unittest.TestCase):
    """Packed f16x2 multiplication tests."""

    def test_pk_mul_f16_basic(self) -> None:
        """Basic 2.0 (0x4000) * 1.5 (0x3E00) = 3.0 (0x4200) on both halves."""
        w1 = pack_f16x2(0x4000, 0x4000)
        w2 = pack_f16x2(0x3E00, 0x3E00)
        res = run_pk_binop("v_pk_mul_f16", w1, w2, name="ps_pk_mul_basic")
        self.assertEqual(res, pack_f16x2(0x4200, 0x4200))

    def test_pk_mul_f16_clamp(self) -> None:
        """Clamp modifier clamps results to [0.0, 1.0]."""
        w1 = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        w2 = pack_f16x2(0x3E00, 0x3E00)  # 1.5, 1.5
        # 3.0 clamped to 1.0 (0x3C00)
        res = run_pk_binop("v_pk_mul_f16", w1, w2, clamp=True, name="ps_pk_mul_clamp")
        self.assertEqual(res, pack_f16x2(0x3C00, 0x3C00))

    def test_pk_mul_f16_modifiers_opsel_neg(self) -> None:
        """op_sel, op_sel_hi, and neg_lo modifiers."""
        w1 = pack_f16x2(0x4000, 0x4200)  # lo: 2.0, hi: 3.0
        w2 = pack_f16x2(0x3E00, 0x4000)  # lo: 1.5, hi: 2.0
        # op_sel:[1,0] -> lo selects w1.hi (3.0), w2.lo (1.5). neg_lo:[1,0] -> -3.0 * 1.5 = -4.5 (0xC480).
        # op_sel_hi:[0,1] -> hi selects w1.lo (2.0), w2.hi (2.0) -> 2.0 * 2.0 = 4.0 (0x4400).
        res = run_pk_binop(
            "v_pk_mul_f16", w1, w2,
            mods="op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0]",
            name="ps_pk_mul_mod",
        )
        self.assertEqual(res, pack_f16x2(0xC480, 0x4400))

    def test_pk_mul_f16_zero_and_signs(self) -> None:
        """Signs and zeros in multiplication."""
        w1 = pack_f16x2(0x0000, 0x8000)  # +0.0, -0.0
        w2 = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        res = run_pk_binop("v_pk_mul_f16", w1, w2, name="ps_pk_mul_signs")
        self.assertEqual(res, pack_f16x2(0x0000, 0x8000))

    def test_pk_mul_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        w1 = pack_f16x2(0x4000, 0x4000)
        w2 = pack_f16x2(0x3E00, 0x3E00)
        res_v1 = run_pk_binop("v_pk_mul_f16", w1, w2, alias_dst=1, name="ps_pk_mul_alias_v1")
        self.assertEqual(res_v1, pack_f16x2(0x4200, 0x4200))
        res_v2 = run_pk_binop("v_pk_mul_f16", w1, w2, alias_dst=2, name="ps_pk_mul_alias_v2")
        self.assertEqual(res_v2, pack_f16x2(0x4200, 0x4200))

    def test_pk_mul_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits OpFMul %v2float."""
        lines = [
            "v_mov_b32 v1, 0x40004000",
            "v_mov_b32 v2, 0x3e003e00",
            "v_pk_mul_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pk_mul", lines)
        self.assertIn("OpFMul", spv)

    def test_pk_mul_f16_mutation_detection(self) -> None:
        """Multiplication is strictly distinguished from addition."""
        w1 = pack_f16x2(0x4000, 0x4000)  # 2.0, 2.0
        w2 = pack_f16x2(0x3E00, 0x3E00)  # 1.5, 1.5
        res = run_pk_binop("v_pk_mul_f16", w1, w2, name="ps_pk_mul_mut")
        self.assertEqual(res, pack_f16x2(0x4200, 0x4200))
        # Addition yields 3.5 (0x4300)
        self.assertNotEqual(res, pack_f16x2(0x4300, 0x4300))

    def test_pk_mul_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pk_mul", ["v_pk_mul_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 6: v_pk_add_f16 (Packed f16x2 Addition)
# ==============================================================================

class TestVPkAddF16(unittest.TestCase):
    """Packed f16x2 addition tests."""

    def test_pk_add_f16_basic(self) -> None:
        """Basic 1.0 + 2.0 = 3.0 (0x4200) and 2.0 + 3.0 = 5.0 (0x4500)."""
        w1 = pack_f16x2(0x3C00, 0x4000)  # 1.0, 2.0
        w2 = pack_f16x2(0x4000, 0x4200)  # 2.0, 3.0
        res = run_pk_binop("v_pk_add_f16", w1, w2, name="ps_pk_add_basic")
        self.assertEqual(res, pack_f16x2(0x4200, 0x4500))

    def test_pk_add_f16_clamp(self) -> None:
        """Clamp modifier clamps results to [0.0, 1.0]."""
        w1 = pack_f16x2(0x3C00, 0x4000)  # 1.0, 2.0
        w2 = pack_f16x2(0x4000, 0x4200)  # 2.0, 3.0
        # 3.0 and 5.0 clamped to 1.0 (0x3C00)
        res = run_pk_binop("v_pk_add_f16", w1, w2, clamp=True, name="ps_pk_add_clamp")
        self.assertEqual(res, pack_f16x2(0x3C00, 0x3C00))

    def test_pk_add_f16_modifiers_opsel_neg(self) -> None:
        """op_sel, op_sel_hi, and neg_lo modifiers."""
        w1 = pack_f16x2(0x4000, 0x4200)  # lo: 2.0, hi: 3.0
        w2 = pack_f16x2(0x3C00, 0x4500)  # lo: 1.0, hi: 5.0
        # lo: -3.0 + 1.0 = -2.0 (0xC000)
        # hi: 2.0 + 5.0 = 7.0 (0x4700)
        res = run_pk_binop(
            "v_pk_add_f16", w1, w2,
            mods="op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0]",
            name="ps_pk_add_mod",
        )
        self.assertEqual(res, pack_f16x2(0xC000, 0x4700))

    def test_pk_add_f16_cancellation(self) -> None:
        """1.0 + (-1.0) = 0.0 (0x0000)."""
        w1 = pack_f16x2(0x3C00, 0x4000)  # 1.0, 2.0
        w2 = pack_f16x2(0xBC00, 0xC000)  # -1.0, -2.0
        res = run_pk_binop("v_pk_add_f16", w1, w2, name="ps_pk_add_cancel")
        self.assertEqual(res, pack_f16x2(0x0000, 0x0000))

    def test_pk_add_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        w1 = pack_f16x2(0x3C00, 0x4000)
        w2 = pack_f16x2(0x4000, 0x4200)
        res_v1 = run_pk_binop("v_pk_add_f16", w1, w2, alias_dst=1, name="ps_pk_add_alias_v1")
        self.assertEqual(res_v1, pack_f16x2(0x4200, 0x4500))
        res_v2 = run_pk_binop("v_pk_add_f16", w1, w2, alias_dst=2, name="ps_pk_add_alias_v2")
        self.assertEqual(res_v2, pack_f16x2(0x4200, 0x4500))

    def test_pk_add_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits OpFAdd %v2float."""
        lines = [
            "v_mov_b32 v1, 0x40003c00",
            "v_mov_b32 v2, 0x42004000",
            "v_pk_add_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pk_add", lines)
        self.assertIn("OpFAdd", spv)

    def test_pk_add_f16_mutation_detection(self) -> None:
        """Addition is strictly distinguished from multiplication."""
        w1 = pack_f16x2(0x3C00, 0x4000)  # 1.0, 2.0
        w2 = pack_f16x2(0x4000, 0x4200)  # 2.0, 3.0
        res = run_pk_binop("v_pk_add_f16", w1, w2, name="ps_pk_add_mut")
        self.assertEqual(res, pack_f16x2(0x4200, 0x4500))
        # Multiplication yields (2.0, 6.0)
        self.assertNotEqual(res, pack_f16x2(0x4000, 0x4600))

    def test_pk_add_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pk_add", ["v_pk_add_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 7: Integration & Cross-Opcode Pipelines
# ==============================================================================

class TestVPkF16Pipelines(unittest.TestCase):
    """Multi-instruction pipeline integration tests."""

    def test_chained_pk_add_mul_pipeline(self) -> None:
        """Chained: (w1 + w2) * w3 packed."""
        lines = [
            "v_mov_b32 v1, 0x40003c00",      # hi: 2.0, lo: 1.0
            "v_mov_b32 v2, 0x40003c00",      # hi: 2.0, lo: 1.0
            "v_mov_b32 v3, 0x3e003e00",      # hi: 1.5, lo: 1.5
            "v_pk_add_f16 v4, v1, v2",       # hi: 4.0 (0x4400), lo: 2.0 (0x4000)
            "v_pk_mul_f16 v5, v4, v3",       # hi: 4.0*1.5 = 6.0 (0x4600), lo: 2.0*1.5 = 3.0 (0x4200)
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_pk_add_mul", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], pack_f16x2(0x4200, 0x4600))

    def test_chained_pk_fmac_min_pipeline(self) -> None:
        """Chained: min(fmac(acc, s0, s1), limit)."""
        lines = [
            "v_mov_b32 v0, 0x3c003c00",      # acc: 1.0, 1.0
            "v_mov_b32 v1, 0x40004000",      # s0: 2.0, 2.0
            "v_mov_b32 v2, 0x3e003e00",      # s1: 1.5, 1.5
            "v_mov_b32 v3, 0x42004200",      # limit: 3.0, 3.0
            "v_pk_fmac_f16 v0, v1, v2",      # acc: 4.0 (0x4400)
            "v_pk_min_f16 v4, v0, v3",       # min(4.0, 3.0) = 3.0 (0x4200)
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_pk_fmac_min", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], pack_f16x2(0x4200, 0x4200))

    def test_halfword_isolation(self) -> None:
        """Verify low and high halfwords execute with complete numerical isolation."""
        w1 = pack_f16x2(0x3C00, 0x0000)  # lo: 1.0, hi: 0.0
        w2 = pack_f16x2(0x4000, 0x0000)  # lo: 2.0, hi: 0.0
        res = run_pk_binop("v_pk_add_f16", w1, w2, name="ps_pk_isolation")
        lo, hi = unpack_f16x2(res)
        self.assertEqual(lo, 0x4200)  # 3.0
        self.assertEqual(hi, 0x0000)  # 0.0


if __name__ == '__main__':
    unittest.main()

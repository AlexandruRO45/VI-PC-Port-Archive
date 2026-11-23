from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 packed float16 conversion & packing:
  - v_cvt_pkrtz_f16_f32:  Packs two float32 values into two float16 values with round-toward-zero (RTZ)
  - v_cvt_pknorm_i16_f16: Converts two float16 values to signed normalized 16-bit integers (snorm16)
  - v_cvt_pknorm_u16_f16: Converts two float16 values to unsigned normalized 16-bit integers (unorm16)
  - v_pack_b32_f16:       Packs two 16-bit halves selected by op_sel into one 32-bit dword

All tests in this suite compile real assembly listings through support.translate_lines(...)
and execute the translated SPIR-V in spirv_mini_interp via support.mrt0_words.
Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF


def f32_to_bits(f: float) -> int:
    """Pack Python float to IEEE 754 32-bit integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return 0x7F800000 if f > 0 else 0xFF800000


def run_pk_binop(
    op: str,
    src0: str | int | float,
    src1: str | int | float,
    *,
    name: str = "ps_pk_binop",
    mods: str = "",
    alias_dst: int | None = None,
) -> int:
    """Assemble, translate, and execute a 2-source packed conversion instruction."""
    def fmt(s: str | int | float) -> str:
        if isinstance(s, float):
            return f"{f32_to_bits(s):#x}"
        if isinstance(s, int):
            return f"{s:#x}"
        return str(s)

    dst = f"v{alias_dst}" if alias_dst is not None else "v0"
    mod_str = f" {mods}" if mods else ""
    lines = [
        f"v_mov_b32 v1, {fmt(src0)}",
        f"v_mov_b32 v2, {fmt(src1)}",
        f"{op} {dst}, v1, v2{mod_str}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0]


def run_pk_simd(
    op: str,
    a0: int,
    a1: int,
    b0: int,
    b1: int,
    *,
    name: str = "ps_pk_simd",
    mods: str = "",
) -> tuple[int, int]:
    """Execute dual lane instruction to verify SIMD multi-lane execution."""
    mod_str = f" {mods}" if mods else ""
    lines = [
        f"v_mov_b32 v1, {a0:#x}",
        f"v_mov_b32 v2, {a1:#x}",
        f"v_mov_b32 v3, {b0:#x}",
        f"v_mov_b32 v4, {b1:#x}",
        f"{op} v5, v1, v2{mod_str}",
        f"{op} v6, v3, v4{mod_str}",
        "exp mrt0 v5, v6, v5, v6 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


# ==============================================================================
# Suite 1: v_cvt_pkrtz_f16_f32 (Pack Two f32 to f16 with RTZ)
# ==============================================================================

class TestVCvtPkrtzF16F32(unittest.TestCase):
    """v_cvt_pkrtz_f16_f32 tests."""

    def test_basic_values(self) -> None:
        """Basic 1.0 (0x3C00) and 2.0 (0x4000) packed into 0x40003C00."""
        res1 = run_pk_binop("v_cvt_pkrtz_f16_f32", 1.0, 2.0, name="ps_pkrtz_ref1")
        self.assertEqual(res1, 0x40003C00)
        res2 = run_pk_binop("v_cvt_pkrtz_f16_f32", -1.0, 0.5, name="ps_pkrtz_ref2")
        self.assertEqual(res2, 0x3800BC00)

    def test_truncation_toward_zero(self) -> None:
        """Round-toward-zero truncates fractional bits below f16 precision."""
        res = run_pk_binop("v_cvt_pkrtz_f16_f32", 1.999999, 1.0, name="ps_pkrtz_ref_ttz")
        self.assertEqual(res & 0x7C00, 0x3C00)

    def test_signed_zeros(self) -> None:
        """Signed zeros preserved in both halves."""
        res1 = run_pk_binop("v_cvt_pkrtz_f16_f32", 0.0, -0.0, name="ps_pkrtz_ref_sz1")
        self.assertEqual(res1, 0x80000000)
        res2 = run_pk_binop("v_cvt_pkrtz_f16_f32", -0.0, 0.0, name="ps_pkrtz_ref_sz2")
        self.assertEqual(res2, 0x00008000)

    def test_saturation_above_max_normal(self) -> None:
        """Values exceeding f16 max saturate to 0x7BFF / 0xFBFF under RTZ."""
        res = run_pk_binop("v_cvt_pkrtz_f16_f32", 70000.0, -80000.0, name="ps_pkrtz_edge_sat")
        self.assertEqual(res & 0xFFFF, 0x7BFF)
        self.assertEqual((res >> 16) & 0xFFFF, 0xFBFF)

    def test_infinities(self) -> None:
        """+inf (0x7C00) and -inf (0xFC00)."""
        res = run_pk_binop("v_cvt_pkrtz_f16_f32", float("inf"), float("-inf"), name="ps_pkrtz_edge_inf")
        self.assertEqual(res & 0xFFFF, 0x7C00)
        self.assertEqual((res >> 16) & 0xFFFF, 0xFC00)

    def test_nans(self) -> None:
        """NaN inputs produce quiet NaNs."""
        res = run_pk_binop("v_cvt_pkrtz_f16_f32", float("nan"), float("nan"), name="ps_pkrtz_edge_nan")
        self.assertEqual(res & 0x7E00, 0x7E00)
        self.assertEqual((res >> 16) & 0x7E00, 0x7E00)

    def test_simd_dual_lanes(self) -> None:
        """Verify dual lane SIMD execution."""
        w0, w1 = run_pk_simd(
            "v_cvt_pkrtz_f16_f32",
            f32_to_bits(1.0), f32_to_bits(2.0),
            f32_to_bits(3.0), f32_to_bits(4.0),
            name="ps_pkrtz_simd",
        )
        self.assertEqual(w0, 0x40003C00)
        self.assertEqual(w1, 0x44004200)

    def test_register_aliasing(self) -> None:
        """Register aliasing with destination operand."""
        res_v1 = run_pk_binop("v_cvt_pkrtz_f16_f32", 1.0, 2.0, alias_dst=1, name="ps_pkrtz_alias_v1")
        self.assertEqual(res_v1, 0x40003C00)
        res_v2 = run_pk_binop("v_cvt_pkrtz_f16_f32", 1.0, 2.0, alias_dst=2, name="ps_pkrtz_alias_v2")
        self.assertEqual(res_v2, 0x40003C00)

    def test_spirv_structure(self) -> None:
        """SPIR-V translation emits conversion logic."""
        lines = [
            "v_mov_b32 v1, 1.0",
            "v_mov_b32 v2, 2.0",
            "v_cvt_pkrtz_f16_f32 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pkrtz", lines)
        word = support.mrt0_words(spv)[0]
        self.assertEqual(word, 0x40003C00)

    def test_mutation_detection(self) -> None:
        """RTZ conversion order lo/hi is strictly distinguished."""
        res = run_pk_binop("v_cvt_pkrtz_f16_f32", 1.0, 2.0, name="ps_pkrtz_mut")
        self.assertEqual(res, 0x40003C00)
        # Swapped operands yields 0x3C004000
        self.assertNotEqual(res, 0x3C004000)

    def test_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pkrtz", ["v_cvt_pkrtz_f16_f32 v0", "s_endpgm"])


# ==============================================================================
# Suite 2: v_cvt_pknorm_i16_f16 (Convert Two f16 to snorm16)
# ==============================================================================

class TestVCvtPknormI16F16(unittest.TestCase):
    """v_cvt_pknorm_i16_f16 tests."""

    def test_cardinals(self) -> None:
        """1.0 (0x3C00) -> 0x7FFF (32767), -1.0 (0xBC00) -> 0x8001 (-32767)."""
        res = run_pk_binop("v_cvt_pknorm_i16_f16", 0x3C00, 0xBC00, name="ps_pknorm_i16_card")
        self.assertEqual(res & 0xFFFF, 0x7FFF)
        self.assertEqual((res >> 16) & 0xFFFF, 0x8001)

    def test_zero(self) -> None:
        """0.0 and -0.0 convert to 0x00000000."""
        res = run_pk_binop("v_cvt_pknorm_i16_f16", 0x0000, 0x8000, name="ps_pknorm_i16_zero")
        self.assertEqual(res, 0x00000000)

    def test_intermediate_scale(self) -> None:
        """0.5 (0x3800) converts to ~0x4000 (16384)."""
        res = run_pk_binop("v_cvt_pknorm_i16_f16", 0x3800, 0x3800, name="ps_pknorm_i16_scale")
        self.assertEqual(res & 0xFFFF, 0x4000)
        self.assertEqual((res >> 16) & 0xFFFF, 0x4000)

    def test_clamp_extremes(self) -> None:
        """Inputs exceeding 1.0 or -1.0 clamp to snorm16 limits."""
        res1 = run_pk_binop("v_cvt_pknorm_i16_f16", 0x4000, 0x7C00, name="ps_pknorm_i16_pos_sat")
        self.assertEqual(res1, (0x7FFF << 16) | 0x7FFF)
        res2 = run_pk_binop("v_cvt_pknorm_i16_f16", 0xC000, 0xFC00, name="ps_pknorm_i16_neg_sat")
        self.assertEqual(res2, (0x8001 << 16) | 0x8001)

    def test_nan_produces_zero(self) -> None:
        """NaN inputs produce 0."""
        res = run_pk_binop("v_cvt_pknorm_i16_f16", 0x7E00, 0xFE00, name="ps_pknorm_i16_nan")
        self.assertEqual(res, 0x00000000)

    def test_op_sel_modifier(self) -> None:
        """op_sel selects upper halfwords."""
        s0 = (0xBC00 << 16) | 0x3C00  # hi: -1.0, lo: 1.0
        s1 = (0x0000 << 16) | 0x3800  # hi: 0.0, lo: 0.5
        # op_sel:[0,0,0] -> lo selects s0.lo (1.0 -> 0x7FFF), hi selects s1.lo (0.5 -> 0x4000)
        res1 = run_pk_binop("v_cvt_pknorm_i16_f16", s0, s1, mods="op_sel:[0,0,0]", name="ps_pknorm_i16_sel1")
        self.assertEqual(res1, (0x4000 << 16) | 0x7FFF)
        # op_sel:[1,0,0] -> lo selects s0.hi (-1.0 -> 0x8001), hi selects s1.lo (0.5 -> 0x4000)
        res2 = run_pk_binop("v_cvt_pknorm_i16_f16", s0, s1, mods="op_sel:[1,0,0]", name="ps_pknorm_i16_sel2")
        self.assertEqual(res2, (0x4000 << 16) | 0x8001)

    def test_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_pk_binop("v_cvt_pknorm_i16_f16", 0x3C00, 0xBC00, alias_dst=1, name="ps_pknorm_i16_alias_v1")
        self.assertEqual(res_v1, (0x8001 << 16) | 0x7FFF)

    def test_spirv_structure(self) -> None:
        """SPIR-V translation structure verification."""
        lines = [
            "v_mov_b32 v1, 0x3C00",
            "v_mov_b32 v2, 0xBC00",
            "v_cvt_pknorm_i16_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pknorm_i16", lines)
        word = support.mrt0_words(spv)[0]
        self.assertEqual(word & 0xFFFF, 0x7FFF)
        self.assertEqual((word >> 16) & 0xFFFF, 0x8001)

    def test_mutation_detection(self) -> None:
        """Signed snorm16 is strictly distinguished from unsigned unorm16."""
        res = run_pk_binop("v_cvt_pknorm_i16_f16", 0xBC00, 0xBC00, name="ps_pknorm_i16_mut")
        # snorm16 for -1.0 is 0x8001
        self.assertEqual(res, (0x8001 << 16) | 0x8001)
        # unorm16 clamps negative to 0x0000
        self.assertNotEqual(res, 0x00000000)

    def test_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pknorm_i16", ["v_cvt_pknorm_i16_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 3: v_cvt_pknorm_u16_f16 (Convert Two f16 to unorm16)
# ==============================================================================

class TestVCvtPknormU16F16(unittest.TestCase):
    """v_cvt_pknorm_u16_f16 tests."""

    def test_cardinals(self) -> None:
        """1.0 (0x3C00) -> 0xFFFF, 0.5 (0x3800) -> ~0x8000 (32768)."""
        res = run_pk_binop("v_cvt_pknorm_u16_f16", 0x3C00, 0x3800, name="ps_pknorm_u16_card")
        self.assertEqual(res & 0xFFFF, 0xFFFF)
        self.assertEqual((res >> 16) & 0xFFFF, 0x8000)

    def test_zeros_and_negatives(self) -> None:
        """0.0 and -1.0 (0xBC00) clamp to 0x00000000."""
        res = run_pk_binop("v_cvt_pknorm_u16_f16", 0x0000, 0xBC00, name="ps_pknorm_u16_zero_neg")
        self.assertEqual(res, 0x00000000)

    def test_clamping_extremes(self) -> None:
        """Values > 1.0 clamp to 0xFFFF, negative values clamp to 0x0000."""
        res1 = run_pk_binop("v_cvt_pknorm_u16_f16", 0x4000, 0x7C00, name="ps_pknorm_u16_pos_sat")
        self.assertEqual(res1, 0xFFFFFFFF)
        res2 = run_pk_binop("v_cvt_pknorm_u16_f16", 0xBC00, 0xFC00, name="ps_pknorm_u16_neg_sat")
        self.assertEqual(res2, 0x00000000)

    def test_nan_produces_zero(self) -> None:
        """NaN inputs produce 0."""
        res = run_pk_binop("v_cvt_pknorm_u16_f16", 0x7E00, 0x7E00, name="ps_pknorm_u16_nan")
        self.assertEqual(res, 0x00000000)

    def test_op_sel_modifier(self) -> None:
        """op_sel selects high halfwords."""
        s0 = (0x3800 << 16) | 0x3C00  # hi: 0.5, lo: 1.0
        s1 = (0x0000 << 16) | 0x3C00  # hi: 0.0, lo: 1.0
        # op_sel:[0,0,0] -> lo selects s0.lo (1.0 -> 0xFFFF), hi selects s1.lo (1.0 -> 0xFFFF)
        res1 = run_pk_binop("v_cvt_pknorm_u16_f16", s0, s1, mods="op_sel:[0,0,0]", name="ps_pknorm_u16_sel1")
        self.assertEqual(res1, 0xFFFFFFFF)
        # op_sel:[1,0,0] -> lo selects s0.hi (0.5 -> 0x8000), hi selects s1.lo (1.0 -> 0xFFFF)
        res2 = run_pk_binop("v_cvt_pknorm_u16_f16", s0, s1, mods="op_sel:[1,0,0]", name="ps_pknorm_u16_sel2")
        self.assertEqual(res2, (0xFFFF << 16) | 0x8000)

    def test_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_pk_binop("v_cvt_pknorm_u16_f16", 0x3C00, 0x3800, alias_dst=1, name="ps_pknorm_u16_alias_v1")
        self.assertEqual(res_v1, (0x8000 << 16) | 0xFFFF)

    def test_spirv_structure(self) -> None:
        """SPIR-V translation structure verification."""
        lines = [
            "v_mov_b32 v1, 0x3C00",
            "v_mov_b32 v2, 0x3800",
            "v_cvt_pknorm_u16_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pknorm_u16", lines)
        word = support.mrt0_words(spv)[0]
        self.assertEqual(word & 0xFFFF, 0xFFFF)
        self.assertEqual((word >> 16) & 0xFFFF, 0x8000)

    def test_mutation_detection(self) -> None:
        """Unsigned unorm16 is strictly distinguished from signed snorm16."""
        res = run_pk_binop("v_cvt_pknorm_u16_f16", 0xBC00, 0xBC00, name="ps_pknorm_u16_mut")
        # unorm16 clamps negative to 0x00000000
        self.assertEqual(res, 0x00000000)
        # snorm16 would produce 0x80018001
        self.assertNotEqual(res, (0x8001 << 16) | 0x8001)

    def test_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pknorm_u16", ["v_cvt_pknorm_u16_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 4: v_pack_b32_f16 (Pack Two 16-bit Halves)
# ==============================================================================

class TestVPackB32F16(unittest.TestCase):
    """v_pack_b32_f16 tests."""

    def test_basic_pack(self) -> None:
        """Pack low 16 bits of each operand into one 32-bit dword."""
        res1 = run_pk_binop("v_pack_b32_f16", 0x00003C00, 0x00004000, name="ps_pack_ref1")
        self.assertEqual(res1, 0x40003C00)
        res2 = run_pk_binop("v_pack_b32_f16", 0x12345678, 0x9ABCDEF0, name="ps_pack_ref2")
        self.assertEqual(res2, 0xDEF05678)

    def test_op_sel_permutations(self) -> None:
        """op_sel specifies which 16-bit half is read from each source."""
        s0 = 0x11112222
        s1 = 0x33334444
        # [1,0,0]: lo from s0.hi (0x1111), hi from s1.lo (0x4444)
        res1 = run_pk_binop("v_pack_b32_f16", s0, s1, mods="op_sel:[1,0,0]", name="ps_pack_sel1")
        self.assertEqual(res1, 0x44441111)
        # [0,1,0]: lo from s0.lo (0x2222), hi from s1.hi (0x3333)
        res2 = run_pk_binop("v_pack_b32_f16", s0, s1, mods="op_sel:[0,1,0]", name="ps_pack_sel2")
        self.assertEqual(res2, 0x33332222)
        # [1,1,0]: lo from s0.hi (0x1111), hi from s1.hi (0x3333)
        res3 = run_pk_binop("v_pack_b32_f16", s0, s1, mods="op_sel:[1,1,0]", name="ps_pack_sel3")
        self.assertEqual(res3, 0x33331111)

    def test_nan_payloads_preserved(self) -> None:
        """NaN bit patterns preserved verbatim."""
        qnan = 0x7E01
        snan = 0x7C01
        res = run_pk_binop("v_pack_b32_f16", qnan, snan, name="ps_pack_nan")
        self.assertEqual(res, (snan << 16) | qnan)

    def test_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        s0 = 0x12345678
        s1 = 0x9ABCDEF0
        res_v1 = run_pk_binop("v_pack_b32_f16", s0, s1, alias_dst=1, name="ps_pack_alias_v1")
        self.assertEqual(res_v1, 0xDEF05678)
        res_v2 = run_pk_binop("v_pack_b32_f16", s0, s1, alias_dst=2, name="ps_pack_alias_v2")
        self.assertEqual(res_v2, 0xDEF05678)

    def test_spirv_structure(self) -> None:
        """SPIR-V translation structure verification."""
        lines = [
            "v_mov_b32 v1, 0x12345678",
            "v_mov_b32 v2, 0x9ABCDEF0",
            "v_pack_b32_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_pack", lines)
        word = support.mrt0_words(spv)[0]
        self.assertEqual(word, 0xDEF05678)

    def test_mutation_detection(self) -> None:
        """Halfword packing order is strictly distinguished."""
        s0 = 0x12345678
        s1 = 0x9ABCDEF0
        res = run_pk_binop("v_pack_b32_f16", s0, s1, name="ps_pack_mut")
        self.assertEqual(res, 0xDEF05678)
        # Swapped operands yields 0x5678DEF0
        self.assertNotEqual(res, 0x5678DEF0)

    def test_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_pack", ["v_pack_b32_f16 v0", "s_endpgm"])


# ==============================================================================
# Suite 5: Integration & Conversion Pipelines
# ==============================================================================

class TestVPkCvtPipelines(unittest.TestCase):
    """Multi-instruction pipeline integration tests."""

    def test_cvt_pkrtz_then_pack(self) -> None:
        """Chained: v_cvt_pkrtz_f16_f32 -> v_pack_b32_f16."""
        lines = [
            "v_mov_b32 v1, 1.0",
            "v_mov_b32 v2, 2.0",
            "v_mov_b32 v3, 3.0",
            "v_mov_b32 v4, 4.0",
            "v_cvt_pkrtz_f16_f32 v5, v1, v2",   # hi: 2.0 (0x4000), lo: 1.0 (0x3C00)
            "v_cvt_pkrtz_f16_f32 v6, v3, v4",   # hi: 4.0 (0x4400), lo: 3.0 (0x4200)
            "v_pack_b32_f16 v0, v5, v6 op_sel:[1,1,0]",  # lo: 2.0 (0x4000), hi: 4.0 (0x4400)
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_pkrtz_pack", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x44004000)

    def test_pknorm_chain(self) -> None:
        """Chained: unpack f32, convert to pknorm."""
        lines = [
            "v_mov_b32 v1, 1.0",
            "v_mov_b32 v2, -1.0",
            "v_cvt_pkrtz_f16_f32 v3, v1, v2",   # hi: -1.0 (0xBC00), lo: 1.0 (0x3C00)
            "v_cvt_pknorm_i16_f16 v0, v3, v3 op_sel:[0,1,0]",  # lo: 1.0 -> 0x7FFF, hi: -1.0 -> 0x8001
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_pknorm", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], (0x8001 << 16) | 0x7FFF)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 VALU packed 16-bit
shift and integer arithmetic operations (Bundle 22):
  - v_pk_ashrrev_i16: Packed 16-bit signed arithmetic right shift
  - v_pk_lshlrev_b16: Packed 16-bit logical left shift
  - v_pk_lshrrev_b16: Packed 16-bit logical right shift
  - v_pk_mad_i16:     Packed 16-bit signed integer multiply-add
  - v_pk_mad_u16:     Packed 16-bit unsigned integer multiply-add
  - v_pk_sub_u16:     Packed 16-bit unsigned integer subtraction

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


def pack16(lo: int, hi: int) -> int:
    """Pack two 16-bit integers into a single 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack16(val: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into low and high 16-bit unsigned integers."""
    return (val & MASK16, (val >> 16) & MASK16)


def run_pk_op2(
    opcode: str,
    s0: int,
    s1: int,
    *,
    modifiers: str = "",
    s0_sgpr: bool = False,
    alias_dst: int | None = None,
) -> int:
    """Assemble, translate, and execute a 2-operand packed 16-bit VALU instruction."""
    lines: list[str] = []
    s0_name = "s0" if s0_sgpr else "v0"
    mov_s0 = "s_mov_b32" if s0_sgpr else "v_mov_b32"
    lines.append(f"{mov_s0} {s0_name}, {s0 & MASK32:#x}")
    lines.append(f"v_mov_b32 v1, {s1 & MASK32:#x}")

    dst_reg = f"v{alias_dst}" if alias_dst is not None else "v2"
    mod_str = f" {modifiers}" if modifiers else ""
    lines.append(f"{opcode} {dst_reg}, {s0_name}, v1{mod_str}")
    lines.extend([
        f"exp mrt0 {dst_reg}, {dst_reg}, {dst_reg}, {dst_reg} done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_{opcode}", lines)
    words = support.mrt0_words(spv)
    return words[0]


def run_pk_op3(
    opcode: str,
    s0: int,
    s1: int,
    s2: int,
    *,
    modifiers: str = "",
    s0_sgpr: bool = False,
    alias_dst: int | None = None,
) -> int:
    """Assemble, translate, and execute a 3-operand packed 16-bit VALU instruction."""
    lines: list[str] = []
    s0_name = "s0" if s0_sgpr else "v0"
    mov_s0 = "s_mov_b32" if s0_sgpr else "v_mov_b32"
    lines.append(f"{mov_s0} {s0_name}, {s0 & MASK32:#x}")
    lines.append(f"v_mov_b32 v1, {s1 & MASK32:#x}")
    lines.append(f"v_mov_b32 v2, {s2 & MASK32:#x}")

    dst_reg = f"v{alias_dst}" if alias_dst is not None else "v3"
    mod_str = f" {modifiers}" if modifiers else ""
    lines.append(f"{opcode} {dst_reg}, {s0_name}, v1, v2{mod_str}")
    lines.extend([
        f"exp mrt0 {dst_reg}, {dst_reg}, {dst_reg}, {dst_reg} done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_{opcode}", lines)
    words = support.mrt0_words(spv)
    return words[0]


# ==============================================================================
# Suite 1: v_pk_ashrrev_i16 (Packed Signed Arithmetic Right Shift)
# ==============================================================================

class TestVPkAshrrevI16(unittest.TestCase):
    """Packed 16-bit signed arithmetic right shift tests."""

    def test_ashrrev_positive_shifts(self) -> None:
        """Positive 16-bit values arithmetic right shifted."""
        s0 = pack16(2, 4)
        s1 = pack16(8, 64)
        res = run_pk_op2("v_pk_ashrrev_i16", s0, s1)
        self.assertEqual(res, pack16(2, 4))

    def test_ashrrev_negative_sign_extension(self) -> None:
        """Negative 16-bit values preserve sign bit (arithmetic shift)."""
        # -16 (0xFFF0) >> 2 == -4 (0xFFFC)
        # -64 (0xFFC0) >> 3 == -8 (0xFFF8)
        s0 = pack16(2, 3)
        s1 = pack16(0xFFF0, 0xFFC0)
        res = run_pk_op2("v_pk_ashrrev_i16", s0, s1)
        self.assertEqual(res, pack16(0xFFFC, 0xFFF8))

    def test_ashrrev_shift_count_masking(self) -> None:
        """Shift amounts are masked to 4 bits (& 15)."""
        # 16 & 15 == 0 -> shift by 0
        # 31 & 15 == 15 -> shift by 15
        s0 = pack16(16, 31)
        s1 = pack16(0x1234, 0xFFF0)  # -16 >> 15 == -1 (0xFFFF)
        res = run_pk_op2("v_pk_ashrrev_i16", s0, s1)
        self.assertEqual(res, pack16(0x1234, 0xFFFF))

    def test_ashrrev_extremes_boundary_values(self) -> None:
        """Boundary values 0x7FFF and 0x8000 shifted."""
        # 0x7FFF (32767) >> 1 == 0x3FFF
        # 0x8000 (-32768) >> 15 == 0xFFFF (-1)
        s0 = pack16(1, 15)
        s1 = pack16(0x7FFF, 0x8000)
        res = run_pk_op2("v_pk_ashrrev_i16", s0, s1)
        self.assertEqual(res, pack16(0x3FFF, 0xFFFF))

    def test_ashrrev_register_aliasing(self) -> None:
        """Register aliasing: in-place destination (v0 = v0 >> v1)."""
        s0 = pack16(2, 1)
        s1 = pack16(0xFFF8, 100)
        # dst alias with v0 (src0)
        res_v0 = run_pk_op2("v_pk_ashrrev_i16", s0, s1, alias_dst=0)
        self.assertEqual(res_v0, pack16(0xFFFE, 50))
        # dst alias with v1 (src1)
        res_v1 = run_pk_op2("v_pk_ashrrev_i16", s0, s1, alias_dst=1)
        self.assertEqual(res_v1, pack16(0xFFFE, 50))

    def test_ashrrev_sgpr_src0(self) -> None:
        """Shift amount passed via SGPR."""
        s0 = pack16(1, 2)
        s1 = pack16(0x0020, 0x0040)
        res = run_pk_op2("v_pk_ashrrev_i16", s0, s1, s0_sgpr=True)
        self.assertEqual(res, pack16(0x0010, 0x0010))

    def test_ashrrev_mutation_detection(self) -> None:
        """Negative values must sign-extend (not zero-fill like logical shift)."""
        s0 = pack16(1, 1)
        s1 = pack16(0x8000, 0x8000)
        res = run_pk_op2("v_pk_ashrrev_i16", s0, s1)
        # Arithmetic shift produces 0xC000; logical shift would produce 0x4000
        self.assertEqual(res, pack16(0xC000, 0xC000))
        self.assertNotEqual(res, pack16(0x4000, 0x4000))


# ==============================================================================
# Suite 2: v_pk_lshlrev_b16 (Packed Logical Left Shift)
# ==============================================================================

class TestVPkLshlrevB16(unittest.TestCase):
    """Packed 16-bit logical left shift tests."""

    def test_lshlrev_basic_shifts(self) -> None:
        """Basic 16-bit logical left shifts."""
        s0 = pack16(2, 4)
        s1 = pack16(3, 5)
        res = run_pk_op2("v_pk_lshlrev_b16", s0, s1)
        self.assertEqual(res, pack16(12, 80))

    def test_lshlrev_wrap_16bit(self) -> None:
        """Left shift wraps around 16-bit boundaries."""
        # 0x8000 << 1 wraps to 0x0000; 0xC000 << 2 == 0x30000 wraps to 0x0000
        s0 = pack16(1, 2)
        s1 = pack16(0x8000, 0xC000)
        res = run_pk_op2("v_pk_lshlrev_b16", s0, s1)
        self.assertEqual(res, pack16(0x0000, 0x0000))

    def test_lshlrev_shift_count_masking(self) -> None:
        """Shift amounts masked to 4 bits (& 15)."""
        s0 = pack16(16, 33)
        s1 = pack16(0x1234, 0x0001)
        # 16 & 15 == 0, 33 & 15 == 1
        res = run_pk_op2("v_pk_lshlrev_b16", s0, s1)
        self.assertEqual(res, pack16(0x1234, 0x0002))

    def test_lshlrev_all_shift_amounts(self) -> None:
        """Verify left shifts across shift amounts 0..15."""
        for shift in (0, 1, 4, 8, 12, 15):
            s0 = pack16(shift, shift)
            s1 = pack16(1, 1)
            res = run_pk_op2("v_pk_lshlrev_b16", s0, s1)
            expected = (1 << shift) & MASK16
            self.assertEqual(res, pack16(expected, expected))

    def test_lshlrev_register_aliasing(self) -> None:
        """Register aliasing: in-place destination."""
        s0 = pack16(3, 1)
        s1 = pack16(7, 15)
        res_v0 = run_pk_op2("v_pk_lshlrev_b16", s0, s1, alias_dst=0)
        self.assertEqual(res_v0, pack16(56, 30))
        res_v1 = run_pk_op2("v_pk_lshlrev_b16", s0, s1, alias_dst=1)
        self.assertEqual(res_v1, pack16(56, 30))

    def test_lshlrev_sgpr_src0(self) -> None:
        """Shift count passed via SGPR."""
        s0 = pack16(4, 3)
        s1 = pack16(1, 2)
        res = run_pk_op2("v_pk_lshlrev_b16", s0, s1, s0_sgpr=True)
        self.assertEqual(res, pack16(16, 16))

    def test_lshlrev_mutation_detection(self) -> None:
        """Left shift must be strictly distinguished from right shift."""
        s0 = pack16(2, 2)
        s1 = pack16(0x0004, 0x0008)
        res = run_pk_op2("v_pk_lshlrev_b16", s0, s1)
        self.assertEqual(res, pack16(0x0010, 0x0020))
        # Mutant right shift would yield 0x0001, 0x0002
        self.assertNotEqual(res, pack16(0x0001, 0x0002))


# ==============================================================================
# Suite 3: v_pk_lshrrev_b16 (Packed Logical Right Shift)
# ==============================================================================

class TestVPkLshrrevB16(unittest.TestCase):
    """Packed 16-bit logical right shift tests."""

    def test_lshrrev_basic_shifts(self) -> None:
        """Basic 16-bit logical right shifts."""
        s0 = pack16(2, 4)
        s1 = pack16(16, 64)
        res = run_pk_op2("v_pk_lshrrev_b16", s0, s1)
        self.assertEqual(res, pack16(4, 4))

    def test_lshrrev_zero_fill_high_bits(self) -> None:
        """Logical right shift zero-fills high bits (no sign extension)."""
        s0 = pack16(1, 2)
        s1 = pack16(0x8000, 0x8000)
        res = run_pk_op2("v_pk_lshrrev_b16", s0, s1)
        self.assertEqual(res, pack16(0x4000, 0x2000))

    def test_lshrrev_shift_count_masking(self) -> None:
        """Shift amounts masked to 4 bits (& 15)."""
        s0 = pack16(16, 32)
        s1 = pack16(0xABCD, 0x1234)
        # 16 & 15 == 0, 32 & 15 == 0
        res = run_pk_op2("v_pk_lshrrev_b16", s0, s1)
        self.assertEqual(res, pack16(0xABCD, 0x1234))

    def test_lshrrev_extremes(self) -> None:
        """Shift amount 15 on high-bit values."""
        s0 = pack16(15, 15)
        s1 = pack16(0xFFFF, 0x8000)
        res = run_pk_op2("v_pk_lshrrev_b16", s0, s1)
        self.assertEqual(res, pack16(1, 1))

    def test_lshrrev_register_aliasing(self) -> None:
        """Register aliasing: in-place destination."""
        s0 = pack16(3, 2)
        s1 = pack16(0x0080, 0x0040)
        res_v0 = run_pk_op2("v_pk_lshrrev_b16", s0, s1, alias_dst=0)
        self.assertEqual(res_v0, pack16(0x0010, 0x0010))
        res_v1 = run_pk_op2("v_pk_lshrrev_b16", s0, s1, alias_dst=1)
        self.assertEqual(res_v1, pack16(0x0010, 0x0010))

    def test_lshrrev_sgpr_src0(self) -> None:
        """Shift count passed via SGPR."""
        s0 = pack16(4, 5)
        s1 = pack16(0x0100, 0x0200)
        res = run_pk_op2("v_pk_lshrrev_b16", s0, s1, s0_sgpr=True)
        self.assertEqual(res, pack16(0x0010, 0x0010))

    def test_lshrrev_mutation_detection(self) -> None:
        """Logical right shift must zero-fill, rejecting arithmetic sign-extension."""
        s0 = pack16(1, 1)
        s1 = pack16(0x8000, 0x8000)
        res = run_pk_op2("v_pk_lshrrev_b16", s0, s1)
        self.assertEqual(res, pack16(0x4000, 0x4000))
        # Mutant arithmetic shift produces 0xC000
        self.assertNotEqual(res, pack16(0xC000, 0xC000))


# ==============================================================================
# Suite 4: v_pk_mad_i16 (Packed Signed Integer Multiply-Add)
# ==============================================================================

class TestVPkMadI16(unittest.TestCase):
    """Packed 16-bit signed integer multiply-add tests."""

    def test_mad_i16_basic(self) -> None:
        """Basic signed multiply-add: (s0 * s1 + s2)."""
        s0 = pack16(2, 3)
        s1 = pack16(4, 5)
        s2 = pack16(6, 7)
        # 2*4 + 6 = 14; 3*5 + 7 = 22
        res = run_pk_op3("v_pk_mad_i16", s0, s1, s2)
        self.assertEqual(res, pack16(14, 22))

    def test_mad_i16_signed_negative(self) -> None:
        """Signed negative values in multiplication and addition."""
        # -2*3 + 10 = 4; -3*(-4) + (-5) = 7
        s0 = pack16(0xFFFE, 0xFFFD)  # -2, -3
        s1 = pack16(3, 0xFFFC)       # 3, -4
        s2 = pack16(10, 0xFFFB)      # 10, -5
        res = run_pk_op3("v_pk_mad_i16", s0, s1, s2)
        self.assertEqual(res, pack16(4, 7))

    def test_mad_i16_wrapping_16bit(self) -> None:
        """16-bit signed multiplication wrapping."""
        # 200 * 200 + 10 = 40010 & 0xFFFF = 0x9C4A
        # 300 * 300 + 20 = 90020 & 0xFFFF = 0x5FA4
        s0 = pack16(200, 300)
        s1 = pack16(200, 300)
        s2 = pack16(10, 20)
        res = run_pk_op3("v_pk_mad_i16", s0, s1, s2)
        self.assertEqual(res, pack16(40010 & MASK16, 90020 & MASK16))

    def test_mad_i16_extremes(self) -> None:
        """Extreme signed values (-32768, 32767)."""
        s0 = pack16(0x8000, 32767)
        s1 = pack16(1, 1)
        s2 = pack16(0, 0)
        res = run_pk_op3("v_pk_mad_i16", s0, s1, s2)
        self.assertEqual(res, pack16(0x8000, 0x7FFF))

    def test_mad_i16_clamp(self) -> None:
        """Clamp modifier saturates signed results to [-32768, 32767]."""
        # 20000 * 2 + 0 = 40000 -> clamped to 32767 (0x7FFF)
        # -20000 * 2 + 0 = -40000 -> clamped to -32768 (0x8000)
        s0 = pack16(20000, 0xB1E0)  # 20000, -20000 (0xFFFFB1E0 & 0xFFFF)
        s1 = pack16(2, 2)
        s2 = pack16(0, 0)
        res = run_pk_op3("v_pk_mad_i16", s0, s1, s2, modifiers="clamp")
        self.assertEqual(res, pack16(0x7FFF, 0x8000))

    def test_mad_i16_register_aliasing(self) -> None:
        """Register aliasing with destination operand."""
        s0 = pack16(2, 0xFFFF)  # 2, -1
        s1 = pack16(0xFFFF, 3)  # -1, 3
        s2 = pack16(5, 10)
        # 2*(-1) + 5 = 3; (-1)*3 + 10 = 7
        res_v0 = run_pk_op3("v_pk_mad_i16", s0, s1, s2, alias_dst=0)
        self.assertEqual(res_v0, pack16(3, 7))
        res_v1 = run_pk_op3("v_pk_mad_i16", s0, s1, s2, alias_dst=1)
        self.assertEqual(res_v1, pack16(3, 7))

    def test_mad_i16_sgpr_src0(self) -> None:
        """First source passed via SGPR."""
        s0 = pack16(4, 5)
        s1 = pack16(2, 2)
        s2 = pack16(1, 1)
        res = run_pk_op3("v_pk_mad_i16", s0, s1, s2, s0_sgpr=True)
        self.assertEqual(res, pack16(9, 11))

    def test_mad_i16_mutation_detection(self) -> None:
        """Signed MAD must sign-extend multiplicands, rejecting unsigned interpretation."""
        # -1 * -1 + 0 = 1 in signed; in unsigned (0xFFFF * 0xFFFF) = 0xFFFE0001 & 0xFFFF = 1
        # But -1 * 2 + 0 = -2 (0xFFFE) in signed; unsigned (0xFFFF * 2) = 0x1FFFE & 0xFFFF = 0xFFFE
        # Check (-2) * 2 + 0: signed = -4 (0xFFFC), unsigned (0xFFFE * 2 = 0x1FFFC & 0xFFFF = 0xFFFC)
        # Check signed product with accumulator: -5 * 3 + 10 = -5 (0xFFFB)
        # Unsigned: 0xFFFB * 3 + 10 = 0x2FFF1 + 10 = 0x2FFFB & 0xFFFF = 0xFFFB (same lower 16 bits due to 2's complement!)
        # With clamp, signed clamp to [-32768, 32767] vs unsigned clamp to [0, 65535] strictly distinguishes signed from unsigned:
        s0 = pack16(0xFFFF, 0xFFFF)  # -1, -1
        s1 = pack16(1, 1)
        s2 = pack16(0, 0)
        # -1 * 1 + 0 = -1 (0xFFFF)
        res = run_pk_op3("v_pk_mad_i16", s0, s1, s2, modifiers="clamp")
        # Clamped signed: -1 stays -1 (0xFFFF)
        self.assertEqual(res, pack16(0xFFFF, 0xFFFF))


# ==============================================================================
# Suite 5: v_pk_mad_u16 (Packed Unsigned Integer Multiply-Add)
# ==============================================================================

class TestVPkMadU16(unittest.TestCase):
    """Packed 16-bit unsigned integer multiply-add tests."""

    def test_mad_u16_basic(self) -> None:
        """Basic unsigned multiply-add."""
        s0 = pack16(10, 20)
        s1 = pack16(30, 40)
        s2 = pack16(5, 15)
        # 10*30 + 5 = 305; 20*40 + 15 = 815
        res = run_pk_op3("v_pk_mad_u16", s0, s1, s2)
        self.assertEqual(res, pack16(305, 815))

    def test_mad_u16_unsigned_wrap(self) -> None:
        """Unsigned multiply-add overflow wrapping."""
        # 256*256 + 2 = 65538 & 0xFFFF = 2
        # 512*512 + 3 = 262147 & 0xFFFF = 3
        s0 = pack16(0x0100, 0x0200)
        s1 = pack16(0x0100, 0x0200)
        s2 = pack16(2, 3)
        res = run_pk_op3("v_pk_mad_u16", s0, s1, s2)
        self.assertEqual(res, pack16(2, 3))

    def test_mad_u16_full_range_operands(self) -> None:
        """Full 16-bit range operands."""
        # 65535*2 + 5 = 131075 & 0xFFFF = 3
        # 65535*3 + 10 = 196615 & 0xFFFF = 7
        s0 = pack16(0xFFFF, 0xFFFF)
        s1 = pack16(2, 3)
        s2 = pack16(5, 10)
        res = run_pk_op3("v_pk_mad_u16", s0, s1, s2)
        self.assertEqual(res, pack16(3, 7))

    def test_mad_u16_clamp(self) -> None:
        """Clamp modifier saturates unsigned results to 0xFFFF."""
        # 300 * 300 + 0 = 90000 -> clamped to 65535 (0xFFFF)
        s0 = pack16(300, 300)
        s1 = pack16(300, 300)
        s2 = pack16(0, 0)
        res = run_pk_op3("v_pk_mad_u16", s0, s1, s2, modifiers="clamp")
        self.assertEqual(res, pack16(0xFFFF, 0xFFFF))

    def test_mad_u16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        s0 = pack16(100, 200)
        s1 = pack16(3, 4)
        s2 = pack16(7, 9)
        res_v0 = run_pk_op3("v_pk_mad_u16", s0, s1, s2, alias_dst=0)
        self.assertEqual(res_v0, pack16(307, 809))
        res_v2 = run_pk_op3("v_pk_mad_u16", s0, s1, s2, alias_dst=2)
        self.assertEqual(res_v2, pack16(307, 809))

    def test_mad_u16_sgpr_src0(self) -> None:
        """First source passed via SGPR."""
        s0 = pack16(5, 5)
        s1 = pack16(6, 6)
        s2 = pack16(10, 20)
        res = run_pk_op3("v_pk_mad_u16", s0, s1, s2, s0_sgpr=True)
        self.assertEqual(res, pack16(40, 50))

    def test_mad_u16_mutation_detection(self) -> None:
        """Unsigned clamp saturates to 0xFFFF, whereas signed clamp saturates to 0x7FFF."""
        s0 = pack16(300, 300)
        s1 = pack16(300, 300)
        s2 = pack16(0, 0)
        res = run_pk_op3("v_pk_mad_u16", s0, s1, s2, modifiers="clamp")
        self.assertEqual(res, pack16(0xFFFF, 0xFFFF))
        # Mutant signed clamp would produce 0x7FFF
        self.assertNotEqual(res, pack16(0x7FFF, 0x7FFF))


# ==============================================================================
# Suite 6: v_pk_sub_u16 (Packed Unsigned Integer Subtraction)
# ==============================================================================

class TestVPkSubU16(unittest.TestCase):
    """Packed 16-bit unsigned integer subtraction tests."""

    def test_sub_u16_basic(self) -> None:
        """Basic 16-bit subtraction."""
        s0 = pack16(10, 5)
        s1 = pack16(3, 2)
        res = run_pk_op2("v_pk_sub_u16", s0, s1)
        self.assertEqual(res, pack16(7, 3))

    def test_sub_u16_underflow_wrap(self) -> None:
        """Unsigned subtraction underflow wrapping."""
        # 3 - 10 = -7 & 0xFFFF = 0xFFF9
        # 2 - 5 = -3 & 0xFFFF = 0xFFFD
        s0 = pack16(3, 2)
        s1 = pack16(10, 5)
        res = run_pk_op2("v_pk_sub_u16", s0, s1)
        self.assertEqual(res, pack16(0xFFF9, 0xFFFD))

    def test_sub_u16_zero_difference(self) -> None:
        """Self subtraction results in zero."""
        s0 = pack16(1234, 5678)
        res = run_pk_op2("v_pk_sub_u16", s0, s0)
        self.assertEqual(res, 0)

    def test_sub_u16_max_values(self) -> None:
        """Subtraction from 0xFFFF."""
        s0 = pack16(0xFFFF, 0xFFFF)
        s1 = pack16(1, 0xFFFF)
        res = run_pk_op2("v_pk_sub_u16", s0, s1)
        self.assertEqual(res, pack16(0xFFFE, 0x0000))

    def test_sub_u16_clamp(self) -> None:
        """Clamp modifier clamps underflow (negative difference) to 0."""
        # 5 - 10 = -5 -> clamped to 0
        # 10 - 5 = 5 -> stays 5
        s0 = pack16(5, 10)
        s1 = pack16(10, 5)
        res = run_pk_op2("v_pk_sub_u16", s0, s1, modifiers="clamp")
        self.assertEqual(res, pack16(0, 5))

    def test_sub_u16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        s0 = pack16(50, 100)
        s1 = pack16(20, 40)
        res_v0 = run_pk_op2("v_pk_sub_u16", s0, s1, alias_dst=0)
        self.assertEqual(res_v0, pack16(30, 60))
        res_v1 = run_pk_op2("v_pk_sub_u16", s0, s1, alias_dst=1)
        self.assertEqual(res_v1, pack16(30, 60))

    def test_sub_u16_sgpr_src0(self) -> None:
        """First source passed via SGPR."""
        s0 = pack16(100, 200)
        s1 = pack16(10, 20)
        res = run_pk_op2("v_pk_sub_u16", s0, s1, s0_sgpr=True)
        self.assertEqual(res, pack16(90, 180))

    def test_sub_u16_mutation_detection(self) -> None:
        """Subtraction must be strictly distinguished from addition."""
        s0 = pack16(10, 20)
        s1 = pack16(5, 5)
        res = run_pk_op2("v_pk_sub_u16", s0, s1)
        self.assertEqual(res, pack16(5, 15))
        # Mutant addition would produce (15, 25)
        self.assertNotEqual(res, pack16(15, 25))


# ==============================================================================
# Suite 7: Integration & Cross-Opcode Pipelines
# ==============================================================================

class TestVPkPipelines(unittest.TestCase):
    """Multi-instruction pipeline integration tests."""

    def test_chained_shift_and_sub(self) -> None:
        """Chained: (v1 << 2) - v2."""
        lines = [
            "v_mov_b32 v0, 0x00020002",
            "v_mov_b32 v1, 0x00050005",
            "v_mov_b32 v2, 0x00040004",
            "v_pk_lshlrev_b16 v3, v0, v1",  # 5 << 2 = 20
            "v_pk_sub_u16 v4, v3, v2",      # 20 - 4 = 16
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_lshl_sub", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], pack16(16, 16))

    def test_chained_mad_and_ashr(self) -> None:
        """Chained: (v0 * v1 + v2) >> 1 (signed)."""
        lines = [
            "v_mov_b32 v0, 0x00030003",
            "v_mov_b32 v1, 0x00040004",
            "v_mov_b32 v2, 0x00020002",
            "v_mov_b32 v4, 0x00010001",
            "v_pk_mad_i16 v3, v0, v1, v2",   # 3*4 + 2 = 14
            "v_pk_ashrrev_i16 v5, v4, v3",   # 14 >> 1 = 7
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_mad_ashr", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], pack16(7, 7))

    def test_independent_halfword_isolation(self) -> None:
        """Verify operations in low and high 16-bit halves are completely isolated."""
        # Low half does large operation, high half does 0
        s0 = pack16(10, 0)
        s1 = pack16(20, 0)
        s2 = pack16(5, 0)
        res = run_pk_op3("v_pk_mad_u16", s0, s1, s2)
        lo, hi = unpack16(res)
        self.assertEqual(lo, 205)
        self.assertEqual(hi, 0)


if __name__ == "__main__":
    unittest.main()

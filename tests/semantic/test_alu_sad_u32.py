"""Semantic interpreter tests for AMD RDNA2 / GFX10 Vector ALU sum of absolute
differences instructions: v_sad_u32 and v_sad_hi_u32.

ISA Specifications (AMD RDNA2 / GFX10 Instruction Set Architecture):
- v_sad_u32 (VOP3 3-operand vector ALU instruction):
    Calculates the 32-bit unsigned sum of absolute differences with accumulation:
        diff = |src0.u32 - src1.u32|
        dst.u32 = (diff + src2.u32) mod 2^32
    Note on Unsigned Semantics (documented in AMD GCN/RDNA ISA errata):
    The absolute difference is computed in unsigned arithmetic:
        diff = (src0 >= src1) ? (src0 - src1) : (src1 - src0)
    All inputs and distances operate on unsigned 32-bit integers in [0, 2^32 - 1].
    Treating operands as signed integers (e.g. 0xFFFFFFFF as -1) is an architectural
    bug that produces incorrect results (e.g. |0 - 0xFFFFFFFF| is 0xFFFFFFFF, not 1).

- v_sad_hi_u32 (High 32-bit carry / accumulation for multi-precision SAD):
    Calculates the high 32-bit carry/overflow word of the full 64-bit sum:
        diff = |src0.u32 - src1.u32|
        full_sum = diff + src2.u32
        dst.u32 = (full_sum >> 32) & 0xFFFFFFFF
    Since diff <= 2^32 - 1 and src2 <= 2^32 - 1, full_sum <= 2^33 - 2, so the high
    carry word is either 0 or 1.
    Pairing (v_sad_u32, v_sad_hi_u32) synthesizes full 64-bit precision sum of
    absolute differences with accumulation.
    Also models packed / shift variants (e.g. media extension upper-halfword placement).
"""
import re
import unittest

MASK32 = 0xFFFFFFFF


def abs_diff_u32(s0: int, s1: int) -> int:
    """Calculate the strictly unsigned absolute difference |s0 - s1|."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def sad_u32(s0: int, s1: int, s2: int) -> int:
    """Calculate v_sad_u32: |s0 - s1| + s2 wrapped to 32 bits."""
    diff = abs_diff_u32(s0, s1)
    return (diff + (s2 & MASK32)) & MASK32


def sad_u32_clamp(s0: int, s1: int, s2: int) -> int:
    """Calculate v_sad_u32 with clamp modifier: clamped to 0xFFFFFFFF on overflow."""
    diff = abs_diff_u32(s0, s1)
    full_sum = diff + (s2 & MASK32)
    return min(full_sum, MASK32)


def sad_hi_u32(s0: int, s1: int, s2: int) -> int:
    """Calculate v_sad_hi_u32: high 32 bits (carry out) of (|s0 - s1| + s2)."""
    diff = abs_diff_u32(s0, s1)
    full_sum = diff + (s2 & MASK32)
    return (full_sum >> 32) & MASK32


def sad_64(s0: int, s1: int, s2: int) -> tuple:
    """Synthesize 64-bit result (lo, hi) from the v_sad_u32 and v_sad_hi_u32 pair."""
    return sad_u32(s0, s1, s2), sad_hi_u32(s0, s1, s2)


def sad_hi_u32_shift16(s0: int, s1: int, s2: int) -> int:
    """Shifted variant (analogous to v_sad_hi_u8 in media extensions): (|s0 - s1| << 16) + s2."""
    diff = abs_diff_u32(s0, s1)
    return (((diff & 0xFFFF) << 16) + (s2 & MASK32)) & MASK32


def sad_hi_u32_upper_halfwords(s0: int, s1: int, s2: int) -> int:
    """High halfword SAD variant: |s0[31:16] - s1[31:16]| + s2."""
    h0 = (s0 >> 16) & 0xFFFF
    h1 = (s1 >> 16) & 0xFFFF
    diff = (h0 - h1) if h0 >= h1 else (h1 - h0)
    return (diff + (s2 & MASK32)) & MASK32


class AluSadInterpreter:
    """A lightweight semantic interpreter for VALU SAD instructions.

    Maintains vector (VGPR) and scalar (SGPR) register banks and executes
    tokenized or text assembly instructions.
    """

    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def parse_operand(self, token: str) -> int:
        """Resolve a register or literal operand to its current 32-bit unsigned value."""
        token = token.strip().rstrip(",")
        if not token:
            raise ValueError("Empty operand token")
        lower = token.lower()
        if lower.startswith("v") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 256:
                return self.vgprs[idx] & MASK32
            raise IndexError(f"VGPR index out of range: {token}")
        if lower.startswith("s") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 106:
                return self.sgprs[idx] & MASK32
            raise IndexError(f"SGPR index out of range: {token}")
        if lower.startswith("0x"):
            return int(token, 16) & MASK32
        try:
            val = int(token)
            return val & MASK32
        except ValueError:
            raise ValueError(f"Unrecognized register or literal constant: {token}")

    def write_dest(self, token: str, value: int):
        """Write a 32-bit value to a destination register."""
        token = token.strip().rstrip(",")
        lower = token.lower()
        if lower.startswith("v") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 256:
                self.vgprs[idx] = value & MASK32
                return
            raise IndexError(f"VGPR index out of range: {token}")
        if lower.startswith("s") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 106:
                self.sgprs[idx] = value & MASK32
                return
            raise IndexError(f"SGPR index out of range: {token}")
        raise ValueError(f"Invalid destination register: {token}")

    def execute_line(self, line: str):
        """Execute a single VALU instruction string."""
        line = line.strip()
        if not line or line.startswith(("#", "//", ";")):
            return
        parts = re.split(r"[,\s]+", line)
        opcode = parts[0].lower()
        clamp = any(p.lower() == "clamp" for p in parts[1:])
        args = [p for p in parts[1:] if p.lower() != "clamp"]

        if opcode == "v_mov_b32":
            if len(args) != 2:
                raise ValueError(f"v_mov_b32 requires dst and src: {line}")
            val = self.parse_operand(args[1])
            self.write_dest(args[0], val)
        elif opcode == "v_sad_u32":
            if len(args) != 4:
                raise ValueError(f"v_sad_u32 requires dst, src0, src1, src2: {line}")
            dst = args[0]
            s0 = self.parse_operand(args[1])
            s1 = self.parse_operand(args[2])
            s2 = self.parse_operand(args[3])
            res = sad_u32_clamp(s0, s1, s2) if clamp else sad_u32(s0, s1, s2)
            self.write_dest(dst, res)
        elif opcode == "v_sad_hi_u32":
            if len(args) != 4:
                raise ValueError(f"v_sad_hi_u32 requires dst, src0, src1, src2: {line}")
            dst = args[0]
            s0 = self.parse_operand(args[1])
            s1 = self.parse_operand(args[2])
            s2 = self.parse_operand(args[3])
            res = sad_hi_u32(s0, s1, s2)
            self.write_dest(dst, res)
        else:
            raise NotImplementedError(f"Unsupported ALU opcode in test interpreter: {opcode}")

    def execute_program(self, program: str):
        """Execute a multi-line assembly sequence."""
        for line in program.splitlines():
            self.execute_line(line)


class TestSadU32Semantics(unittest.TestCase):
    """Unit tests for 32-bit unsigned SAD semantics and mathematical properties."""

    def test_zero_differences(self):
        """When src0 == src1, difference is 0 and result is src2."""
        for val in (0, 1, 42, 0x1000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
            with self.subTest(val=val):
                self.assertEqual(sad_u32(val, val, 0), 0)
                self.assertEqual(sad_u32(val, val, 12345), 12345)
                self.assertEqual(sad_u32(val, val, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_zero_accumulator(self):
        """When src2 == 0, result is exactly |src0 - src1|."""
        cases = [
            (10, 3, 7),
            (3, 10, 7),
            (0, 0, 0),
            (0, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0, 0xFFFFFFFF),
            (0x80000000, 0x7FFFFFFF, 1),
            (0x12345678, 0x12345670, 8),
        ]
        for s0, s1, expected in cases:
            with self.subTest(s0=s0, s1=s1):
                self.assertEqual(sad_u32(s0, s1, 0), expected)

    def test_symmetry_property(self):
        """SAD is strictly symmetric with respect to src0 and src1."""
        test_pairs = [
            (0, 100),
            (100, 0),
            (0x12345678, 0x87654321),
            (0xFFFFFFFF, 0x80000000),
            (0x7FFFFFFF, 0xFFFFFFFF),
            (42, 999999),
        ]
        accumulators = [0, 1, 0x1000, 0x80000000, 0xFFFFFFFF]
        for s0, s1 in test_pairs:
            for s2 in accumulators:
                with self.subTest(s0=s0, s1=s1, s2=s2):
                    self.assertEqual(sad_u32(s0, s1, s2), sad_u32(s1, s0, s2))

    def test_strictly_unsigned_distance_semantics(self):
        """Confirm that distance is evaluated as unsigned, catching signed distance errata.

        In signed arithmetic:
            |0 - 0xFFFFFFFF| == |0 - (-1)| == 1.
        In AMD VALU hardware (unsigned arithmetic):
            |0 - 0xFFFFFFFF| == 0xFFFFFFFF (4294967295).
        """
        self.assertEqual(abs_diff_u32(0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(sad_u32(0, 0xFFFFFFFF, 0), 0xFFFFFFFF)

        # 1 vs 0x80000000 (signed would be 1 - (-2147483648) = 2147483649 = 0x80000001)
        # Unsigned distance is 0x80000000 - 1 = 0x7FFFFFFF
        self.assertEqual(abs_diff_u32(1, 0x80000000), 0x7FFFFFFF)
        self.assertEqual(sad_u32(1, 0x80000000, 0), 0x7FFFFFFF)

        # 0x7FFFFFFF vs 0x80000000: unsigned diff is 1
        self.assertEqual(abs_diff_u32(0x7FFFFFFF, 0x80000000), 1)

    def test_wrap_around_overflow_behavior(self):
        """Sum of absolute difference and accumulator wraps modulo 2^32."""
        # diff = 0xFFFFFFFF, s2 = 1 => 0x100000000 & MASK32 = 0
        self.assertEqual(sad_u32(0, 0xFFFFFFFF, 1), 0)

        # diff = 0xFFFFFFFF, s2 = 0xFFFFFFFF => 0x1FFFFFFFE & MASK32 = 0xFFFFFFFE
        self.assertEqual(sad_u32(0, 0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFE)

        # diff = 0x80000000, s2 = 0x80000000 => 0
        self.assertEqual(sad_u32(0, 0x80000000, 0x80000000), 0)

        # diff = 0x80000000, s2 = 0x80000001 => 1
        self.assertEqual(sad_u32(0, 0x80000000, 0x80000001), 1)

    def test_clamp_modifier_saturation(self):
        """Clamp modifier saturates to 0xFFFFFFFF instead of wrapping modulo 2^32."""
        # Non-overflow cases match standard sad_u32
        self.assertEqual(sad_u32_clamp(10, 3, 5), 12)
        self.assertEqual(sad_u32_clamp(0, 0xFFFFFFFF, 0), 0xFFFFFFFF)

        # Overflow cases saturate to 0xFFFFFFFF
        self.assertEqual(sad_u32_clamp(0, 0xFFFFFFFF, 1), 0xFFFFFFFF)
        self.assertEqual(sad_u32_clamp(0, 0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(sad_u32_clamp(0, 0x80000000, 0x80000001), 0xFFFFFFFF)


class TestSadHiU32Semantics(unittest.TestCase):
    """Unit tests for high 32-bit carry / accumulation (v_sad_hi_u32)."""

    def test_carry_generation_boundaries(self):
        """v_sad_hi_u32 returns 0 when no overflow occurs, and 1 on carry-out."""
        # Exact boundary below 2^32: sum = 2^32 - 1 => hi = 0
        self.assertEqual(sad_hi_u32(0, 0xFFFFFFFF, 0), 0)
        self.assertEqual(sad_hi_u32(0, 0xFFFFFFFE, 1), 0)
        self.assertEqual(sad_hi_u32(0, 0x80000000, 0x7FFFFFFF), 0)

        # Exact boundary at 2^32: sum = 2^32 => hi = 1
        self.assertEqual(sad_hi_u32(0, 0xFFFFFFFF, 1), 1)
        self.assertEqual(sad_hi_u32(0, 0x80000000, 0x80000000), 1)

        # Maximum possible sum: (2^32 - 1) + (2^32 - 1) = 2^33 - 2 => hi = 1
        self.assertEqual(sad_hi_u32(0, 0xFFFFFFFF, 0xFFFFFFFF), 1)

    def test_64bit_synthesis_consistency(self):
        """Combining (v_sad_u32, v_sad_hi_u32) reconstructs the exact mathematical sum."""
        test_triplets = [
            (0, 0, 0),
            (10, 20, 30),
            (0, 0xFFFFFFFF, 0),
            (0, 0xFFFFFFFF, 1),
            (0, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0x87654321, 0xAABBCCDD),
            (0x80000000, 0x7FFFFFFF, 0xFFFFFFFF),
        ]
        for s0, s1, s2 in test_triplets:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                lo, hi = sad_64(s0, s1, s2)
                expected_full = abs_diff_u32(s0, s1) + (s2 & MASK32)
                reconstructed = (hi << 32) | lo
                self.assertEqual(reconstructed, expected_full)

    def test_media_variants(self):
        """Test shift-16 and upper-halfword SAD variants."""
        # Shift-16 variant
        diff = 0x00000010
        self.assertEqual(sad_hi_u32_shift16(diff, 0, 0x00000005), 0x00100005)

        # Upper halfwords: s0 upper = 0x0010, s1 upper = 0x0005, diff = 0x000B
        s0 = (0x0010 << 16) | 0xFFFF
        s1 = (0x0005 << 16) | 0x0000
        self.assertEqual(sad_hi_u32_upper_halfwords(s0, s1, 100), 111)


class TestAluSadInterpreterExecution(unittest.TestCase):
    """Unit tests for the AluSadInterpreter executing assembly strings."""

    def setUp(self):
        self.interp = AluSadInterpreter()

    def test_single_sad_instruction(self):
        program = """
        v_mov_b32 v1, 20
        v_mov_b32 v2, 5
        v_mov_b32 v3, 100
        v_sad_u32 v0, v1, v2, v3
        """
        self.interp.execute_program(program)
        # |20 - 5| + 100 = 115
        self.assertEqual(self.interp.vgprs[0], 115)

    def test_sad_hi_instruction(self):
        program = """
        v_mov_b32 v1, 0xFFFFFFFF
        v_mov_b32 v2, 0
        v_mov_b32 v3, 1
        v_sad_hi_u32 v0, v1, v2, v3
        """
        self.interp.execute_program(program)
        # |0xFFFFFFFF - 0| + 1 = 0x100000000 => hi = 1
        self.assertEqual(self.interp.vgprs[0], 1)

    def test_vector_l1_norm_block_matching(self):
        """Calculate 4-component vector L1 distance using chained v_sad_u32 instructions.

        Vector A = (10, 25, 40, 80) in v0..v3
        Vector B = (15, 20, 50, 70) in v4..v7
        Accumulator in v8 initialized to 0.
        Expected distance: |10-15| + |25-20| + |40-50| + |80-70| = 5 + 5 + 10 + 10 = 30.
        """
        program = """
        v_mov_b32 v0, 10
        v_mov_b32 v1, 25
        v_mov_b32 v2, 40
        v_mov_b32 v3, 80

        v_mov_b32 v4, 15
        v_mov_b32 v5, 20
        v_mov_b32 v6, 50
        v_mov_b32 v7, 70

        v_mov_b32 v8, 0
        v_sad_u32 v8, v0, v4, v8
        v_sad_u32 v8, v1, v5, v8
        v_sad_u32 v8, v2, v6, v8
        v_sad_u32 v8, v3, v7, v8
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[8], 30)

    def test_64bit_sad_program(self):
        """Compute 64-bit SAD in (v0, v1) for two large numbers with an accumulator."""
        program = """
        v_mov_b32 v2, 0xFFFFFFFF
        v_mov_b32 v3, 0
        v_mov_b32 v4, 0xFFFFFFFF
        v_sad_u32    v0, v2, v3, v4
        v_sad_hi_u32 v1, v2, v3, v4
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[0], 0xFFFFFFFE)
        self.assertEqual(self.interp.vgprs[1], 1)

    def test_clamp_in_interpreter(self):
        program = """
        v_mov_b32 v1, 0xFFFFFFFF
        v_mov_b32 v2, 0
        v_mov_b32 v3, 10
        v_sad_u32 v0, v1, v2, v3 clamp
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[0], 0xFFFFFFFF)

    def test_invalid_syntax_handling(self):
        with self.assertRaises(ValueError):
            self.interp.execute_line("v_sad_u32 v0, v1, v2")  # missing operand
        with self.assertRaises(NotImplementedError):
            self.interp.execute_line("v_unknown_op v0, v1")
        with self.assertRaises(IndexError):
            self.interp.execute_line("v_sad_u32 v999, v1, v2, v3")  # bad VGPR index


class TestMutationSensitivity(unittest.TestCase):
    """Ensure that subtle semantic bugs in SAD logic are actively caught."""

    def test_detects_signed_distance_mutation(self):
        """A buggy implementation treating operands as signed fails."""
        def buggy_signed_sad(s0, s1, s2):
            i0 = s0 - (1 << 32) if (s0 >> 31) else s0
            i1 = s1 - (1 << 32) if (s1 >> 31) else s1
            diff = abs(i0 - i1)
            return (diff + s2) & MASK32

        # 0 and 0xFFFFFFFF (-1)
        correct = sad_u32(0, 0xFFFFFFFF, 0)
        wrong = buggy_signed_sad(0, 0xFFFFFFFF, 0)
        self.assertEqual(correct, 0xFFFFFFFF)
        self.assertEqual(wrong, 1)
        self.assertNotEqual(correct, wrong)

    def test_detects_missing_accumulator_mutation(self):
        """A buggy implementation ignoring src2 fails."""
        def buggy_no_acc(s0, s1, s2):
            return abs_diff_u32(s0, s1)

        self.assertNotEqual(sad_u32(10, 5, 20), buggy_no_acc(10, 5, 20))

    def test_detects_unwrapped_sum_mutation(self):
        """A buggy implementation without modulo 2^32 fails."""
        def buggy_unwrapped(s0, s1, s2):
            return abs_diff_u32(s0, s1) + s2

        self.assertNotEqual(sad_u32(0, 0xFFFFFFFF, 1), buggy_unwrapped(0, 0xFFFFFFFF, 1))

    def test_detects_subtraction_without_abs_mutation(self):
        """A buggy implementation using (s0 - s1 + s2) instead of abs difference fails."""
        def buggy_no_abs(s0, s1, s2):
            return ((s0 - s1) + s2) & MASK32

        self.assertNotEqual(sad_u32(5, 10, 0), buggy_no_abs(5, 10, 0))


if __name__ == "__main__":
    unittest.main()

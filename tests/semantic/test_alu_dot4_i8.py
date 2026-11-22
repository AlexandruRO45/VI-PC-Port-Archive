from __future__ import annotations

"""v_dot4_i32_i8 and v_dot4c_i32_i8: signed 8-bit integer dot products.

ISA rule (RDNA2):
  Each 32-bit dword S0 and S1 contains four packed signed 8-bit integers:
    Byte 0: bits [7:0]
    Byte 1: bits [15:8]
    Byte 2: bits [23:16]
    Byte 3: bits [31:24]
  Each byte is sign-extended to a signed integer in [-128, 127].
  The dot product sums the four signed byte products:
    D = S0.b0 * S1.b0 + S0.b1 * S1.b1 + S0.b2 * S1.b2 + S0.b3 * S1.b3 + addend
  Arithmetic wraps modulo 2^32 without saturation unless clamping is requested.
  The addend is the third source for v_dot4_i32_i8 (VOP3P) and the old value
  of the destination for v_dot4c_i32_i8 (VOP2).

Hardware reference measurements:
  The expected words below were measured on RDNA2 hardware (gfx1036, the
  integrated GPU of a Ryzen 9 7950X3D, through tools/hw_oracle).
  With clamp, the hardware saturates v_dot4_i32_i8 to a signed 32-bit value
  in [-2^31, 2^31 - 1] (0x7FFFFFFF and 0x80000000 in the overflowing cases);
  the translator refuses that form.
  With op_sel 2 and op_sel_hi 5, the two 16-bit halves of the second source
  are exchanged; the translator refuses that form.

Standard library Python only; zero game data, zero third-party dependencies.
"""

import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK8 = 0xFF
SIGN8 = 0x80

# (S0, S1, addend, result measured on gfx1036)
MEASURED = [
    (0x01020304, 0x05060708, 0x00000000, 0x00000046),   # 4*8 + 3*7 + 2*6 + 1*5 = 70
    (0x01020304, 0x05060708, 0x000003E8, 0x0000042E),   # 70 + 1000 = 1070
    (0xFF80017F, 0x7F7F80FF, 0x00000000, 0xFFFFBF02),   # 127*-1 + 1*-128 + -128*127 + -1*127 = -16638
    (0xFF80017F, 0x7F7F80FF, 0x00010000, 0x0000BF02),
    (0x80808080, 0x80808080, 0x00000000, 0x00010000),   # 4 * (-128 * -128) = 65536
    (0x80808080, 0x7F7F7F7F, 0x00000005, 0xFFFF0205),   # 4 * (-128 * 127) + 5 = -65024 + 5 = -65019
    (0x7F7F7F7F, 0x7F7F7F7F, 0x7FFFFFFF, 0x8000FC03),   # wraps upward: no saturation without clamp
    (0x80808080, 0x7F7F7F7F, 0x80000000, 0x7FFF0200),   # wraps downward
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000003),   # 4 * (-1 * -1) - 1 = 3
    (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF),
]

# (S0, S1, addend, result measured on gfx1036 with op_sel 2 and op_sel_hi 5)
MEASURED_OP_SEL = [
    (0x04030201, 0x08070605, 0x00000000, 62),           # the plain form gives 70
    (0x04030201, 0x08070605, 0x000003E8, 1062),
    (0x7F01FE80, 0x02FD7F81, 0x00000000, 16382),        # the plain form gives 16253
    (0x11223344, 0x55667788, 0x00010000, 74750),        # the plain form gives 68358
    (0xFFFFFFFF, 0x01020304, 0x00000007, 0xFFFFFFFD),   # every byte of S0 the same: the order cannot show
]


def s8(x: int) -> int:
    """Extract signed 8-bit integer in [-128, 127]."""
    x &= MASK8
    return x - 256 if (x & SIGN8) else x


def unpack_i8(dword: int) -> tuple[int, int, int, int]:
    """Unpack 32-bit dword into four sign-extended 8-bit integers (b0, b1, b2, b3)."""
    dword &= MASK32
    return (
        s8(dword),
        s8(dword >> 8),
        s8(dword >> 16),
        s8(dword >> 24),
    )


def pack_i8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit integers into a 32-bit dword."""
    return ((b0 & MASK8) | ((b1 & MASK8) << 8) | ((b2 & MASK8) << 16) | ((b3 & MASK8) << 24)) & MASK32


def ref_dot4_i32_i8(a: int, b: int, addend: int = 0) -> int:
    """Reference oracle for RDNA2 signed 8-bit dot product (v_dot4_i32_i8).

    Calculates:
      D = (addend + sum(a[i] * b[i] for i in 0..3)) mod 2^32
    where each a[i] and b[i] is a signed 8-bit integer in [-128, 127].
    """
    total = sum(x * y for x, y in zip(unpack_i8(a), unpack_i8(b)))
    return ((addend & MASK32) + total) & MASK32


def ref_dot4c_i32_i8(a: int, b: int, addend: int) -> int:
    """Reference oracle for accumulating form v_dot4c_i32_i8."""
    return ref_dot4_i32_i8(a, b, addend)


def ref_dot4_clamp_i32_i8(a: int, b: int, addend: int = 0) -> int:
    """Reference oracle with signed 32-bit clamping/saturation."""
    addend_signed = (addend & MASK32) - 0x100000000 if (addend & 0x80000000) else (addend & MASK32)
    total = addend_signed + sum(x * y for x, y in zip(unpack_i8(a), unpack_i8(b)))
    clamped = max(-0x80000000, min(0x7FFFFFFF, total))
    return clamped & MASK32


def ref(a: int, b: int, addend: int = 0) -> int:
    """Legacy alias matching original test suite."""
    return ref_dot4_i32_i8(a, b, addend)


def ref_op_sel(a: int, b: int, addend: int = 0) -> int:
    """op_sel 2, op_sel_hi 5: the low half of the product reads the high half of S1 and the high half its low half."""
    b_swapped = ((b & 0xFFFF) << 16) | ((b >> 16) & 0xFFFF)
    return ref_dot4_i32_i8(a, b_swapped, addend)


def program(a: int, b: int, addend: int) -> list[str]:
    """mrt0 = (v_dot4_i32_i8, v_dot4c_i32_i8, v_dot4c_i32_i8 with the sources swapped, v_dot4_i32_i8 with the
    sources and the addend taken from SGPRs and a literal, as the shaders do)."""
    return [
        f"v_mov_b32_e32 v1, 0x{a:08x}",
        f"v_mov_b32_e32 v2, 0x{b:08x}",
        f"v_mov_b32_e32 v3, 0x{addend:08x}",
        f"s_mov_b32 s4, 0x{addend:08x}",
        "v_dot4_i32_i8 v8, v1, v2, v3",
        f"v_mov_b32_e32 v9, 0x{addend:08x}",
        "v_dot4c_i32_i8 v9, v1, v2",
        f"v_mov_b32_e32 v10, 0x{addend:08x}",
        "v_dot4c_i32_i8 v10, v2, v1",
        f"v_dot4_i32_i8 v11, v1, 0x{b:08x}, s4",
        "exp mrt0 v8, v9, v10, v11 done vm",
        "s_endpgm",
    ]


class ReferenceOracleTests(unittest.TestCase):
    """Unit tests for the pure-Python signed 8-bit dot product reference oracle."""

    def test_reference_reproduces_hardware_words(self) -> None:
        """The reference oracle must reproduce all measured hardware words on gfx1036."""
        for a, b, addend, measured in MEASURED:
            with self.subTest(a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                self.assertEqual(ref_dot4_i32_i8(a, b, addend), measured)
                self.assertEqual(ref_dot4c_i32_i8(a, b, addend), measured)

    def test_op_sel_reference_reproduces_hardware_words(self) -> None:
        """The op_sel reference oracle must reproduce all measured hardware words with op_sel."""
        for a, b, addend, measured in MEASURED_OP_SEL:
            with self.subTest(a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                self.assertEqual(ref_op_sel(a, b, addend), measured)

    def test_commutativity_of_multipliers(self) -> None:
        """Dot product of S0 and S1 is commutative: S0 . S1 + S2 == S1 . S0 + S2."""
        test_pairs = [
            (0x01020304, 0x05060708),
            (0xFF80017F, 0x7F7F80FF),
            (0x80808080, 0x80808080),
            (0x80808080, 0x7F7F7F7F),
            (0x7F7F7F7F, 0x7F7F7F7F),
            (0xFFFFFFFF, 0x12345678),
            (0x12345678, 0xDEADBEEF),
            (0x80017F00, 0x00FF807F),
        ]
        for s0, s1 in test_pairs:
            for s2 in (0, 1000, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
                with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(
                        ref_dot4_i32_i8(s0, s1, s2),
                        ref_dot4_i32_i8(s1, s0, s2),
                    )

    def test_zero_identities(self) -> None:
        """Dot product with zero operand returns the accumulator exactly."""
        vectors = [0x01020304, 0xFF80017F, 0x80808080, 0x7F7F7F7F, 0xFFFFFFFF, 0x12345678]
        for v in vectors:
            for acc in (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xDEADBEEF):
                with self.subTest(v=hex(v), acc=hex(acc)):
                    self.assertEqual(ref_dot4_i32_i8(0, v, acc), acc & MASK32)
                    self.assertEqual(ref_dot4_i32_i8(v, 0, acc), acc & MASK32)

    def test_identity_positive_one(self) -> None:
        """Dot product with 0x01010101 computes the sum of the four signed bytes."""
        test_cases = [
            (0x01020304, 4 + 3 + 2 + 1),
            (0xFFFFFFFF, -1 + -1 + -1 + -1),
            (0x7F7F7F7F, 127 * 4),
            (0x80808080, -128 * 4),
            (0x0000007F, 127),
            (0x00000080, -128),
            (0x7F8001FF, 127 - 128 + 1 - 1),
        ]
        for word, expected_sum in test_cases:
            with self.subTest(word=hex(word)):
                expected = (expected_sum) & MASK32
                self.assertEqual(ref_dot4_i32_i8(word, 0x01010101, 0), expected)
                self.assertEqual(ref_dot4_i32_i8(0x01010101, word, 0), expected)

    def test_identity_negative_one(self) -> None:
        """Dot product with 0xFFFFFFFF (all bytes -1) negates the sum of the four signed bytes."""
        test_cases = [
            (0x01020304, -(4 + 3 + 2 + 1)),
            (0xFFFFFFFF, -(-4)),
            (0x7F7F7F7F, -(127 * 4)),
            (0x80808080, -(-128 * 4)),
            (0x0000007F, -127),
            (0x00000080, 128),
        ]
        for word, expected_val in test_cases:
            with self.subTest(word=hex(word)):
                expected = expected_val & MASK32
                self.assertEqual(ref_dot4_i32_i8(word, 0xFFFFFFFF, 0), expected)
                self.assertEqual(ref_dot4_i32_i8(0xFFFFFFFF, word, 0), expected)

    def test_byte_lane_isolation(self) -> None:
        """Each of the four byte lanes operates independently without bleed across byte boundaries."""
        for lane in range(4):
            for val_a in (-128, -1, 0, 1, 127):
                for val_b in (-128, -1, 0, 1, 127):
                    s0 = (val_a & MASK8) << (8 * lane)
                    s1 = (val_b & MASK8) << (8 * lane)
                    expected = (val_a * val_b) & MASK32
                    with self.subTest(lane=lane, a=val_a, b=val_b):
                        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), expected)

    def test_addend_linearity(self) -> None:
        """ref_dot4_i32_i8(s0, s1, s2) == (ref_dot4_i32_i8(s0, s1, 0) + s2) mod 2^32."""
        s0, s1 = 0xFF80017F, 0x7F7F80FF
        base = ref_dot4_i32_i8(s0, s1, 0)
        for addend in (0, 1, 100, 0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
            with self.subTest(addend=hex(addend)):
                want = (base + addend) & MASK32
                got = ref_dot4_i32_i8(s0, s1, addend)
                self.assertEqual(got, want)

    def test_signed_vs_unsigned_divergence(self) -> None:
        """Signed dot product diverges completely from unsigned interpretation when sign bits are set."""
        divergent_pairs = [
            # 0xFF * 0x01: signed is -1 (0xFFFFFFFF), unsigned is 255
            (0x000000FF, 0x00000001, 0xFFFFFFFF, 255),
            # 0xFF * 0xFF: signed is 1, unsigned is 65025
            (0x000000FF, 0x000000FF, 1, 65025),
            # 0x80 * 0x01: signed is -128 (0xFFFFFF80), unsigned is 128
            (0x00000080, 0x00000001, 0xFFFFFF80, 128),
            # 0x80808080 . 0x7F7F7F7F: signed is -65024 (0xFFFF0200), unsigned is 65024
            (0x80808080, 0x7F7F7F7F, 0xFFFF0200, 65024),
            # 0xFFFFFFFF . 0x01010101: signed is -4 (0xFFFFFFFC), unsigned is 1020
            (0xFFFFFFFF, 0x01010101, 0xFFFFFFFC, 1020),
        ]
        for s0, s1, expected_signed, expected_unsigned in divergent_pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                signed_res = ref_dot4_i32_i8(s0, s1, 0)
                self.assertEqual(signed_res, expected_signed)
                self.assertNotEqual(signed_res, expected_unsigned)

    def test_extreme_min_max_byte_products(self) -> None:
        """Extreme signed byte product combinations."""
        # 4 * (-128 * -128) = 4 * 16384 = 65536
        self.assertEqual(ref_dot4_i32_i8(0x80808080, 0x80808080, 0), 65536)
        # 4 * (127 * 127) = 4 * 16129 = 64516
        self.assertEqual(ref_dot4_i32_i8(0x7F7F7F7F, 0x7F7F7F7F, 0), 64516)
        # 4 * (-128 * 127) = 4 * -16256 = -65024 = 0xFFFF0200
        self.assertEqual(ref_dot4_i32_i8(0x80808080, 0x7F7F7F7F, 0), 0xFFFF0200)
        # 4 * (-128 * -1) = 4 * 128 = 512
        self.assertEqual(ref_dot4_i32_i8(0x80808080, 0xFFFFFFFF, 0), 512)
        # 4 * (127 * -1) = 4 * -127 = -508 = 0xFFFFFE04
        self.assertEqual(ref_dot4_i32_i8(0x7F7F7F7F, 0xFFFFFFFF, 0), 0xFFFFFE04)

    def test_pseudorandom_fuzz_properties(self) -> None:
        """Deterministic property fuzzing over 100 pseudo-random 32-bit words."""
        rng = random.Random(0x444F5434)  # ASCII 'DOT4'
        for _ in range(100):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            s2 = rng.getrandbits(32)

            # Commutativity
            self.assertEqual(ref_dot4_i32_i8(s0, s1, s2), ref_dot4_i32_i8(s1, s0, s2))

            # Distributivity over unpacked signed byte terms
            b0 = unpack_i8(s0)
            b1 = unpack_i8(s1)
            manual_sum = ((s2 & MASK32) + b0[0] * b1[0] + b0[1] * b1[1] + b0[2] * b1[2] + b0[3] * b1[3]) & MASK32
            self.assertEqual(ref_dot4_i32_i8(s0, s1, s2), manual_sum)

            # Accumulator oracle equivalence
            self.assertEqual(ref_dot4c_i32_i8(s0, s1, s2), ref_dot4_i32_i8(s0, s1, s2))


class Dot4I8(unittest.TestCase):
    """End-to-end SPIR-V translator tests for v_dot4_i32_i8 and v_dot4c_i32_i8."""

    def test_the_reference_reproduces_the_hardware_words(self):
        for a, b, addend, measured in MEASURED:
            self.assertEqual(ref(a, b, addend), measured, f"{a:#x} . {b:#x} + {addend:#x}")

    def test_both_forms_give_the_hardware_words(self):
        for a, b, addend, measured in MEASURED:
            with self.subTest(a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                spv = support.translate_lines("ps_dot4_i8", program(a, b, addend))
                self.assertEqual(support.mrt0_words(spv), [measured] * 4)

    def test_the_accumulating_form_reads_its_destination(self):
        # v9 is never written before v_dot4c reads it: the old value is then the register's initial 0, not garbage
        # from an expression that dropped the addend. 70 is the dot product alone.
        lines = [
            "v_mov_b32_e32 v1, 0x01020304",
            "v_mov_b32_e32 v2, 0x05060708",
            "v_dot4c_i32_i8 v9, v1, v2",
            "v_dot4c_i32_i8 v9, v1, v2",
            "exp mrt0 v9, v9, v9, v9 done vm",
            "s_endpgm",
        ]
        self.assertEqual(support.mrt0_words(support.translate_lines("ps_dot4c_twice", lines)), [140] * 4)

    def test_the_bytes_are_signed(self):
        # 0xff * 0x01: -1 as a signed byte, 255 as an unsigned one.
        lines = [
            "v_mov_b32_e32 v1, 0x000000ff",
            "v_mov_b32_e32 v2, 0x00000001",
            "v_mov_b32_e32 v3, 0",
            "v_dot4_i32_i8 v8, v1, v2, v3",
            "exp mrt0 v8, v8, v8, v8 done vm",
            "s_endpgm",
        ]
        self.assertEqual(support.mrt0_words(support.translate_lines("ps_dot4_signed", lines)), [0xFFFFFFFF] * 4)

    def test_the_op_sel_reference_reproduces_the_hardware_words(self):
        for a, b, addend, measured in MEASURED_OP_SEL:
            self.assertEqual(ref_op_sel(a, b, addend), measured, f"{a:#x} . {b:#x} + {addend:#x}")
        # The table can tell the two readings apart: the plain rule gives another word in four of the five cases.
        self.assertEqual(sum(ref(a, b, c) != m for a, b, c, m in MEASURED_OP_SEL), 4)

    def test_op_sel_is_refused(self):
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 1",
            "v_mov_b32_e32 v3, 0",
            "v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
            "exp mrt0 v8, v8, v8, v8 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_dot4_op_sel", lines)
        self.assertEqual(str(refused.exception), "v_dot4_i32_i8 op_sel")

    def test_clamp_is_refused(self):
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 1",
            "v_mov_b32_e32 v3, 0",
            "v_dot4_i32_i8 v8, v1, v2, v3 clamp",
            "exp mrt0 v8, v8, v8, v8 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_dot4_clamp", lines)
        self.assertEqual(str(refused.exception), "v_dot4_i32_i8 clamp")


if __name__ == "__main__":
    unittest.main()

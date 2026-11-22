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
import spirv_wave_interp as wi  # noqa: E402
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
                expected = expected_sum & MASK32
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


class WavefrontDot4Interpreter:
    """Simulates an RDNA2 SIMD wavefront executing VALU/SALU dot-product instructions.

    Models:
      - Wave32 (32 lanes) and Wave64 (64 lanes).
      - Per-lane VGPR file: vgprs[reg][lane].
      - Uniform SGPR file: sgprs[reg].
      - EXEC mask: 32-bit or 64-bit integer mask.
      - Instructions: v_mov_b32, s_mov_b32, v_dot4_i32_i8, v_dot4c_i32_i8.
    """

    def __init__(self, wave_size: int = 32):
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size {wave_size}; must be 32 or 64")
        self.wave_size = wave_size
        self.vgprs = [[0] * wave_size for _ in range(256)]
        self.sgprs = [0] * 106
        self.exec_mask = (1 << wave_size) - 1

    def set_vgpr(self, reg: int, lane: int, val: int) -> None:
        if not (0 <= reg < 256):
            raise IndexError(f"VGPR index out of range: {reg}")
        self.vgprs[reg][lane] = val & MASK32

    def get_vgpr(self, reg: int, lane: int) -> int:
        if not (0 <= reg < 256):
            raise IndexError(f"VGPR index out of range: {reg}")
        return self.vgprs[reg][lane] & MASK32

    def set_sgpr(self, reg: int, val: int) -> None:
        if not (0 <= reg < 106):
            raise IndexError(f"SGPR index out of range: {reg}")
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        if not (0 <= reg < 106):
            raise IndexError(f"SGPR index out of range: {reg}")
        return self.sgprs[reg] & MASK32

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def parse_operand(self, token: str, lane: int) -> int:
        token = token.strip().rstrip(",")
        lower = token.lower()
        if lower.startswith("v") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 256:
                return self.vgprs[idx][lane] & MASK32
            raise IndexError(f"VGPR index out of range: {token}")
        if lower.startswith("s") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 106:
                return self.sgprs[idx] & MASK32
            raise IndexError(f"SGPR index out of range: {token}")
        if lower.startswith("0x") or lower.startswith("-0x"):
            return int(token, 16) & MASK32
        return int(token) & MASK32

    def execute_line(self, line: str) -> None:
        line = line.split(";", 1)[0].split("//", 1)[0].split("#", 1)[0].strip()
        if not line:
            return
        parts = line.split(None, 1)
        opcode = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""
        tokens = [t.strip() for t in args_str.split(",") if t.strip()]

        if opcode == "s_mov_b32":
            dst = tokens[0].lower()
            if not (dst.startswith("s") and dst[1:].isdigit()):
                raise ValueError(f"s_mov_b32 dst must be SGPR, got {tokens[0]}")
            s_idx = int(dst[1:])
            if not (0 <= s_idx < 106):
                raise IndexError(f"SGPR index out of range: {tokens[0]}")
            val = self.parse_operand(tokens[1], 0)
            self.sgprs[s_idx] = val

        elif opcode in ("v_mov_b32", "v_mov_b32_e32"):
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"v_mov_b32 dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            if not (0 <= v_idx < 256):
                raise IndexError(f"VGPR index out of range: {tokens[0]}")
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    val = self.parse_operand(tokens[1], lane)
                    self.vgprs[v_idx][lane] = val

        elif opcode == "v_dot4_i32_i8":
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"{opcode} dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            if not (0 <= v_idx < 256):
                raise IndexError(f"VGPR index out of range: {tokens[0]}")
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    s0 = self.parse_operand(tokens[1], lane)
                    s1 = self.parse_operand(tokens[2], lane)
                    s2 = self.parse_operand(tokens[3], lane) if len(tokens) > 3 else 0
                    self.vgprs[v_idx][lane] = ref_dot4_i32_i8(s0, s1, s2)

        elif opcode == "v_dot4c_i32_i8":
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"{opcode} dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            if not (0 <= v_idx < 256):
                raise IndexError(f"VGPR index out of range: {tokens[0]}")
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    old_acc = self.vgprs[v_idx][lane]
                    s0 = self.parse_operand(tokens[1], lane)
                    s1 = self.parse_operand(tokens[2], lane)
                    self.vgprs[v_idx][lane] = ref_dot4_i32_i8(s0, s1, old_acc)

        else:
            raise NotImplementedError(f"Unsupported instruction: {opcode}")

    def execute_lines(self, lines: list[str]) -> None:
        for line in lines:
            self.execute_line(line)


class WavefrontInterpreterTests(unittest.TestCase):
    """Multi-lane wavefront execution and SIMD divergence tests for signed 8-bit dot products."""

    def test_wave32_uniform_execution(self) -> None:
        """Wave32 wavefront with uniform inputs executes correctly across all 32 lanes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        lines = [
            "v_mov_b32_e32 v1, 0x01020304",
            "v_mov_b32_e32 v2, 0x05060708",
            "v_mov_b32_e32 v3, 0x000003E8",
            "v_dot4_i32_i8 v0, v1, v2, v3",
        ]
        interp.execute_lines(lines)
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 1070)

    def test_wave64_uniform_execution(self) -> None:
        """Wave64 wavefront with uniform inputs executes correctly across all 64 lanes."""
        interp = WavefrontDot4Interpreter(wave_size=64)
        lines = [
            "v_mov_b32_e32 v1, 0xFF80017F",
            "v_mov_b32_e32 v2, 0x7F7F80FF",
            "v_mov_b32_e32 v3, 0x00010000",
            "v_dot4_i32_i8 v0, v1, v2, v3",
        ]
        interp.execute_lines(lines)
        for lane in range(64):
            self.assertEqual(interp.get_vgpr(0, lane), 0x0000BF02)

    def test_wave32_per_lane_varying_execution(self) -> None:
        """Each lane in Wave32 computes an independent signed dot product."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        expected = []
        for lane in range(32):
            val_a = (lane - 16) & MASK8
            val_b = (127 - 4 * lane) & MASK8
            s0 = pack_i8(val_a, -(lane + 1), lane, 15)
            s1 = pack_i8(2, -3, 4, -1)
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)
            expected.append(ref_dot4_i32_i8(s0, s1, 0))

        interp.execute_line("v_dot4_i32_i8 v0, v1, v2, 0")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), expected[lane])

    def test_v_dot4c_accumulating_chain(self) -> None:
        """Accumulating dot product v_dot4c_i32_i8 chains properly on destination register."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(0, lane, 500)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)

        # First dot product adds 70 to 500 = 570
        interp.execute_line("v_dot4c_i32_i8 v0, v1, v2")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 570)

        # Second successive dot product adds another 70 = 640
        interp.execute_line("v_dot4c_i32_i8 v0, v1, v2")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 640)

    def test_inner_loop_reduction(self) -> None:
        """Simulate an inner-loop dot-product reduction accumulating across iterations."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        running_totals = [0] * 32
        steps = [
            (0x01010101, 0x02020202),  # 4 * 2 = 8
            (0x03030303, 0x04040404),  # 4 * 12 = 48
            (0x05050505, 0x06060606),  # 4 * 30 = 120
            (0x07070707, 0x08080808),  # 4 * 56 = 224
        ]
        for s0, s1 in steps:
            for lane in range(32):
                interp.set_vgpr(1, lane, s0)
                interp.set_vgpr(2, lane, s1)
                running_totals[lane] = (running_totals[lane] + ref_dot4_i32_i8(s0, s1, 0)) & MASK32
            interp.execute_line("v_dot4c_i32_i8 v0, v1, v2")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), running_totals[lane])
            self.assertEqual(interp.get_vgpr(0, lane), 8 + 48 + 120 + 224)

    def test_exec_mask_partial_divergence(self) -> None:
        """Inactive lanes under partial EXEC mask must strictly preserve destination registers."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        sentinel = 0xDEADBEEF

        for lane in range(32):
            interp.set_vgpr(5, lane, sentinel)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)
            interp.set_vgpr(3, lane, 1000)

        # Set EXEC to lower 16 lanes only (0x0000FFFF)
        interp.set_exec(0x0000FFFF)
        interp.execute_line("v_dot4_i32_i8 v5, v1, v2, v3")

        # Active lanes (0..15) updated to 1070 (70 + 1000)
        # Inactive lanes (16..31) strictly preserve sentinel 0xDEADBEEF
        for lane in range(32):
            if lane < 16:
                self.assertEqual(interp.get_vgpr(5, lane), 1070)
            else:
                self.assertEqual(interp.get_vgpr(5, lane), sentinel)

    def test_exec_mask_alternating_even_odd(self) -> None:
        """Divergent execution between alternating even and odd lanes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(10, lane, 0)
            interp.set_vgpr(1, lane, 0x01020304)  # dot = 70
            interp.set_vgpr(2, lane, 0x05060708)
            interp.set_vgpr(3, lane, 0x80808080)  # dot = 4 * (-128 * -1) = 512
            interp.set_vgpr(4, lane, 0xFFFFFFFF)

        # Step 1: Execute even lanes with (v1 . v2)
        interp.set_exec(0x55555555)  # even lanes
        interp.execute_line("v_dot4_i32_i8 v10, v1, v2, 0")

        # Step 2: Execute odd lanes with (v3 . v4)
        interp.set_exec(0xAAAAAAAA)  # odd lanes
        interp.execute_line("v_dot4_i32_i8 v10, v3, v4, 0")

        for lane in range(32):
            expected = 70 if (lane % 2 == 0) else 512
            self.assertEqual(interp.get_vgpr(10, lane), expected)

    def test_exec_mask_single_lane(self) -> None:
        """Only a single lane active (lane 19), ensuring no collateral writes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(7, lane, 0xBAADF00D)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)

        target_lane = 19
        interp.set_exec(1 << target_lane)
        interp.execute_line("v_dot4_i32_i8 v7, v1, v2, 0")

        for lane in range(32):
            if lane == target_lane:
                self.assertEqual(interp.get_vgpr(7, lane), 70)
            else:
                self.assertEqual(interp.get_vgpr(7, lane), 0xBAADF00D)

    def test_scalar_source_broadcast(self) -> None:
        """SGPR sources broadcast uniform values across all lanes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        interp.set_sgpr(4, 0x05060708)
        interp.set_sgpr(5, 5000)

        for lane in range(32):
            interp.set_vgpr(1, lane, 0x01020304)

        interp.execute_line("v_dot4_i32_i8 v0, v1, s4, s5")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 5070)


class SpirvWaveInterpTests(unittest.TestCase):
    """Multi-lane SIMD execution of translated shaders via spirv_wave_interp."""

    def test_spirv_wave32_execution(self) -> None:
        """Execute translated v_dot4_i32_i8 in Wave32 mode via spirv_wave_interp."""
        spv = support.translate_lines("ps_dot4_w32", program(0x01020304, 0x05060708, 1000))
        res = wi.run_wave(spv, wave=32)
        outputs = res["outputs"]
        self.assertEqual(len(outputs), 32)
        for lane in range(32):
            self.assertIsNotNone(outputs[lane])
            self.assertEqual(list(outputs[lane]["%out_mrt0"]), [1070] * 4)

    def test_spirv_wave64_execution(self) -> None:
        """Execute translated v_dot4_i32_i8 in Wave64 mode via spirv_wave_interp."""
        spv = support.translate_lines("ps_dot4_w64", program(0xFF80017F, 0x7F7F80FF, 0x00010000))
        res = wi.run_wave(spv, wave=64)
        outputs = res["outputs"]
        self.assertEqual(len(outputs), 64)
        for lane in range(64):
            self.assertIsNotNone(outputs[lane])
            self.assertEqual(list(outputs[lane]["%out_mrt0"]), [0x0000BF02] * 4)

    def test_spirv_wave_dot4c_accumulation(self) -> None:
        """Execute translated chained v_dot4c_i32_i8 in Wave32 mode."""
        lines = [
            "v_mov_b32_e32 v1, 0x01020304",
            "v_mov_b32_e32 v2, 0x05060708",
            "v_dot4c_i32_i8 v9, v1, v2",
            "v_dot4c_i32_i8 v9, v1, v2",
            "exp mrt0 v9, v9, v9, v9 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_dot4c_accum_w32", lines)
        res = wi.run_wave(spv, wave=32)
        outputs = res["outputs"]
        self.assertEqual(len(outputs), 32)
        for lane in range(32):
            self.assertIsNotNone(outputs[lane])
            self.assertEqual(list(outputs[lane]["%out_mrt0"]), [140] * 4)


class SignExtensionBoundaryTests(unittest.TestCase):
    """Detailed boundary checks for signed 8-bit integers and sign propagation."""

    def test_byte_sign_bit_boundary_transitions(self) -> None:
        """Transition between positive max 0x7F (+127) and negative min 0x80 (-128)."""
        # Testing transition across sign bit for each byte lane with multiplier 1
        for lane in range(4):
            # +127 * 1 = 127
            s0_pos = 0x7F << (8 * lane)
            s1 = 0x01 << (8 * lane)
            self.assertEqual(ref_dot4_i32_i8(s0_pos, s1, 0), 127)

            # -128 * 1 = -128 (0xFFFFFF80 in 32-bit two's complement)
            s0_neg = 0x80 << (8 * lane)
            self.assertEqual(ref_dot4_i32_i8(s0_neg, s1, 0), 0xFFFFFF80)

            # -1 * 1 = -1 (0xFFFFFFFF)
            s0_minus1 = 0xFF << (8 * lane)
            self.assertEqual(ref_dot4_i32_i8(s0_minus1, s1, 0), 0xFFFFFFFF)

            # 0 * 1 = 0
            s0_zero = 0x00 << (8 * lane)
            self.assertEqual(ref_dot4_i32_i8(s0_zero, s1, 0), 0)

    def test_individual_byte_sign_propagation(self) -> None:
        """Sign-extension of negative bytes must propagate through 32-bit addition."""
        # When multiplying by -1 (0xFF), sign flips:
        # (-128) * (-1) = +128
        for lane in range(4):
            s0 = 0x80 << (8 * lane)
            s1 = 0xFF << (8 * lane)
            self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 128)

        # When multiplying by +2:
        # (-128) * 2 = -256 = 0xFFFFFF00
        for lane in range(4):
            s0 = 0x80 << (8 * lane)
            s1 = 0x02 << (8 * lane)
            self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0xFFFFFF00)

    def test_mixed_signs_cancellation(self) -> None:
        """Positive and negative byte products cancel each other out to zero."""
        # lane 0: 50 * 2 = +100
        # lane 1: (-50) * 2 = -100
        # lane 2: 25 * 4 = +100
        # lane 3: (-25) * 4 = -100
        # Total = 100 - 100 + 100 - 100 = 0
        s0 = pack_i8(50, -50, 25, -25)
        s1 = pack_i8(2, 2, 4, 4)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 12345), 12345)

    def test_negative_times_negative_is_positive(self) -> None:
        """Product of two negative bytes produces a strictly positive product."""
        # 4 * (-128 * -128) = 4 * 16384 = 65536
        s0 = pack_i8(-128, -128, -128, -128)
        s1 = pack_i8(-128, -128, -128, -128)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 65536)

        # (-1)*(-5) + (-2)*(-6) + (-3)*(-7) + (-4)*(-8) = 5 + 12 + 21 + 32 = 70
        s0 = pack_i8(-1, -2, -3, -4)
        s1 = pack_i8(-5, -6, -7, -8)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 70)

    def test_negative_times_positive_is_negative(self) -> None:
        """Product of negative and positive bytes produces a negative sum."""
        # (-1)*5 + (-2)*6 + (-3)*7 + (-4)*8 = -5 - 12 - 21 - 32 = -70 = 0xFFFFFFBA
        s0 = pack_i8(-1, -2, -3, -4)
        s1 = pack_i8(5, 6, 7, 8)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0xFFFFFFBA)

        # 4 * (-128 * 1) = -512 = 0xFFFFFE00
        s0 = pack_i8(-128, -128, -128, -128)
        s1 = pack_i8(1, 1, 1, 1)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0xFFFFFE00)


class AccumulatorSaturationTests(unittest.TestCase):
    """Accumulator wrapping without clamp and saturation with clamp."""

    def test_unclamped_modulo32_upward_overflow(self) -> None:
        """Without clamp, positive dot product overflows past 2^32 and wraps cleanly."""
        # s0 = 0x7F7F7F7F, s1 = 0x7F7F7F7F -> dot product is +64516
        s0, s1 = 0x7F7F7F7F, 0x7F7F7F7F
        # 64516 + 0xFFFFFFFF = 64515 (mod 2^32)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0xFFFFFFFF), 64515)
        # 64516 + 0x7FFFFFFF = 0x8000FC03 (as measured on gfx1036)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0x7FFFFFFF), 0x8000FC03)
        # 64516 + (0xFFFFFFFF - 64516 + 1) = 0 (exact wrap to 0)
        acc_exact_wrap = (MASK32 - 64516 + 1) & MASK32
        self.assertEqual(ref_dot4_i32_i8(s0, s1, acc_exact_wrap), 0)

    def test_unclamped_modulo32_downward_underflow(self) -> None:
        """Without clamp, negative dot product underflows below 0 and wraps cleanly."""
        # s0 = 0x80808080, s1 = 0x7F7F7F7F -> dot product is -65024
        s0, s1 = 0x80808080, 0x7F7F7F7F
        # 5 + (-65024) = -65019 = 0xFFFF0205 (as measured on gfx1036)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0x00000005), 0xFFFF0205)
        # 0x80000000 + (-65024) = 0x7FFF0200 (as measured on gfx1036)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0x80000000), 0x7FFF0200)
        # 0 + (-65024) = 0xFFFF0200
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0xFFFF0200)

    def test_clamping_saturation_model_upward(self) -> None:
        """With clamp, overflowing positive totals saturate at INT32_MAX (0x7FFFFFFF)."""
        # 64516 + 0x7FFFFFFF saturates to 0x7FFFFFFF
        s0, s1 = 0x7F7F7F7F, 0x7F7F7F7F
        self.assertEqual(ref_dot4_clamp_i32_i8(s0, s1, 0x7FFFFFFF), 0x7FFFFFFF)
        # 70 + 0x7FFFFFFF saturates to 0x7FFFFFFF
        self.assertEqual(ref_dot4_clamp_i32_i8(0x01020304, 0x05060708, 0x7FFFFFFF), 0x7FFFFFFF)

    def test_clamping_saturation_model_downward(self) -> None:
        """With clamp, underflowing negative totals saturate at INT32_MIN (0x80000000)."""
        # -65024 + 0x80000000 saturates to 0x80000000
        s0, s1 = 0x80808080, 0x7F7F7F7F
        self.assertEqual(ref_dot4_clamp_i32_i8(s0, s1, 0x80000000), 0x80000000)
        # -16638 + 0x80000000 saturates to 0x80000000
        self.assertEqual(ref_dot4_clamp_i32_i8(0xFF80017F, 0x7F7F80FF, 0x80000000), 0x80000000)

    def test_clamping_saturation_within_bounds(self) -> None:
        """Within [-2^31, 2^31 - 1], clamp does not modify the result."""
        cases = [
            (0x01020304, 0x05060708, 0, 70),
            (0x01020304, 0x05060708, 1000, 1070),
            (0xFF80017F, 0x7F7F80FF, 0, 0xFFFFBF02),
            (0x80808080, 0x80808080, 0, 65536),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 3),
        ]
        for s0, s1, acc, expected in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1), acc=hex(acc)):
                self.assertEqual(ref_dot4_clamp_i32_i8(s0, s1, acc), expected)
                self.assertEqual(ref_dot4_clamp_i32_i8(s0, s1, acc), ref_dot4_i32_i8(s0, s1, acc))


class NegativeControlTests(unittest.TestCase):
    """Negative controls, unsupported modifier rejection, and fault injection."""

    def test_clamp_modifier_rejected_v_dot4(self) -> None:
        """The translator refuses clamp modifier on v_dot4_i32_i8."""
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 1",
            "v_mov_b32_e32 v3, 0",
            "v_dot4_i32_i8 v8, v1, v2, v3 clamp",
            "exp mrt0 v8, v8, v8, v8 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_dot4_clamp_neg", lines)
        self.assertEqual(str(refused.exception), "v_dot4_i32_i8 clamp")

    def test_clamp_modifier_rejected_v_dot4c(self) -> None:
        """The translator refuses clamp modifier on v_dot4c_i32_i8."""
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 1",
            "v_dot4c_i32_i8 v9, v1, v2 clamp",
            "exp mrt0 v9, v9, v9, v9 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_dot4c_clamp_neg", lines)
        self.assertEqual(str(refused.exception), "v_dot4c_i32_i8 clamp")

    def test_op_sel_modifier_rejected_v_dot4(self) -> None:
        """The translator refuses op_sel modifier on v_dot4_i32_i8."""
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 1",
            "v_mov_b32_e32 v3, 0",
            "v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
            "exp mrt0 v8, v8, v8, v8 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_dot4_op_sel_neg", lines)
        self.assertEqual(str(refused.exception), "v_dot4_i32_i8 op_sel")

    def test_op_sel_modifier_rejected_v_dot4c(self) -> None:
        """The translator refuses op_sel modifier on v_dot4c_i32_i8."""
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 1",
            "v_dot4c_i32_i8 v9, v1, v2 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
            "exp mrt0 v9, v9, v9, v9 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_dot4c_op_sel_neg", lines)
        self.assertEqual(str(refused.exception), "v_dot4c_i32_i8 op_sel")

    def test_interpreter_invalid_destination(self) -> None:
        """VALU instructions cannot write to SGPRs as destinations."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        with self.assertRaises(ValueError):
            interp.execute_line("v_dot4_i32_i8 s0, v1, v2, v3")
        with self.assertRaises(ValueError):
            interp.execute_line("v_dot4c_i32_i8 s0, v1, v2")
        with self.assertRaises(ValueError):
            interp.execute_line("s_mov_b32 v0, 1")

    def test_interpreter_out_of_bounds_registers(self) -> None:
        """VGPR index >= 256 or SGPR index >= 106 raises IndexError."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        with self.assertRaises(IndexError):
            interp.execute_line("v_dot4_i32_i8 v0, v256, v1, v2")
        with self.assertRaises(IndexError):
            interp.execute_line("v_dot4_i32_i8 v0, v1, s110, v2")
        with self.assertRaises(IndexError):
            interp.set_vgpr(256, 0, 1)
        with self.assertRaises(IndexError):
            interp.get_vgpr(256, 0)
        with self.assertRaises(IndexError):
            interp.set_sgpr(106, 1)
        with self.assertRaises(IndexError):
            interp.get_sgpr(106)

    def test_interpreter_unsupported_opcode(self) -> None:
        """Unknown opcodes raise NotImplementedError."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        with self.assertRaises(NotImplementedError):
            interp.execute_line("v_nonexistent_op v0, v1, v2")

    def test_interpreter_invalid_wave_size(self) -> None:
        """Wave sizes other than 32 or 64 are rejected."""
        with self.assertRaises(ValueError):
            WavefrontDot4Interpreter(wave_size=16)
        with self.assertRaises(ValueError):
            WavefrontDot4Interpreter(wave_size=128)

    def test_bit_corruption_sensitivity(self) -> None:
        """Flipping any bit in multiplier operands must perturb dot product result."""
        s0 = 0x01020304
        s1 = 0x05060708
        acc = 1000
        golden = ref_dot4_i32_i8(s0, s1, acc)

        for bit in range(32):
            corrupted_s0 = s0 ^ (1 << bit)
            corrupted_s1 = s1 ^ (1 << bit)
            self.assertNotEqual(
                ref_dot4_i32_i8(corrupted_s0, s1, acc),
                golden,
                f"Flipping bit {bit} in S0 produced no change",
            )
            self.assertNotEqual(
                ref_dot4_i32_i8(s0, corrupted_s1, acc),
                golden,
                f"Flipping bit {bit} in S1 produced no change",
            )


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

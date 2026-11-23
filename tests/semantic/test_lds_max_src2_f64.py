from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_max_src2_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision maxNum and AMD RDNA2 / GFX10
Local Data Share (LDS) dual-source atomic maximum semantics:
  - ds_max_src2_f64: Reads two 64-bit floating-point operands from LDS memory locations A and B,
    computes their double-precision maximum, and writes the result in-place into location A.
    Non-returning: VGPRs are not modified.

Syntax:
    ds_max_src2_f64 vaddr [offset0:offset8] [offset1:offset8] [gds]

Mathematical Specification:
    eff_addr_a = (vaddr + (offset0 & 0xFF) * 8) & 0xFFFFFFFF
    eff_addr_b = (vaddr + (offset1 & 0xFF) * 8) & 0xFFFFFFFF
    qword_idx_a = (eff_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (eff_addr_b >> 3) & (lds_qwords - 1)
    val_a = LDS64[qword_idx_a]
    val_b = LDS64[qword_idx_b]
    new_val_a = oracle_max_f64(val_a, val_b)
    LDS64[qword_idx_a] = new_val_a

Key Architectural Characteristics:
  1. Dual-Source LDS Operation (src2):
     Both input operands (val_a and val_b) are read from Local Data Share.
     Base byte address is provided in VGPR vaddr.
     offset0 and offset1 are unsigned 8-bit immediates scaled by 8 bytes (qword stride).
     Location B remains unaltered unless Location A and Location B alias to the same qword index.
  2. Non-Returning Semantics:
     The instruction does not return a value to the vector register file (no vdst operand).
     VGPRs remain unmodified throughout execution.
  3. IEEE 754-2008 / IEEE 754-2019 maxNum Semantics (binary64):
     - NaN Suppression: If one operand is NaN and the other is a valid float, the valid float
       is returned. If both operands are NaN, a quiet NaN is returned.
     - Signed Zero Ordering: -0.0 is strictly less than +0.0:
       max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0.
     - Infinities & Subnormals: -INF < all finite numbers < +INF.
       Subnormal double-precision values are preserved without flushing.
  4. LDS Capacity and Alignment:
     Standard allocation is 64 KiB (8,192 64-bit qwords) across 32 memory banks.
     Addresses are 8-byte aligned (low 3 bits dropped).

Standard library Python only; zero external dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
import enum
import math
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

# IEEE 754 binary64 architectural bitmasks
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET8_MASK: int = 0xFF
SIGN_MASK64: int = 0x8000000000000000
EXP_MASK64: int = 0x7FF0000000000000
FRAC_MASK64: int = 0x000FFFFFFFFFFFFF
QNAN_BIT64: int = 0x0008000000000000
CANONICAL_QNAN64: int = 0x7FF8000000000000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO64: int = 0x0000000000000000
NEG_ZERO64: int = 0x8000000000000000
ONE_F64: int = 0x3FF0000000000000           # 1.0
NEG_ONE_F64: int = 0xBFF0000000000000       # -1.0
HALF_F64: int = 0x3FE0000000000000          # 0.5
NEG_HALF_F64: int = 0xBFE0000000000000      # -0.5
TWO_F64: int = 0x4000000000000000           # 2.0
NEG_TWO_F64: int = 0xC000000000000000       # -2.0
THREE_F64: int = 0x4008000000000000         # 3.0
TEN_F64: int = 0x4024000000000000           # 10.0
NEG_TEN_F64: int = 0xC024000000000000       # -10.0

DBL_MIN: int = 0x0010000000000000           # Smallest positive normal: 2^-1022
NEG_DBL_MIN: int = 0x8010000000000000       # -2^-1022
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF           # Largest finite normal double
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF64: int = 0x7FF0000000000000         # +infinity
NEG_INF64: int = 0xFFF0000000000000         # -infinity
QNAN_PAYLOAD_A64: int = 0x7FF8000000000001  # Quiet NaN with payload A
QNAN_PAYLOAD_B64: int = 0xFFF8000000000002  # Quiet NaN with payload B
SNAN_PAYLOAD_A64: int = 0x7FF0000000000001  # Signaling NaN
SNAN_PAYLOAD_B64: int = 0xFFF0000000000002  # Signaling NaN
SUBNORM_MIN64: int = 0x0000000000000001     # Smallest positive subnormal: 2^-1074
SUBNORM_MAX64: int = 0x000FFFFFFFFFFFFF     # Largest positive subnormal
NEG_SUBNORM_MIN64: int = 0x8000000000000001
NEG_SUBNORM_MAX64: int = 0x800FFFFFFFFFFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB default capacity
DEFAULT_LDS_NUM_QWORDS: int = 8192   # 8,192 64-bit qwords
NUM_LDS_BANKS: int = 32              # 32 memory banks
DEFAULT_NUM_VGPRS: int = 256         # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106         # Standard GFX10 scalar register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword [0, 2^64 - 1]."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_MASK64) else val


def f64_to_bits(val: float) -> int:
    """Encode float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF64 if val > 0 else NEG_INF64


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_nan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double NaN."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == EXP_MASK64) and ((bits & FRAC_MASK64) != 0)


def is_qnan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double Quiet NaN."""
    return is_nan64(bits) and bool(bits & QNAN_BIT64)


def is_snan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double Signaling NaN."""
    return is_nan64(bits) and not bool(bits & QNAN_BIT64)


def is_inf64(bits: int) -> bool:
    """Return True if 64-bit integer represents IEEE 754 double positive or negative infinity."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == EXP_MASK64) and ((bits & FRAC_MASK64) == 0)


def is_subnormal64(bits: int) -> bool:
    """Return True if 64-bit integer represents a subnormal double-precision float."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == 0) and ((bits & FRAC_MASK64) != 0)


def is_zero64(bits: int) -> bool:
    """Return True if 64-bit integer represents positive or negative zero."""
    return (bits & ~SIGN_MASK64) == 0


def oracle_max_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_max_src2_f64 atomic operation.

    Follows IEEE 754-2008/2019 maxNum for binary64:
      - If one operand is NaN and the other is a valid float, returns the valid float.
      - If both are NaN, returns a quiet NaN (preserving payload or quietened).
      - Signed zero ordering: -0.0 < +0.0 (max(+0.0, -0.0) -> +0.0).
      - Subnormals and infinities preserved.
    """
    a = a_bits & MASK64
    b = b_bits & MASK64

    a_is_nan = is_nan64(a)
    b_is_nan = is_nan64(b)

    if a_is_nan and b_is_nan:
        return a | QNAN_BIT64
    if a_is_nan:
        return b
    if b_is_nan:
        return a

    # Signed zeros
    if is_zero64(a) and is_zero64(b):
        return POS_ZERO64 if ((a == POS_ZERO64) or (b == POS_ZERO64)) else NEG_ZERO64

    fa = bits_to_f64(a)
    fb = bits_to_f64(b)

    if fa > fb:
        return a
    elif fb > fa:
        return b
    else:
        # Numerically equal: positive sign bit is strictly larger (+0.0 > -0.0)
        if (a & SIGN_MASK64) == 0:
            return a
        if (b & SIGN_MASK64) == 0:
            return b
        return a


def calc_lds_src2_effective_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    capacity_qwords: int = DEFAULT_LDS_NUM_QWORDS,
) -> Tuple[int, int, int, int]:
    """Calculate effective addresses and qword indices for locations A and B.

    Offset values are unsigned 8-bit integers multiplied by 8 (qword stride).
    Returns (eff_addr_a, eff_addr_b, qword_idx_a, qword_idx_b).
    """
    eff_addr_a = (vaddr + (offset0 & OFFSET8_MASK) * 8) & MASK32
    eff_addr_b = (vaddr + (offset1 & OFFSET8_MASK) * 8) & MASK32
    qword_idx_a = (eff_addr_a >> 3) & (capacity_qwords - 1)
    qword_idx_b = (eff_addr_b >> 3) & (capacity_qwords - 1)
    return eff_addr_a, eff_addr_b, qword_idx_a, qword_idx_b


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


class LDSMemory64:
    """Architectural memory model simulating AMD RDNA2 Local Data Share (LDS) for 64-bit operations."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_qwords = size_bytes // 8
        self.qwords = [0] * self.num_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = (eff_addr >> 3) & (self.num_qwords - 1)
        return self.qwords[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = (eff_addr >> 3) & (self.num_qwords - 1)
        self.qwords[idx] = u64(val)

    def atomic_max_src2_f64(
        self,
        vaddr: int,
        offset0: int = 0,
        offset1: int = 0,
    ) -> int:
        """Execute atomic 64-bit double maximum from locations A and B, writing result into location A.

        Returns:
            new_val_a
        """
        _, _, idx_a, idx_b = calc_lds_src2_effective_addresses(vaddr, offset0, offset1, self.num_qwords)
        val_a = self.qwords[idx_a]
        val_b = self.qwords[idx_b]
        new_val_a = oracle_max_f64(val_a, val_b)
        self.qwords[idx_a] = new_val_a
        return new_val_a


class TestLDSMemoryAndOracle64(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS 64-bit dual-source memory model."""

    def test_oracle_standard_doubles(self) -> None:
        self.assertEqual(oracle_max_f64(ONE_F64, TWO_F64), TWO_F64)
        self.assertEqual(oracle_max_f64(TWO_F64, ONE_F64), TWO_F64)
        self.assertEqual(oracle_max_f64(TEN_F64, NEG_TWO_F64), TEN_F64)
        self.assertEqual(oracle_max_f64(NEG_TWO_F64, TEN_F64), TEN_F64)
        self.assertEqual(oracle_max_f64(NEG_TEN_F64, NEG_TWO_F64), NEG_TWO_F64)

    def test_oracle_signed_zeros(self) -> None:
        self.assertEqual(oracle_max_f64(POS_ZERO64, NEG_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(POS_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, NEG_ZERO64), NEG_ZERO64)

    def test_oracle_infinities(self) -> None:
        self.assertEqual(oracle_max_f64(POS_INF64, ONE_F64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, ONE_F64), ONE_F64)
        self.assertEqual(oracle_max_f64(POS_INF64, NEG_INF64), POS_INF64)

    def test_oracle_nan_suppression(self) -> None:
        self.assertEqual(oracle_max_f64(QNAN_PAYLOAD_A64, ONE_F64), ONE_F64)
        self.assertEqual(oracle_max_f64(ONE_F64, QNAN_PAYLOAD_A64), ONE_F64)
        self.assertEqual(oracle_max_f64(SNAN_PAYLOAD_A64, NEG_TEN_F64), NEG_TEN_F64)

        res_nan = oracle_max_f64(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64)
        self.assertTrue(is_qnan64(res_nan))

    def test_oracle_subnormals(self) -> None:
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, DBL_MIN), DBL_MIN)
        self.assertEqual(oracle_max_f64(SUBNORM_MAX64, SUBNORM_MIN64), SUBNORM_MAX64)
        self.assertEqual(oracle_max_f64(NEG_SUBNORM_MAX64, POS_ZERO64), POS_ZERO64)

    def test_lds_memory_initialization(self) -> None:
        mem = LDSMemory64(65536)
        self.assertEqual(mem.size_bytes, 65536)
        self.assertEqual(mem.num_qwords, 8192)

        for invalid_size in (0, -64, 1000, 30000, 65537):
            with self.assertRaises(ValueError):
                LDSMemory64(invalid_size)

    def test_lds_memory_dual_source_execution(self) -> None:
        mem = LDSMemory64(1024)
        # Location A at offset 0 (0x00), Location B at offset 1 (0x08)
        mem.write_qword(0x00, ONE_F64)
        mem.write_qword(0x08, TEN_F64)

        res = mem.atomic_max_src2_f64(vaddr=0x00, offset0=0, offset1=1)

        self.assertEqual(res, TEN_F64)
        self.assertEqual(mem.read_qword(0x00), TEN_F64)  # Location A updated to max(1.0, 10.0) = 10.0
        self.assertEqual(mem.read_qword(0x08), TEN_F64)  # Location B preserved

    def test_lds_memory_dual_source_same_location(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x10, TWO_F64)

        # Both offset0 and offset1 point to location A
        res = mem.atomic_max_src2_f64(vaddr=0x10, offset0=0, offset1=0)
        self.assertEqual(res, TWO_F64)
        self.assertEqual(mem.read_qword(0x10), TWO_F64)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(enum.Enum):
    """Data Share (DS) dual-source atomic opcodes."""
    DS_MAX_SRC2_F64 = "ds_max_src2_f64"


class RegType(enum.Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    count: int = 1
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index and self.index + self.count <= limit):
            raise ValueError(f"Register span out of range: {self.reg_type.value}{self.index} count={self.count}")
        if self.count <= 0:
            raise ValueError(f"Count must be positive, got {self.count}")

    @property
    def name(self) -> str:
        if self.count == 1:
            return f"{self.reg_type.value}{self.index}"
        return f"{self.reg_type.value}[{self.index}:{self.index + self.count - 1}]"


@dataclasses.dataclass(frozen=True)
class Offset8Immediate:
    """8-bit immediate qword offset descriptor."""
    offset: int

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET8_MASK):
            raise ValueError(f"Offset out of 8-bit range: {self.offset}")


@dataclasses.dataclass(frozen=True)
class DSMaxSrc2F64Instruction:
    """Symbolic AST node representing a ds_max_src2_f64 instruction."""
    vaddr: RegisterOperand
    offset0: int = 0
    offset1: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_MAX_SRC2_F64

    def __post_init__(self) -> None:
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be VGPR, got {self.vaddr.reg_type.value}")
        if self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a 32-bit register (count=1), got {self.vaddr.count}")
        if not (0 <= self.offset0 <= OFFSET8_MASK):
            raise ValueError(f"offset0 out of 8-bit range: {self.offset0}")
        if not (0 <= self.offset1 <= OFFSET8_MASK):
            raise ValueError(f"offset1 out of 8-bit range: {self.offset1}")

    def disasm(self) -> str:
        parts = [f"{self.opcode.value} {self.vaddr.name}"]
        if self.offset0 != 0:
            parts.append(f"offset0:{self.offset0}")
        if self.offset1 != 0:
            parts.append(f"offset1:{self.offset1}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)

    @classmethod
    def parse(cls, line: str) -> DSMaxSrc2F64Instruction:
        import re
        raw = line.split("//")[0].split(";")[0].strip()
        if not raw:
            raise ValueError("Empty instruction line")
        tokens = raw.replace(",", " ").split()
        if not tokens:
            raise ValueError("No tokens found")
        mnemonic = tokens[0].lower()
        if mnemonic != DSOpcode.DS_MAX_SRC2_F64.value:
            raise ValueError(f"Unknown opcode: {mnemonic}")

        if len(tokens) < 2:
            raise ValueError(f"Insufficient operands for {mnemonic}")

        m = re.match(r"^([vs])\[?(\d+)\]?$", tokens[1].lower())
        if not m:
            raise ValueError(f"Invalid register operand: {tokens[1]}")
        rtype = RegType.VGPR if m.group(1) == "v" else RegType.SGPR
        vaddr = RegisterOperand(int(m.group(2)), count=1, reg_type=rtype)

        offset0 = 0
        offset1 = 0
        gds = False
        for opt in tokens[2:]:
            opt_l = opt.lower()
            if opt_l == "gds":
                gds = True
            elif opt_l.startswith("offset0:"):
                offset0 = int(opt_l.split(":", 1)[1], 0)
            elif opt_l.startswith("offset1:"):
                offset1 = int(opt_l.split(":", 1)[1], 0)
            elif opt_l.startswith("offset:"):
                offset0 = int(opt_l.split(":", 1)[1], 0)
            else:
                raise ValueError(f"Unexpected token in instruction: {opt}")

        return cls(vaddr=vaddr, offset0=offset0, offset1=offset1, gds=gds)


class TestDSMaxSrc2F64AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_and_disasm(self) -> None:
        v2 = RegisterOperand(2, count=1, reg_type=RegType.VGPR)

        inst_simple = DSMaxSrc2F64Instruction(vaddr=v2)
        self.assertEqual(inst_simple.disasm(), "ds_max_src2_f64 v2")
        self.assertEqual(inst_simple.offset0, 0)
        self.assertEqual(inst_simple.offset1, 0)
        self.assertFalse(inst_simple.gds)

        inst_opts = DSMaxSrc2F64Instruction(vaddr=v2, offset0=1, offset1=2, gds=True)
        self.assertEqual(inst_opts.disasm(), "ds_max_src2_f64 v2 offset0:1 offset1:2 gds")

    def test_ast_parse_canonical(self) -> None:
        inst1 = DSMaxSrc2F64Instruction.parse("ds_max_src2_f64 v2")
        self.assertEqual(inst1.vaddr, RegisterOperand(2, count=1, reg_type=RegType.VGPR))
        self.assertEqual(inst1.offset0, 0)
        self.assertEqual(inst1.offset1, 0)

        inst2 = DSMaxSrc2F64Instruction.parse("ds_max_src2_f64 v5 offset0:4 offset1:8 gds")
        self.assertEqual(inst2.vaddr, RegisterOperand(5, count=1, reg_type=RegType.VGPR))
        self.assertEqual(inst2.offset0, 4)
        self.assertEqual(inst2.offset1, 8)
        self.assertTrue(inst2.gds)

    def test_ast_validation_errors(self) -> None:
        s2 = RegisterOperand(2, count=1, reg_type=RegType.SGPR)
        v_pair = RegisterOperand(0, count=2, reg_type=RegType.VGPR)
        v0 = RegisterOperand(0, count=1, reg_type=RegType.VGPR)

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            DSMaxSrc2F64Instruction(vaddr=s2)

        # vaddr cannot be 64-bit pair
        with self.assertRaises(ValueError):
            DSMaxSrc2F64Instruction(vaddr=v_pair)

        # Offsets out of 8-bit range
        with self.assertRaises(ValueError):
            DSMaxSrc2F64Instruction(vaddr=v0, offset0=-1)
        with self.assertRaises(ValueError):
            DSMaxSrc2F64Instruction(vaddr=v0, offset0=256)
        with self.assertRaises(ValueError):
            DSMaxSrc2F64Instruction(vaddr=v0, offset1=-1)
        with self.assertRaises(ValueError):
            DSMaxSrc2F64Instruction(vaddr=v0, offset1=256)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState64:
    """Architectural state for a single execution thread with scalar register file and 64-bit LDS."""

    def __init__(
        self,
        lds: Optional[LDSMemory64] = None,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
        num_sgprs: int = DEFAULT_NUM_SGPRS,
    ) -> None:
        self.lds: LDSMemory64 = lds if lds is not None else LDSMemory64()
        self.vgprs: List[int] = [0] * num_vgprs
        self.sgprs: List[int] = [0] * num_sgprs

    def get_vgpr32(self, idx: int) -> int:
        return self.vgprs[idx]

    def set_vgpr32(self, idx: int, val: int) -> None:
        self.vgprs[idx] = u32(val)


class ScalarDSInterpreter64:
    """Deterministic scalar interpreter executing dual-source 64-bit DS instructions."""

    def execute(self, state: ScalarThreadState64, inst: DSMaxSrc2F64Instruction) -> int:
        base_addr = state.get_vgpr32(inst.vaddr.index)
        new_val_a = state.lds.atomic_max_src2_f64(
            vaddr=base_addr,
            offset0=inst.offset0,
            offset1=inst.offset1,
        )
        return new_val_a


class TestScalarInterpreter64(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory64(DEFAULT_LDS_SIZE_BYTES)
        self.state = ScalarThreadState64(lds=self.lds)
        self.interp = ScalarDSInterpreter64()

    def test_scalar_execution_basic(self) -> None:
        # vaddr = 0x40. offset0 = 1 (0x48), offset1 = 2 (0x50)
        self.lds.write_qword(0x48, ONE_F64)
        self.lds.write_qword(0x50, TEN_F64)
        self.state.set_vgpr32(2, 0x40)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(2, count=1),
            offset0=1,
            offset1=2,
        )

        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, TEN_F64)
        self.assertEqual(self.lds.read_qword(0x48), TEN_F64)
        self.assertEqual(self.lds.read_qword(0x50), TEN_F64)
        # Verify vaddr VGPR is untouched (non-returning)
        self.assertEqual(self.state.get_vgpr32(2), 0x40)

    def test_scalar_execution_zero_offsets(self) -> None:
        self.lds.write_qword(0x100, TWO_F64)
        self.lds.write_qword(0x108, THREE_F64)
        self.state.set_vgpr32(0, 0x100)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(0, count=1),
            offset0=0,
            offset1=1,
        )

        res = self.interp.execute(self.state, inst)
        self.assertEqual(res, THREE_F64)
        self.assertEqual(self.lds.read_qword(0x100), THREE_F64)
        self.assertEqual(self.lds.read_qword(0x108), THREE_F64)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

class SIMDWaveState64:
    """Architectural state for a SIMD wavefront executing dual-source 64-bit LDS instructions."""

    def __init__(
        self,
        wave_size: int = 64,
        exec_mask: Optional[int] = None,
        lds: Optional[LDSMemory64] = None,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
    ) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.mask_limit: int = (1 << wave_size) - 1
        self.exec_mask: int = (self.mask_limit if exec_mask is None else exec_mask) & self.mask_limit
        self.lds: LDSMemory64 = lds if lds is not None else LDSMemory64()
        self.lane_vgprs: List[List[int]] = [[0] * num_vgprs for _ in range(wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            return False
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr32(self, lane: int, idx: int) -> int:
        return self.lane_vgprs[lane][idx]

    def set_vgpr32(self, lane: int, idx: int, val: int) -> None:
        self.lane_vgprs[lane][idx] = u32(val)


class SIMDDSInterpreter64:
    """SIMD wavefront interpreter handling divergence and EXEC masking for dual-source DS instructions."""

    def execute(
        self,
        wave: SIMDWaveState64,
        inst: DSMaxSrc2F64Instruction,
    ) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * wave.wave_size

        for lane in range(wave.wave_size):
            if not wave.is_lane_active(lane):
                continue

            base_addr = wave.get_vgpr32(lane, inst.vaddr.index)
            new_val_a = wave.lds.atomic_max_src2_f64(
                vaddr=base_addr,
                offset0=inst.offset0,
                offset1=inst.offset1,
            )
            results[lane] = new_val_a

        return results

    def detect_bank_conflicts(
        self,
        wave: SIMDWaveState64,
        inst: DSMaxSrc2F64Instruction,
    ) -> Dict[int, List[int]]:
        bank_map: Dict[int, Dict[int, List[int]]] = {}
        for lane in range(wave.wave_size):
            if not wave.is_lane_active(lane):
                continue
            base_addr = wave.get_vgpr32(lane, inst.vaddr.index)
            eff_a, eff_b, q_a, q_b = calc_lds_src2_effective_addresses(
                base_addr, inst.offset0, inst.offset1, wave.lds.num_qwords
            )

            for eff, qword in ((eff_a, q_a), (eff_b, q_b)):
                bank0 = calc_lds_bank(eff)
                bank1 = calc_lds_bank(eff + 4)
                for b in (bank0, bank1):
                    if b not in bank_map:
                        bank_map[b] = {}
                    if qword not in bank_map[b]:
                        bank_map[b][qword] = []
                    bank_map[b][qword].append(lane)

        conflicts: Dict[int, List[int]] = {}
        for b, qwords in bank_map.items():
            if len(qwords) > 1:
                all_lanes: List[int] = []
                for lane_list in qwords.values():
                    all_lanes.extend(lane_list)
                conflicts[b] = sorted(list(set(all_lanes)))

        return conflicts


class TestSIMDInterpreter64(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_full_exec_independent_addresses(self) -> None:
        lds = LDSMemory64(65536)
        wave = SIMDWaveState64(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter64()

        # Each lane has an independent slice of 2 qwords:
        # location A at lane*16, location B at lane*16 + 8
        for lane in range(32):
            base = lane * 16
            lds.write_qword(base, ONE_F64)
            lds.write_qword(base + 8, f64_to_bits(float(lane + 2)))
            wave.set_vgpr32(lane, 2, base)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(2, count=1),
            offset0=0,
            offset1=1,
        )

        results = interp.execute(wave, inst)

        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = oracle_max_f64(ONE_F64, f64_to_bits(float(lane + 2)))
            self.assertEqual(results[lane], expected)
            self.assertEqual(lds.read_qword(lane * 16), expected)

    def test_simd_exec_mask_inactive_lanes(self) -> None:
        lds = LDSMemory64(65536)
        even_mask = 0x5555555555555555
        wave = SIMDWaveState64(wave_size=64, exec_mask=even_mask, lds=lds)
        interp = SIMDDSInterpreter64()

        for lane in range(64):
            base = lane * 16
            lds.write_qword(base, ONE_F64)
            lds.write_qword(base + 8, TEN_F64)
            wave.set_vgpr32(lane, 2, base)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(2, count=1),
            offset0=0,
            offset1=1,
        )

        results = interp.execute(wave, inst)

        for lane in range(64):
            base = lane * 16
            if lane % 2 == 0:
                self.assertEqual(results[lane], TEN_F64)
                self.assertEqual(lds.read_qword(base), TEN_F64)
            else:
                self.assertIsNone(results[lane])
                # Inactive lane: memory location A remains untouched (1.0)
                self.assertEqual(lds.read_qword(base), ONE_F64)

    def test_simd_zero_exec_mask(self) -> None:
        lds = LDSMemory64(65536)
        wave = SIMDWaveState64(wave_size=32, exec_mask=0, lds=lds)
        interp = SIMDDSInterpreter64()

        lds.write_qword(0, ONE_F64)
        lds.write_qword(8, TEN_F64)
        wave.set_vgpr32(0, 2, 0)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(2, count=1),
            offset0=0,
            offset1=1,
        )

        results = interp.execute(wave, inst)
        self.assertTrue(all(r is None for r in results))
        self.assertEqual(lds.read_qword(0), ONE_F64)

    def test_bank_conflict_detection(self) -> None:
        lds = LDSMemory64(65536)
        wave = SIMDWaveState64(wave_size=32, exec_mask=0x3, lds=lds)
        interp = SIMDDSInterpreter64()

        wave.set_vgpr32(0, 2, 0)
        wave.set_vgpr32(1, 2, 128)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(2, count=1),
            offset0=0,
            offset1=1,
        )

        conflicts = interp.detect_bank_conflicts(wave, inst)
        self.assertIn(0, conflicts)
        self.assertEqual(conflicts[0], [0, 1])


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing64(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory64(65536)
        self.interp = ScalarDSInterpreter64()

    def test_vaddr_preservation_non_returning(self) -> None:
        # Non-returning ds_max_src2_f64 must never overwrite vaddr
        state = ScalarThreadState64(lds=self.lds)
        state.set_vgpr32(5, 0x80)
        self.lds.write_qword(0x80, ONE_F64)
        self.lds.write_qword(0x88, TEN_F64)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(5, count=1),
            offset0=0,
            offset1=1,
        )

        self.interp.execute(state, inst)

        # v5 must strictly remain 0x80
        self.assertEqual(state.get_vgpr32(5), 0x80)
        self.assertEqual(self.lds.read_qword(0x80), TEN_F64)

    def test_all_vgprs_isolated(self) -> None:
        state = ScalarThreadState64(lds=self.lds)
        for i in range(256):
            state.set_vgpr32(i, 0x55550000 + i)

        addr = 0x20
        state.set_vgpr32(10, addr)
        self.lds.write_qword(addr, TWO_F64)
        self.lds.write_qword(addr + 8, THREE_F64)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(10, count=1),
            offset0=0,
            offset1=1,
        )

        self.interp.execute(state, inst)

        for i in range(256):
            if i == 10:
                self.assertEqual(state.get_vgpr32(i), addr)
            else:
                self.assertEqual(state.get_vgpr32(i), 0x55550000 + i)

    def test_location_aliasing_same_offset(self) -> None:
        # When offset0 == offset1, Location A and B alias to the exact same cell
        state = ScalarThreadState64(lds=self.lds)
        state.set_vgpr32(0, 0x100)
        self.lds.write_qword(0x100, HALF_F64)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(0, count=1),
            offset0=0,
            offset1=0,
        )

        res = self.interp.execute(state, inst)
        self.assertEqual(res, HALF_F64)
        self.assertEqual(self.lds.read_qword(0x100), HALF_F64)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class Float64ConditionCodes:
    """Architectural condition codes evaluating double-precision float maximum results."""
    zero: bool
    negative: bool
    nan: bool
    infinity: bool
    subnormal: bool
    updated: bool


@dataclasses.dataclass(frozen=True)
class LDSStatusFlags64:
    """Status flags for 64-bit LDS addressing and dual-bank mapping."""
    qword_aligned: bool
    address_wrapped: bool
    bank_span: Tuple[int, int]


def compute_float64_condition_codes(old_val_a: int, val_b: int, result: int) -> Float64ConditionCodes:
    """Compute condition codes from double-precision float atomic max result."""
    res_u = u64(result)
    old_u = u64(old_val_a)
    return Float64ConditionCodes(
        zero=is_zero64(res_u),
        negative=bool(res_u & SIGN_MASK64),
        nan=is_nan64(res_u),
        infinity=is_inf64(res_u),
        subnormal=is_subnormal64(res_u),
        updated=(res_u != old_u),
    )


def compute_lds64_src2_status_flags(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_size_bytes: int = DEFAULT_LDS_SIZE_BYTES,
) -> Tuple[LDSStatusFlags64, LDSStatusFlags64]:
    """Evaluate 8-byte alignment, dual-bank span, and boundary wrap for both locations."""
    eff_a, eff_b, _, _ = calc_lds_src2_effective_addresses(vaddr, offset0, offset1, lds_size_bytes // 8)
    st_a = LDSStatusFlags64(
        qword_aligned=((eff_a & 7) == 0),
        address_wrapped=(eff_a >= lds_size_bytes),
        bank_span=(calc_lds_bank(eff_a), calc_lds_bank(eff_a + 4)),
    )
    st_b = LDSStatusFlags64(
        qword_aligned=((eff_b & 7) == 0),
        address_wrapped=(eff_b >= lds_size_bytes),
        bank_span=(calc_lds_bank(eff_b), calc_lds_bank(eff_b + 4)),
    )
    return st_a, st_b


class TestConditionCodesAndStatusFlags64(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and update detection."""

    def test_condition_code_zero(self) -> None:
        cc_pos = compute_float64_condition_codes(NEG_ONE_F64, POS_ZERO64, POS_ZERO64)
        self.assertTrue(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.updated)

    def test_condition_code_negative(self) -> None:
        cc = compute_float64_condition_codes(NEG_TEN_F64, NEG_TWO_F64, NEG_TWO_F64)
        self.assertFalse(cc.zero)
        self.assertTrue(cc.negative)
        self.assertTrue(cc.updated)

    def test_condition_code_nan(self) -> None:
        res_nan = oracle_max_f64(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64)
        cc = compute_float64_condition_codes(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64, res_nan)
        self.assertTrue(cc.nan)

    def test_condition_code_infinity(self) -> None:
        cc_pos_inf = compute_float64_condition_codes(TEN_F64, POS_INF64, POS_INF64)
        self.assertTrue(cc_pos_inf.infinity)
        self.assertTrue(cc_pos_inf.updated)

    def test_condition_code_subnormal(self) -> None:
        cc = compute_float64_condition_codes(NEG_ONE_F64, SUBNORM_MIN64, SUBNORM_MIN64)
        self.assertTrue(cc.subnormal)
        self.assertTrue(cc.updated)

    def test_condition_code_updated_flag(self) -> None:
        cc1 = compute_float64_condition_codes(ONE_F64, TEN_F64, TEN_F64)
        self.assertTrue(cc1.updated)

        cc2 = compute_float64_condition_codes(TEN_F64, ONE_F64, TEN_F64)
        self.assertFalse(cc2.updated)

    def test_lds_status_flags(self) -> None:
        st_a, st_b = compute_lds64_src2_status_flags(0x100, offset0=0, offset1=1)
        self.assertTrue(st_a.qword_aligned)
        self.assertTrue(st_b.qword_aligned)
        self.assertFalse(st_a.address_wrapped)
        self.assertFalse(st_b.address_wrapped)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BoundaryExtremes64(unittest.TestCase):
    """Phase 7: Unit tests for double-precision IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zeros_comprehensive(self) -> None:
        self.assertEqual(oracle_max_f64(POS_ZERO64, NEG_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(POS_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, NEG_ZERO64), NEG_ZERO64)

    def test_infinities_comprehensive(self) -> None:
        self.assertEqual(oracle_max_f64(POS_INF64, POS_INF64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, NEG_INF64), NEG_INF64)
        self.assertEqual(oracle_max_f64(POS_INF64, NEG_INF64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, POS_INF64), POS_INF64)
        self.assertEqual(oracle_max_f64(POS_INF64, DBL_MAX), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, NEG_DBL_MAX), NEG_DBL_MAX)

    def test_subnormals_preservation_and_comparison(self) -> None:
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, DBL_MIN), DBL_MIN)
        self.assertNotEqual(oracle_max_f64(SUBNORM_MIN64, DBL_MIN), POS_ZERO64)

        self.assertEqual(oracle_max_f64(SUBNORM_MAX64, DBL_MIN), DBL_MIN)
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, POS_ZERO64), SUBNORM_MIN64)
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, NEG_ZERO64), SUBNORM_MIN64)

        self.assertEqual(oracle_max_f64(NEG_SUBNORM_MIN64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_SUBNORM_MAX64, NEG_SUBNORM_MIN64), NEG_SUBNORM_MIN64)

    def test_nan_suppression_comprehensive(self) -> None:
        for normal_val in (ONE_F64, NEG_ONE_F64, POS_INF64, NEG_INF64, POS_ZERO64, NEG_ZERO64, SUBNORM_MIN64):
            with self.subTest(val=hex(normal_val)):
                self.assertEqual(oracle_max_f64(QNAN_PAYLOAD_A64, normal_val), normal_val)
                self.assertEqual(oracle_max_f64(normal_val, QNAN_PAYLOAD_A64), normal_val)

        for normal_val in (TWO_F64, NEG_TWO_F64, DBL_MAX, NEG_DBL_MAX):
            with self.subTest(val=hex(normal_val)):
                self.assertEqual(oracle_max_f64(SNAN_PAYLOAD_A64, normal_val), normal_val)
                self.assertEqual(oracle_max_f64(normal_val, SNAN_PAYLOAD_A64), normal_val)

    def test_dual_nans(self) -> None:
        q_q = oracle_max_f64(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64)
        self.assertTrue(is_qnan64(q_q))

        s_s = oracle_max_f64(SNAN_PAYLOAD_A64, SNAN_PAYLOAD_B64)
        self.assertTrue(is_qnan64(s_s))

        q_s = oracle_max_f64(QNAN_PAYLOAD_A64, SNAN_PAYLOAD_A64)
        self.assertTrue(is_qnan64(q_s))

    def test_floating_point_normal_extremes(self) -> None:
        self.assertEqual(oracle_max_f64(DBL_MAX, NEG_DBL_MAX), DBL_MAX)
        self.assertEqual(oracle_max_f64(NEG_DBL_MAX, DBL_MAX), DBL_MAX)
        self.assertEqual(oracle_max_f64(DBL_MIN, NEG_DBL_MIN), DBL_MIN)
        self.assertEqual(oracle_max_f64(NEG_DBL_MIN, DBL_MIN), DBL_MIN)


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutations64(unittest.TestCase):
    """Phase 8: Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_assembly(self) -> None:
        invalid_assemblies = [
            "",
            "   ",
            "v_max_f64 v0",
            "ds_max_src2_f32 v0",
            "ds_min_src2_f64 v0",
            "ds_max_src2_f64",
            "ds_max_src2_f64 s0",
            "ds_max_src2_f64 r0",
            "ds_max_src2_f64 v256",
            "ds_max_src2_f64 v0 offset0:300",
            "ds_max_src2_f64 v0 offset1:300",
            "ds_max_src2_f64 v0 offset0:-1",
            "ds_max_src2_f64 v0 bad_token",
        ]
        for asm in invalid_assemblies:
            with self.subTest(asm=asm):
                with self.assertRaises((ValueError, IndexError)):
                    DSMaxSrc2F64Instruction.parse(asm)

    def test_architectural_model_negative_controls(self) -> None:
        with self.assertRaises(ValueError):
            LDSMemory64(0)
        with self.assertRaises(ValueError):
            LDSMemory64(-1024)
        with self.assertRaises(ValueError):
            LDSMemory64(1000)

        with self.assertRaises(ValueError):
            SIMDWaveState64(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWaveState64(wave_size=128)

    def test_mutation_detection_inverted_min_max(self) -> None:
        def mutant_max(a: int, b: int) -> int:
            fa = bits_to_f64(a)
            fb = bits_to_f64(b)
            return a if fa < fb else b

        self.assertNotEqual(mutant_max(ONE_F64, TWO_F64), oracle_max_f64(ONE_F64, TWO_F64))

    def test_mutation_detection_integer_comparison(self) -> None:
        def mutant_max_uint(a: int, b: int) -> int:
            return a if (a & MASK64) > (b & MASK64) else b

        self.assertNotEqual(
            mutant_max_uint(ONE_F64, NEG_TWO_F64),
            oracle_max_f64(ONE_F64, NEG_TWO_F64),
        )

    def test_mutation_detection_flush_subnormals_to_zero(self) -> None:
        def mutant_max_ftz(a: int, b: int) -> int:
            a_clean = POS_ZERO64 if is_subnormal64(a) else a
            b_clean = POS_ZERO64 if is_subnormal64(b) else b
            return oracle_max_f64(a_clean, b_clean)

        self.assertNotEqual(
            mutant_max_ftz(SUBNORM_MIN64, NEG_ONE_F64),
            oracle_max_f64(SUBNORM_MIN64, NEG_ONE_F64),
        )

    def test_mutation_detection_nan_propagation(self) -> None:
        def mutant_max_nan_propagate(a: int, b: int) -> int:
            if is_nan64(a):
                return a
            if is_nan64(b):
                return b
            return oracle_max_f64(a, b)

        self.assertNotEqual(
            mutant_max_nan_propagate(QNAN_PAYLOAD_A64, ONE_F64),
            oracle_max_f64(QNAN_PAYLOAD_A64, ONE_F64),
        )

    def test_mutation_detection_ignore_exec_mask(self) -> None:
        class MutantSIMDInterpreter64(SIMDDSInterpreter64):
            def execute(self, wave: SIMDWaveState64, inst: DSMaxSrc2F64Instruction) -> List[Optional[int]]:
                saved_mask = wave.exec_mask
                wave.exec_mask = wave.mask_limit
                try:
                    return super().execute(wave, inst)
                finally:
                    wave.exec_mask = saved_mask

        lds = LDSMemory64(65536)
        lds.write_qword(0, ONE_F64)
        lds.write_qword(8, TEN_F64)
        wave = SIMDWaveState64(wave_size=32, exec_mask=0, lds=lds)
        for l in range(32):
            wave.set_vgpr32(l, 2, l * 16)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(2, count=1),
            offset0=0,
            offset1=1,
        )
        mutant = MutantSIMDInterpreter64()
        mutant.execute(wave, inst)

        self.assertEqual(lds.read_qword(0), TEN_F64)

        lds.write_qword(0, ONE_F64)
        SIMDDSInterpreter64().execute(wave, inst)
        self.assertEqual(lds.read_qword(0), ONE_F64)

    def test_mutation_detection_swapped_offsets(self) -> None:
        class SwappedOffsetInterpreter64(ScalarDSInterpreter64):
            def execute(self, state: ScalarThreadState64, inst: DSMaxSrc2F64Instruction) -> int:
                swapped = DSMaxSrc2F64Instruction(
                    vaddr=inst.vaddr,
                    offset0=inst.offset1,
                    offset1=inst.offset0,
                    gds=inst.gds,
                )
                return super().execute(state, swapped)

        lds = LDSMemory64(65536)
        lds.write_qword(0x20, ONE_F64)   # offset0: A
        lds.write_qword(0x28, TEN_F64)   # offset1: B
        state = ScalarThreadState64(lds=lds)
        state.set_vgpr32(2, 0x20)

        inst = DSMaxSrc2F64Instruction(
            vaddr=RegisterOperand(2, count=1),
            offset0=0,
            offset1=1,
        )

        mutant = SwappedOffsetInterpreter64()
        mutant.execute(state, inst)

        # Mutant updated location B (0x28) instead of location A (0x20)
        self.assertEqual(lds.read_qword(0x20), ONE_F64)
        self.assertEqual(lds.read_qword(0x28), TEN_F64)


if __name__ == "__main__":
    unittest.main()

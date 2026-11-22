from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_sub_rtn_f32.

Opcode:
  - ds_sub_rtn_f32 vdst, vaddr, vdata [offset:N]
      Atomic 32-bit single-precision floating-point subtraction on LDS memory with
      destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[idx]
        LDS[idx] = IEEE754_sub(old_val, vdata)
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in VGPR vaddr + optional unsigned 16-bit immediate byte offset.
  - Floating-point arithmetic: IEEE 754-2008 / 2019 binary32 single-precision rules.
  - Return value: Prior 32-bit value stored in memory at the targeted location is atomically
    returned and written to destination VGPR vdst.
"""

from dataclasses import dataclass
from enum import Enum
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB LDS capacity in dwords
DEFAULT_NUM_LANES: int = 64
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
NUM_LDS_BANKS: int = 32

# IEEE 754 binary32 Architectural Constants and Bit Patterns
SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000
CANONICAL_QNAN: int = 0x7FC00000

POS_ZERO_F32: int = 0x00000000       # +0.0
NEG_ZERO_F32: int = 0x80000000       # -0.0
ONE_F32: int = 0x3F800000            # 1.0
NEG_ONE_F32: int = 0xBF800000        # -1.0
TWO_F32: int = 0x40000000            # 2.0
NEG_TWO_F32: int = 0xC0000000        # -2.0
THREE_F32: int = 0x40400000          # 3.0
FIVE_F32: int = 0x40A00000           # 5.0
TEN_F32: int = 0x41200000            # 10.0
POS_INF_F32: int = 0x7F800000        # +inf
NEG_INF_F32: int = 0xFF800000        # -inf
QNAN_F32: int = 0x7FC00000           # Quiet NaN
SNAN_F32: int = 0x7F800001           # Signaling NaN
FLT_MIN: int = 0x00800000            # 2^-126
FLT_MAX: int = 0x7F7FFFFF            # (2 - 2^-23) * 2^127
SUBNORM_MIN: int = 0x00000001        # Smallest positive subnormal: 2^-149
SUBNORM_MAX: int = 0x007FFFFF        # Largest positive subnormal: (1 - 2^-23) * 2^-126


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def f32_to_bits(val: float) -> int:
    """Encode Python float to IEEE 754 single-precision 32-bit bit pattern."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit bit pattern to IEEE 754 single-precision Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_inf(bits: int) -> bool:
    """Return True if bits represent IEEE 754 single-precision infinity."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent IEEE 754 single-precision +0.0 or -0.0."""
    return (bits & ~SIGN_MASK) == 0


def is_subnormal(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormalized) float."""
    bits &= MASK32
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def oracle_sub_f32(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for IEEE 754 single-precision subtraction (a - b).

    Returns 32-bit unsigned integer bit pattern of the result.
    """
    a_bits &= MASK32
    b_bits &= MASK32

    # NaN propagation
    if is_nan(a_bits) or is_nan(b_bits):
        return CANONICAL_QNAN

    # Infinity arithmetic
    if is_inf(a_bits) and is_inf(b_bits):
        # inf - inf of same sign is NaN (+inf - +inf = NaN, -inf - -inf = NaN)
        if (a_bits & SIGN_MASK) == (b_bits & SIGN_MASK):
            return CANONICAL_QNAN
        return a_bits
    if is_inf(a_bits):
        return a_bits
    if is_inf(b_bits):
        # a - (+inf) = -inf, a - (-inf) = +inf
        return NEG_INF_F32 if (b_bits == POS_INF_F32) else POS_INF_F32

    # Zeros and equal operands
    if is_zero(a_bits) and is_zero(b_bits):
        # (+0.0) - (+0.0) = +0.0
        # (-0.0) - (-0.0) = +0.0
        # (+0.0) - (-0.0) = +0.0
        # (-0.0) - (+0.0) = -0.0
        if a_bits == NEG_ZERO_F32 and b_bits == POS_ZERO_F32:
            return NEG_ZERO_F32
        return POS_ZERO_F32

    fa = bits_to_f32(a_bits)
    fb = bits_to_f32(b_bits)
    res_f = fa - fb
    res_bits = f32_to_bits(res_f)

    # When x - x yields zero in round-to-nearest-even, the sign is positive
    if is_zero(res_bits) and fa == fb:
        return POS_ZERO_F32

    return res_bits


class LDSMemory:
    """Model of Local Data Share memory storage."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data: list[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)


def ref_ds_sub_rtn_f32(
    mem: LDSMemory,
    vaddr: int,
    vdata_bits: int,
    offset: int = 0,
) -> int:
    """Reference mathematical oracle for ds_sub_rtn_f32.

    Atomically computes:
      old_val = LDS[idx]
      LDS[idx] = old_val - vdata (IEEE 754 single precision)
      return old_val
    """
    effective_byte = u32(vaddr + offset)
    idx = (effective_byte >> 2) & (mem.size_dwords - 1)
    old_val = mem.read_dword(idx)
    new_val = oracle_sub_f32(old_val, vdata_bits)
    mem.write_dword(idx, new_val)
    return old_val


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_SUB_RTN_F32 = "ds_sub_rtn_f32"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}[{self.index}] "
                f"(expected 0 <= index < {max_limit})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}[{self.index}]"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Immediate:
    """32-bit immediate constant operand."""
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", u32(self.val))

    def __str__(self) -> str:
        return f"0x{self.val:X}"


Operand = Union[RegisterOperand, Immediate]


@dataclass(frozen=True)
class DSSubRtnF32Instruction:
    """Symbolic AST node for ds_sub_rtn_f32 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: Operand
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_SUB_RTN_F32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be a VGPR, got {self.vdst.reg_type}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be a VGPR, got {self.vaddr.reg_type}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(
                f"offset out of bounds: {self.offset} (expected 0 <= offset <= {OFFSET16_MASK})"
            )

    def disassemble(self) -> str:
        """Emit canonical AMD assembly string."""
        parts = [self.opcode.value, str(self.vdst), str(self.vaddr), str(self.vdata0)]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return f"{parts[0]} {', '.join(parts[1:])}"

    @classmethod
    def parse(cls, asm: str) -> DSSubRtnF32Instruction:
        """Parse canonical AMD assembly string into AST."""
        tokens = [t.strip().rstrip(",") for t in asm.strip().split() if t.strip()]
        if not tokens:
            raise ValueError("Empty assembly string")
        op = tokens[0]
        if op != DSOpcode.DS_SUB_RTN_F32.value:
            raise ValueError(f"Unsupported opcode: {op}")
        if len(tokens) < 4:
            raise ValueError(f"Instruction requires vdst, vaddr, vdata operands: {asm}")

        def parse_reg(token: str) -> RegisterOperand:
            clean = token.rstrip(",")
            if clean.startswith("v[") and clean.endswith("]"):
                idx = int(clean[2:-1])
                return RegisterOperand(idx, RegType.VGPR)
            if clean.startswith("v") and clean[1:].isdigit():
                return RegisterOperand(int(clean[1:]), RegType.VGPR)
            if clean.startswith("s[") and clean.endswith("]"):
                idx = int(clean[2:-1])
                return RegisterOperand(idx, RegType.SGPR)
            if clean.startswith("s") and clean[1:].isdigit():
                return RegisterOperand(int(clean[1:]), RegType.SGPR)
            raise ValueError(f"Invalid register operand: {token}")

        vdst = parse_reg(tokens[1])
        vaddr = parse_reg(tokens[2])

        vdata_token = tokens[3].rstrip(",")
        if vdata_token.startswith("v") or vdata_token.startswith("s"):
            vdata0: Operand = parse_reg(vdata_token)
        else:
            vdata0 = Immediate(int(vdata_token, 0))

        offset = 0
        gds = False
        for opt in tokens[4:]:
            clean = opt.rstrip(",")
            if clean.startswith("offset:"):
                offset = int(clean.split(":", 1)[1], 0)
            elif clean == "gds":
                gds = True
            else:
                raise ValueError(f"Unknown modifier: {clean}")

        return cls(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=offset, gds=gds)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarState:
    """Scalar machine state comprising register files and LDS memory."""

    def __init__(self, lds_size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)
        self.vcc: int = 0

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)

    def get_reg(self, reg: RegisterOperand) -> int:
        if reg.reg_type == RegType.VGPR:
            return self.get_vgpr(reg.index)
        return self.get_sgpr(reg.index)

    def set_reg(self, reg: RegisterOperand, val: int) -> None:
        if reg.reg_type == RegType.VGPR:
            self.set_vgpr(reg.index, val)
        else:
            self.set_sgpr(reg.index, val)

    def resolve_operand(self, op: Operand) -> int:
        if isinstance(op, Immediate):
            return op.val
        return self.get_reg(op)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_sub_rtn_f32."""

    def execute(self, instr: DSSubRtnF32Instruction, state: ScalarState) -> Tuple[int, int]:
        """Execute a single ds_sub_rtn_f32 instruction on scalar state.

        Returns:
            Tuple[int, int]: (old_val_bits, new_mem_bits)
        """
        # 1. Latch inputs prior to any register writeback
        vaddr = state.get_reg(instr.vaddr)
        sub_val = state.resolve_operand(instr.vdata0)

        # 2. Perform atomic LDS float subtraction
        effective_byte = u32(vaddr + instr.offset)
        idx = (effective_byte >> 2) & (state.lds.size_dwords - 1)
        old_val = state.lds.read_dword(idx)
        new_mem = oracle_sub_f32(old_val, sub_val)
        state.lds.write_dword(idx, new_mem)

        # 3. Write pre-subtraction value back to destination register
        state.set_reg(instr.vdst, old_val)

        return old_val, new_mem


# ==============================================================================
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ==============================================================================

class SIMDState:
    """Multi-lane SIMD machine state modeling Wave32 and Wave64 execution."""

    def __init__(
        self,
        num_lanes: int = 64,
        exec_mask: Optional[int] = None,
        lds_size_dwords: int = DEFAULT_LDS_DWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.valid_mask = (1 << num_lanes) - 1
        self.exec_mask = self.valid_mask if exec_mask is None else (exec_mask & self.valid_mask)
        self.vgprs: list[list[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)

    def set_exec_mask(self, mask: int) -> None:
        self.exec_mask = mask & self.valid_mask

    def get_lane_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index]

    def set_lane_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)

    def resolve_lane_operand(self, lane: int, op: Operand) -> int:
        if isinstance(op, Immediate):
            return op.val
        if op.reg_type == RegType.VGPR:
            return self.get_lane_vgpr(lane, op.index)
        return self.get_sgpr(op.index)


class SIMDInterpreter:
    """Multi-lane SIMD interpreter for ds_sub_rtn_f32."""

    def execute(
        self,
        instr: DSSubRtnF32Instruction,
        state: SIMDState,
    ) -> list[Optional[int]]:
        """Execute ds_sub_rtn_f32 across all lanes.

        Active lanes serialize LDS atomic updates in lane order (0..N-1).
        Inactive lanes remain completely unaffected.

        Returns:
            list[Optional[int]]: Returned old float bit values per lane (None for inactive lanes).
        """
        results: list[Optional[int]] = [None] * state.num_lanes

        # 1. Latch inputs across all lanes before writebacks
        lane_inputs: list[tuple[int, int]] = []
        for lane in range(state.num_lanes):
            vaddr = state.resolve_lane_operand(lane, instr.vaddr)
            sub_val = state.resolve_lane_operand(lane, instr.vdata0)
            lane_inputs.append((vaddr, sub_val))

        # 2. Serialize execution across active lanes
        for lane in range(state.num_lanes):
            if not ((state.exec_mask >> lane) & 1):
                continue

            vaddr, sub_val = lane_inputs[lane]
            effective_byte = u32(vaddr + instr.offset)
            idx = (effective_byte >> 2) & (state.lds.size_dwords - 1)

            old_val = state.lds.read_dword(idx)
            new_mem = oracle_sub_f32(old_val, sub_val)
            state.lds.write_dword(idx, new_mem)

            state.set_lane_vgpr(lane, instr.vdst.index, old_val)
            results[lane] = old_val

        return results


class TestLdsSubRtnF32Oracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_dwords=1024)

    def test_basic_sub_and_return(self) -> None:
        # Pre-seed memory at dword 10 with 5.0
        self.mem.write_dword(10, FIVE_F32)
        # vaddr = 40 (byte 40 -> dword 10), subtract 2.0
        ret = ref_ds_sub_rtn_f32(self.mem, vaddr=40, vdata_bits=TWO_F32, offset=0)
        # Old value (5.0) returned
        self.assertEqual(ret, FIVE_F32)
        # Memory updated to 5.0 - 2.0 = 3.0
        self.assertEqual(self.mem.read_dword(10), THREE_F32)

    def test_sub_negative_result(self) -> None:
        self.mem.write_dword(5, ONE_F32)
        ret = ref_ds_sub_rtn_f32(self.mem, vaddr=20, vdata_bits=THREE_F32, offset=0)
        self.assertEqual(ret, ONE_F32)
        # 1.0 - 3.0 = -2.0
        self.assertEqual(self.mem.read_dword(5), NEG_TWO_F32)

    def test_sub_zero_preserves_val_and_returns(self) -> None:
        self.mem.write_dword(2, TEN_F32)
        ret = ref_ds_sub_rtn_f32(self.mem, vaddr=8, vdata_bits=POS_ZERO_F32, offset=0)
        self.assertEqual(ret, TEN_F32)
        self.assertEqual(self.mem.read_dword(2), TEN_F32)

    def test_address_offset_and_wrapping(self) -> None:
        # vaddr = 16, offset = 8 -> effective byte = 24 -> dword 6
        self.mem.write_dword(6, THREE_F32)
        ret = ref_ds_sub_rtn_f32(self.mem, vaddr=16, vdata_bits=ONE_F32, offset=8)
        self.assertEqual(ret, THREE_F32)
        self.assertEqual(self.mem.read_dword(6), TWO_F32)


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)
        instr = DSSubRtnF32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=16)

        self.assertEqual(instr.opcode, DSOpcode.DS_SUB_RTN_F32)
        self.assertEqual(instr.vdst.index, 0)
        self.assertEqual(instr.vaddr.index, 1)
        self.assertEqual(instr.offset, 16)
        self.assertFalse(instr.gds)

    def test_ast_with_immediate_operand(self) -> None:
        vdst = RegisterOperand(5, RegType.VGPR)
        vaddr = RegisterOperand(6, RegType.VGPR)
        imm = Immediate(ONE_F32)
        instr = DSSubRtnF32Instruction(vdst=vdst, vaddr=vaddr, vdata0=imm, offset=0)

        self.assertEqual(instr.vdata0, Immediate(ONE_F32))

    def test_disassembly_formatting(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)

        instr_no_offset = DSSubRtnF32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=0)
        self.assertEqual(
            instr_no_offset.disassemble(),
            "ds_sub_rtn_f32 v[0], v[1], v[2]",
        )

        instr_with_offset = DSSubRtnF32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=32)
        self.assertEqual(
            instr_with_offset.disassemble(),
            "ds_sub_rtn_f32 v[0], v[1], v[2], offset:32",
        )

        instr_gds = DSSubRtnF32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=4, gds=True)
        self.assertEqual(
            instr_gds.disassemble(),
            "ds_sub_rtn_f32 v[0], v[1], v[2], offset:4, gds",
        )

    def test_ast_parse(self) -> None:
        instr = DSSubRtnF32Instruction.parse("ds_sub_rtn_f32 v[0], v[1], v[2], offset:64, gds")
        self.assertEqual(instr.vdst.index, 0)
        self.assertEqual(instr.vaddr.index, 1)
        self.assertEqual(instr.offset, 64)
        self.assertTrue(instr.gds)

    def test_ast_validation_errors(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)

        # vdst cannot be SGPR
        with self.assertRaises(ValueError):
            DSSubRtnF32Instruction(vdst=s0, vaddr=v0, vdata0=v0)

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            DSSubRtnF32Instruction(vdst=v0, vaddr=s0, vdata0=v0)

        # Negative offset
        with self.assertRaises(ValueError):
            DSSubRtnF32Instruction(vdst=v0, vaddr=v0, vdata0=v0, offset=-1)

        # Offset exceeding 16 bits
        with self.assertRaises(ValueError):
            DSSubRtnF32Instruction(vdst=v0, vaddr=v0, vdata0=v0, offset=65536)


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState(lds_size_dwords=1024)
        self.interp = ScalarInterpreter()

    def test_scalar_atomic_sub_execution(self) -> None:
        addr = 0x40
        idx = addr >> 2
        self.state.lds.write_dword(idx, TEN_F32)

        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, THREE_F32)
        self.state.set_vgpr(0, 0xCAFEBABE)  # Initial sentinel in vdst

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, self.state)

        seven_f32 = f32_to_bits(7.0)
        self.assertEqual(old_val, TEN_F32)
        self.assertEqual(new_mem, seven_f32)
        self.assertEqual(self.state.get_vgpr(0), TEN_F32)  # vdst received old_val
        self.assertEqual(self.state.lds.read_dword(idx), seven_f32)

    def test_scalar_execution_with_offset(self) -> None:
        base_addr = 0x100
        offset = 0x14
        eff_addr = base_addr + offset
        idx = eff_addr >> 2

        self.state.lds.write_dword(idx, FIVE_F32)
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr(2, TWO_F32)

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=offset,
        )

        old_val, new_mem = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, FIVE_F32)
        self.assertEqual(new_mem, THREE_F32)
        self.assertEqual(self.state.get_vgpr(0), FIVE_F32)
        self.assertEqual(self.state.lds.read_dword(idx), THREE_F32)

    def test_scalar_execution_with_immediate_operand(self) -> None:
        addr = 0x200
        idx = addr >> 2
        self.state.lds.write_dword(idx, THREE_F32)
        self.state.set_vgpr(1, addr)

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=Immediate(TWO_F32),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, THREE_F32)
        self.assertEqual(new_mem, ONE_F32)
        self.assertEqual(self.state.get_vgpr(0), THREE_F32)
        self.assertEqual(self.state.lds.read_dword(idx), ONE_F32)

    def test_scalar_unaligned_byte_address(self) -> None:
        idx = 0x300 >> 2
        self.state.lds.write_dword(idx, FIVE_F32)
        self.state.set_vgpr(1, 0x303)  # Unaligned byte address

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=Immediate(ONE_F32),
            offset=0,
        )

        old_val, new_mem = self.interp.execute(instr, self.state)
        self.assertEqual(old_val, FIVE_F32)
        four_f32 = f32_to_bits(4.0)
        self.assertEqual(new_mem, four_f32)
        self.assertEqual(self.state.lds.read_dword(idx), four_f32)

    def test_scalar_sequential_subtraction(self) -> None:
        addr = 0x50
        idx = addr >> 2
        self.state.lds.write_dword(idx, TEN_F32)
        self.state.set_vgpr(1, addr)

        steps = [ONE_F32, TWO_F32, THREE_F32]
        expected_olds = [TEN_F32, f32_to_bits(9.0), f32_to_bits(7.0)]
        expected_news = [f32_to_bits(9.0), f32_to_bits(7.0), f32_to_bits(4.0)]

        for s, exp_old, exp_new in zip(steps, expected_olds, expected_news):
            instr = DSSubRtnF32Instruction(
                vdst=RegisterOperand(0, RegType.VGPR),
                vaddr=RegisterOperand(1, RegType.VGPR),
                vdata0=Immediate(s),
            )
            old_val, new_mem = self.interp.execute(instr, self.state)
            self.assertEqual(old_val, exp_old)
            self.assertEqual(new_mem, exp_new)
            self.assertEqual(self.state.get_vgpr(0), exp_old)

        self.assertEqual(self.state.lds.read_dword(idx), f32_to_bits(4.0))


class TestSIMDMultiLane(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        for lane in range(32):
            addr = lane * 4
            idx = addr >> 2
            val = f32_to_bits(100.0 - lane)
            state.lds.write_dword(idx, val)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, ONE_F32)
            state.set_lane_vgpr(lane, 0, 0xDEADBEEF)

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            exp_old = f32_to_bits(100.0 - lane)
            exp_new = f32_to_bits(99.0 - lane)
            self.assertEqual(results[lane], exp_old)
            self.assertEqual(state.get_lane_vgpr(lane, 0), exp_old)
            self.assertEqual(state.lds.read_dword((lane * 4) >> 2), exp_new)

    def test_wave64_convergent_atomic_accumulation(self) -> None:
        # All 64 lanes atomically subtract 1.0 from the exact same address 0 (initial 100.0)
        state = SIMDState(num_lanes=64, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        state.lds.write_dword(0, f32_to_bits(100.0))
        for lane in range(64):
            state.set_lane_vgpr(lane, 1, 0)
            state.set_lane_vgpr(lane, 2, ONE_F32)
            state.set_lane_vgpr(lane, 0, 0)

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        # Final memory must be 100.0 - 64.0 = 36.0
        self.assertEqual(state.lds.read_dword(0), f32_to_bits(36.0))
        # Returned tickets must be descending sequence [100.0, 99.0, ..., 37.0]
        expected_tickets = [f32_to_bits(100.0 - i) for i in range(64)]
        self.assertEqual(results, expected_tickets)
        for lane in range(64):
            self.assertEqual(state.get_lane_vgpr(lane, 0), expected_tickets[lane])

    def test_wave32_exec_masking_partial_lanes(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x000000FF, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        sentinel = 0x12345678
        for lane in range(32):
            addr = lane * 4
            idx = addr >> 2
            state.lds.write_dword(idx, TEN_F32)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, ONE_F32)
            state.set_lane_vgpr(lane, 0, sentinel)

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        nine_f32 = f32_to_bits(9.0)
        for lane in range(32):
            idx = (lane * 4) >> 2
            if lane < 8:
                self.assertEqual(results[lane], TEN_F32)
                self.assertEqual(state.get_lane_vgpr(lane, 0), TEN_F32)
                self.assertEqual(state.lds.read_dword(idx), nine_f32)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr(lane, 0), sentinel)
                self.assertEqual(state.lds.read_dword(idx), TEN_F32)

    def test_wave64_divergent_exec_mask(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        sentinel = 0xCAFEBABE
        for lane in range(64):
            addr = lane * 4
            idx = addr >> 2
            state.lds.write_dword(idx, FIVE_F32)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, TWO_F32)
            state.set_lane_vgpr(lane, 0, sentinel)

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            idx = (lane * 4) >> 2
            if lane % 2 == 0:
                self.assertEqual(results[lane], FIVE_F32)
                self.assertEqual(state.get_lane_vgpr(lane, 0), FIVE_F32)
                self.assertEqual(state.lds.read_dword(idx), THREE_F32)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr(lane, 0), sentinel)
                self.assertEqual(state.lds.read_dword(idx), FIVE_F32)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        addr = 0x100
        idx = addr >> 2
        state.lds.write_dword(idx, TEN_F32)
        for lane in range(32):
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, ONE_F32)
            state.set_lane_vgpr(lane, 0, 777)

        instr = DSSubRtnF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_lane_vgpr(lane, 0), 777)
        self.assertEqual(state.lds.read_dword(idx), TEN_F32)


if __name__ == "__main__":
    unittest.main()

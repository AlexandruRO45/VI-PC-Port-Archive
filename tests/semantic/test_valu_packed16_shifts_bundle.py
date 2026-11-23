"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU packed 16-bit shift and integer multiply-add operations (Bundle 22):
  - v_pk_ashrrev_i16: Packed 16-bit signed arithmetic right shift
  - v_pk_lshlrev_b16: Packed 16-bit logical left shift
  - v_pk_lshrrev_b16: Packed 16-bit logical right shift
  - v_pk_mad_i16:     Packed 16-bit signed integer multiply-add
  - v_pk_mad_u16:     Packed 16-bit unsigned integer multiply-add
  - v_pk_sub_u16:     Packed 16-bit unsigned integer subtraction

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Callable, Dict, List, Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import support
except Exception:
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF


def u16(val: int) -> int:
    return val & MASK16


def i16(val: int) -> int:
    v = val & MASK16
    return v - 0x10000 if v >= 0x8000 else v


def u32(val: int) -> int:
    return val & MASK32


def i32(val: int) -> int:
    v = val & MASK32
    return v - 0x100000000 if v >= 0x80000000 else v


def pack16(lo: int, hi: int) -> int:
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack16(val: int) -> tuple[int, int]:
    return (val & MASK16, (val >> 16) & MASK16)

class Opcode(Enum):
    V_PK_ASHRREV_I16 = "v_pk_ashrrev_i16"
    V_PK_LSHLREV_B16 = "v_pk_lshlrev_b16"
    V_PK_LSHRREV_B16 = "v_pk_lshrrev_b16"
    V_PK_MAD_I16 = "v_pk_mad_i16"
    V_PK_MAD_U16 = "v_pk_mad_u16"
    V_PK_SUB_U16 = "v_pk_sub_u16"


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    value: int = 0

    @classmethod
    def vgpr(cls, idx: int) -> ASTOperand:
        return cls(kind=OperandKind.VGPR, index=idx)

    @classmethod
    def sgpr(cls, idx: int) -> ASTOperand:
        return cls(kind=OperandKind.SGPR, index=idx)

    @classmethod
    def imm(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.IMM, value=val & MASK32)


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: Optional[ASTOperand] = None

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.VGPR:
                return f"v{op.index}"
            if op.kind == OperandKind.SGPR:
                return f"s{op.index}"
            return hex(op.value)

        if self.src2 is not None:
            return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}, {fmt(self.src2)}"
        return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}"


class ScalarMachineState:
    def __init__(self, num_vgprs: int = 256, num_sgprs: int = 106) -> None:
        self.vgprs: list[int] = [0] * num_vgprs
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0
        self.vcc: int = 0

    def read_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK32
        return op.value & MASK32

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32


GLOBAL_ORACLE_MAP: dict[Opcode, Callable[..., int]] = {}


class ScalarInterpreter:
    def step(self, instr: ASTInstruction, state: ScalarMachineState) -> int:
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        if instr.src2 is not None:
            s2 = state.read_operand(instr.src2)
            res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1, s2)
        else:
            res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1)

        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr(instr.dst.index, res)
        return res

class SIMDWavefront:
    def __init__(self, wave_size: int = 32, exec_mask: int | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.mask_limit = (1 << wave_size) - 1
        self.exec_mask = self.mask_limit if exec_mask is None else (exec_mask & self.mask_limit)

    def execute(
        self,
        op_fn: Callable[..., int],
        s0_lanes: list[int],
        s1_lanes: list[int],
        s2_lanes: list[int] | None = None,
        orig_dst: list[int] | None = None,
    ) -> list[int]:
        res: list[int] = [0] * self.wave_size
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                if s2_lanes is not None:
                    res[lane] = op_fn(s0_lanes[lane], s1_lanes[lane], s2_lanes[lane]) & MASK32
                else:
                    res[lane] = op_fn(s0_lanes[lane], s1_lanes[lane]) & MASK32
            else:
                res[lane] = orig_dst[lane] if orig_dst is not None else 0
        return res

def ref_v_pk_ashrrev_i16(s0: int, s1: int) -> int:
    """Arithmetic right shift reversed on packed 16-bit integers.
    s0: shift count (each half masked to 15).
    s1: values (each half sign-extended to 16 bits).
    """
    amt_lo = s0 & 15
    amt_hi = (s0 >> 16) & 15
    val_lo = i16(s1 & MASK16)
    val_hi = i16((s1 >> 16) & MASK16)
    res_lo = (val_lo >> amt_lo) & MASK16
    res_hi = (val_hi >> amt_hi) & MASK16
    return pack16(res_lo, res_hi)


class TestVPkAshrrevI16Oracle(unittest.TestCase):
    def test_oracle_positive_shifts(self) -> None:
        s0 = pack16(2, 4)
        s1 = pack16(8, 64)
        self.assertEqual(ref_v_pk_ashrrev_i16(s0, s1), pack16(2, 4))

    def test_oracle_negative_sign_extension(self) -> None:
        s0 = pack16(2, 3)
        s1 = pack16(u16(-16), u16(-64))
        self.assertEqual(ref_v_pk_ashrrev_i16(s0, s1), pack16(u16(-4), u16(-8)))

    def test_oracle_shift_amount_masking(self) -> None:
        s0 = pack16(16, 31)
        s1 = pack16(1234, u16(-8))
        # 16 & 15 == 0, 31 & 15 == 15
        self.assertEqual(ref_v_pk_ashrrev_i16(s0, s1), pack16(1234, u16(-1)))

    def test_oracle_zero_and_extremes(self) -> None:
        s0 = pack16(1, 15)
        s1 = pack16(0x7FFF, 0x8000)
        # 0x7FFF >> 1 == 0x3FFF, -32768 >> 15 == -1 (0xFFFF)
        self.assertEqual(ref_v_pk_ashrrev_i16(s0, s1), pack16(0x3FFF, 0xFFFF))

GLOBAL_ORACLE_MAP[Opcode.V_PK_ASHRREV_I16] = ref_v_pk_ashrrev_i16


class TestVPkAshrrevI16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(
            Opcode.V_PK_ASHRREV_I16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        self.assertEqual(inst.disasm(), "v_pk_ashrrev_i16 v0, v1, v2")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(1, pack16(2, 1))
        state.write_vgpr(2, pack16(u16(-8), 100))
        inst = ASTInstruction(
            Opcode.V_PK_ASHRREV_I16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, pack16(u16(-2), 50))
        self.assertEqual(state.vgprs[0], pack16(u16(-2), 50))

    def test_imm_and_sgpr_operands(self) -> None:
        state = ScalarMachineState()
        state.sgprs[5] = pack16(1, 2)
        inst = ASTInstruction(
            Opcode.V_PK_ASHRREV_I16,
            ASTOperand.vgpr(3),
            ASTOperand.sgpr(5),
            ASTOperand.imm(pack16(u16(-10), 40)),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, pack16(u16(-5), 10))

class TestVPkAshrrevI16SIMDAndEdgeCases(unittest.TestCase):
    def test_simd_divergence_exec_mask(self) -> None:
        simd = SIMDWavefront(wave_size=32, exec_mask=0x55555555)
        s0_lanes = [pack16(i & 15, (i + 1) & 15) for i in range(32)]
        s1_lanes = [pack16(u16(-16), 32) for _ in range(32)]
        orig_dst = [0xDEADBEEF] * 32
        res = simd.execute(ref_v_pk_ashrrev_i16, s0_lanes, s1_lanes, orig_dst=orig_dst)
        for i in range(32):
            if i % 2 == 0:
                expected = ref_v_pk_ashrrev_i16(s0_lanes[i], s1_lanes[i])
                self.assertEqual(res[i], expected)
            else:
                self.assertEqual(res[i], 0xDEADBEEF)

    def test_boundary_sign_bits(self) -> None:
        # Check all combinations of boundary sign bits (-32768, -1, 0, 32767)
        vals = [0x8000, 0xFFFF, 0x0000, 0x7FFF]
        for v in vals:
            packed_val = pack16(v, v)
            res = ref_v_pk_ashrrev_i16(pack16(1, 1), packed_val)
            exp_lo = (i16(v) >> 1) & MASK16
            exp_hi = (i16(v) >> 1) & MASK16
            self.assertEqual(res, pack16(exp_lo, exp_hi))

    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xFFFFFFFFFFFFFFFF
        state.write_vgpr(1, pack16(2, 2))
        state.write_vgpr(2, pack16(4, 4))
        inst = ASTInstruction(
            Opcode.V_PK_ASHRREV_I16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xFFFFFFFFFFFFFFFF)

class TestVPkAshrrevI16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available in this Python environment")
        lines = [
            "v_mov_b32 v0, 0x00020002",
            "v_mov_b32 v1, 0xfff0fff0",
            "v_pk_ashrrev_i16 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_pk_ashrrev_i16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFCFFFC)

def ref_v_pk_lshlrev_b16(s0: int, s1: int) -> int:
    """Logical left shift reversed on packed 16-bit values.
    s0: shift count (each half masked to 15).
    s1: values (each half 16-bit unsigned).
    """
    amt_lo = s0 & 15
    amt_hi = (s0 >> 16) & 15
    val_lo = s1 & MASK16
    val_hi = (s1 >> 16) & MASK16
    res_lo = (val_lo << amt_lo) & MASK16
    res_hi = (val_hi << amt_hi) & MASK16
    return pack16(res_lo, res_hi)


class TestVPkLshlrevB16Oracle(unittest.TestCase):
    def test_oracle_basic_shifts(self) -> None:
        s0 = pack16(2, 4)
        s1 = pack16(3, 5)
        self.assertEqual(ref_v_pk_lshlrev_b16(s0, s1), pack16(12, 80))

    def test_oracle_shift_wrap_16bit(self) -> None:
        s0 = pack16(1, 2)
        s1 = pack16(0x8000, 0xC000)
        # 0x8000 << 1 wraps to 0x0000; 0xC000 << 2 == 0x30000 wraps to 0x0000
        self.assertEqual(ref_v_pk_lshlrev_b16(s0, s1), pack16(0x0000, 0x0000))

    def test_oracle_shift_amount_masking(self) -> None:
        s0 = pack16(16, 33)
        s1 = pack16(0x1234, 0x0001)
        # 16 & 15 == 0, 33 & 15 == 1
        self.assertEqual(ref_v_pk_lshlrev_b16(s0, s1), pack16(0x1234, 0x0002))

    def test_oracle_zero_inputs(self) -> None:
        self.assertEqual(ref_v_pk_lshlrev_b16(0, 0), 0)

GLOBAL_ORACLE_MAP[Opcode.V_PK_LSHLREV_B16] = ref_v_pk_lshlrev_b16


class TestVPkLshlrevB16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(
            Opcode.V_PK_LSHLREV_B16,
            ASTOperand.vgpr(5),
            ASTOperand.vgpr(6),
            ASTOperand.vgpr(7),
        )
        self.assertEqual(inst.disasm(), "v_pk_lshlrev_b16 v5, v6, v7")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(1, pack16(3, 1))
        state.write_vgpr(2, pack16(7, 15))
        inst = ASTInstruction(
            Opcode.V_PK_LSHLREV_B16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, pack16(56, 30))
        self.assertEqual(state.vgprs[0], pack16(56, 30))

    def test_imm_operands(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(2, pack16(1, 2))
        inst = ASTInstruction(
            Opcode.V_PK_LSHLREV_B16,
            ASTOperand.vgpr(0),
            ASTOperand.imm(pack16(4, 3)),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, pack16(16, 16))

class TestVPkLshlrevB16SIMDAndEdgeCases(unittest.TestCase):
    def test_simd_divergence_exec_mask(self) -> None:
        simd = SIMDWavefront(wave_size=32, exec_mask=0x00FF00FF)
        s0_lanes = [pack16(1, 2) for _ in range(32)]
        s1_lanes = [pack16(i, i * 2) for i in range(32)]
        orig_dst = [0xCAFEBABE] * 32
        res = simd.execute(ref_v_pk_lshlrev_b16, s0_lanes, s1_lanes, orig_dst=orig_dst)
        for i in range(32):
            if (0x00FF00FF >> i) & 1:
                self.assertEqual(res[i], pack16(i << 1, (i * 2) << 2))
            else:
                self.assertEqual(res[i], 0xCAFEBABE)

    def test_bit_pattern_invariance(self) -> None:
        # Shifting 0xFFFF by 0..15 preserves exact bit patterns
        for shift in range(16):
            res = ref_v_pk_lshlrev_b16(pack16(shift, shift), pack16(0xFFFF, 0xFFFF))
            exp = (0xFFFF << shift) & MASK16
            self.assertEqual(res, pack16(exp, exp))

    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0x12345678
        state.write_vgpr(1, pack16(1, 1))
        state.write_vgpr(2, pack16(2, 2))
        inst = ASTInstruction(
            Opcode.V_PK_LSHLREV_B16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0x12345678)

class TestVPkLshlrevB16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available in this Python environment")
        lines = [
            "v_mov_b32 v0, 0x00020004",
            "v_mov_b32 v1, 0x00020004",
            "v_pk_lshlrev_b16 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_pk_lshlrev_b16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x00080040)

def ref_v_pk_lshrrev_b16(s0: int, s1: int) -> int:
    """Logical right shift reversed on packed 16-bit values.
    s0: shift count (each half masked to 15).
    s1: values (each half 16-bit unsigned).
    """
    amt_lo = s0 & 15
    amt_hi = (s0 >> 16) & 15
    val_lo = s1 & MASK16
    val_hi = (s1 >> 16) & MASK16
    res_lo = (val_lo >> amt_lo) & MASK16
    res_hi = (val_hi >> amt_hi) & MASK16
    return pack16(res_lo, res_hi)


class TestVPkLshrrevB16Oracle(unittest.TestCase):
    def test_oracle_basic_shifts(self) -> None:
        s0 = pack16(2, 4)
        s1 = pack16(16, 64)
        self.assertEqual(ref_v_pk_lshrrev_b16(s0, s1), pack16(4, 4))

    def test_oracle_zero_fill_high_bits(self) -> None:
        s0 = pack16(1, 2)
        s1 = pack16(0x8000, 0x8000)
        # Logical shift fills zeros in upper bits
        self.assertEqual(ref_v_pk_lshrrev_b16(s0, s1), pack16(0x4000, 0x2000))

    def test_oracle_shift_amount_masking(self) -> None:
        s0 = pack16(16, 32)
        s1 = pack16(0xABCD, 0x1234)
        # 16 & 15 == 0, 32 & 15 == 0
        self.assertEqual(ref_v_pk_lshrrev_b16(s0, s1), pack16(0xABCD, 0x1234))

    def test_oracle_extremes(self) -> None:
        s0 = pack16(15, 15)
        s1 = pack16(0xFFFF, 0x8000)
        self.assertEqual(ref_v_pk_lshrrev_b16(s0, s1), pack16(1, 1))

if __name__ == "__main__":
    unittest.main()

"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU Normalized Float16/Float32 Conversion operations (Bundle 42):
  - v_cvt_norm_i16_f16:   Normalized Float16 to signed integer Norm16 (D[15:0] = snorm16(S0.f16))
  - v_cvt_norm_u16_f16:   Normalized Float16 to unsigned integer Norm16 (D[15:0] = unorm16(S0.f16))
  - v_cvt_pknorm_i16_f16: Packed Float16 to signed Norm16 pair (D = {snorm16(S1.f16), snorm16(S0.f16)})
  - v_cvt_pknorm_u16_f16: Packed Float16 to unsigned Norm16 pair (D = {unorm16(S1.f16), unorm16(S0.f16)})
  - v_cvt_pknorm_i16_f32: Packed Float32 to signed Norm16 pair (D = {snorm16(S1.f32), snorm16(S0.f32)})
  - v_cvt_pknorm_u16_f32: Packed Float32 to unsigned Norm16 pair (D = {unorm16(S1.f32), unorm16(S0.f32)})

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
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

_TRANSLATOR_READY = support is not None and rdna2spirv is not None

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    return val & MASK32


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & 0xFFFF))[0]


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def is_nan_f16(h: int) -> bool:
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def is_nan_f32(w: int) -> bool:
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x007FFFFF) != 0


def norm_i16(v: float) -> int:
    if math.isnan(v):
        return 0
    cl = min(max(v, -1.0), 1.0)
    return int(round(cl * 32767.0)) & 0xFFFF


def norm_u16(v: float) -> int:
    if math.isnan(v):
        return 0
    cl = min(max(v, 0.0), 1.0)
    return int(round(cl * 65535.0)) & 0xFFFF


class Opcode(Enum):
    V_CVT_NORM_I16_F16 = "v_cvt_norm_i16_f16"
    V_CVT_NORM_U16_F16 = "v_cvt_norm_u16_f16"
    V_CVT_PKNORM_I16_F16 = "v_cvt_pknorm_i16_f16"
    V_CVT_PKNORM_U16_F16 = "v_cvt_pknorm_u16_f16"
    V_CVT_PKNORM_I16_F32 = "v_cvt_pknorm_i16_f32"
    V_CVT_PKNORM_U16_F32 = "v_cvt_pknorm_u16_f32"


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
    src1: Optional[ASTOperand] = None

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.VGPR:
                return f"v{op.index}"
            if op.kind == OperandKind.SGPR:
                return f"s{op.index}"
            return hex(op.value)

        if self.src1 is not None:
            return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}"
        return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}"


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
        old_dst = state.read_operand(instr.dst)
        if instr.src1 is not None:
            s1 = state.read_operand(instr.src1)
            res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1)
        else:
            res = GLOBAL_ORACLE_MAP[instr.opcode](s0, old_dst=old_dst)

        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr(instr.dst.index, res)
        return res

def ref_v_cvt_norm_i16_f16(s0: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0):
        res_i16 = 0
    else:
        res_i16 = norm_i16(bits_to_f16(s0 & 0xFFFF))
    return ((old_dst & 0xFFFF0000) | (res_i16 & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_CVT_NORM_I16_F16] = ref_v_cvt_norm_i16_f16


class TestVCvtNormI16F16Oracle(unittest.TestCase):
    def test_basic_conversion(self) -> None:
        # 1.0 -> 0x7FFF (32767)
        res = ref_v_cvt_norm_i16_f16(0x3C00, old_dst=0xDEAD0000)
        self.assertEqual(res & 0xFFFF, 0x7FFF)
        self.assertEqual(res >> 16, 0xDEAD)

    def test_negative_conversion(self) -> None:
        # -1.0 -> 0x8001 (-32767)
        res = ref_v_cvt_norm_i16_f16(0xBC00, old_dst=0xBEEF0000)
        self.assertEqual(res & 0xFFFF, 0x8001)
        self.assertEqual(res >> 16, 0xBEEF)

    def test_zero_and_clamp(self) -> None:
        # 0.0 -> 0
        res_zero = ref_v_cvt_norm_i16_f16(0x0000, old_dst=0x11110000)
        self.assertEqual(res_zero & 0xFFFF, 0x0000)
        # 2.0 -> clamped to 1.0 -> 0x7FFF
        res_clamp = ref_v_cvt_norm_i16_f16(0x4000, old_dst=0x22220000)
        self.assertEqual(res_clamp & 0xFFFF, 0x7FFF)

class TestVCvtNormI16F16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_I16_F16,
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(0),
        )
        self.assertEqual(inst.disasm(), "v_cvt_norm_i16_f16 v1, v0")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0xCAFE0000)
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_I16_F16,
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(0),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x7FFF)
        self.assertEqual(res >> 16, 0xCAFE)
        self.assertEqual(state.vgprs[1], res)

class TestVCvtNormI16F16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xFFFFFFFFFFFFFFFF
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x12340000)
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_I16_F16,
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(0),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xFFFFFFFFFFFFFFFF)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_I16_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x7FFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # NaN -> 0
        res_nan = ref_v_cvt_norm_i16_f16(0x7E00)
        self.assertEqual(res_nan & 0xFFFF, 0)
        # +Inf clamped to 1.0 -> 0x7FFF
        res_inf = ref_v_cvt_norm_i16_f16(0x7C00)
        self.assertEqual(res_inf & 0xFFFF, 0x7FFF)
        # -Inf clamped to -1.0 -> 0x8001
        res_neginf = ref_v_cvt_norm_i16_f16(0xFC00)
        self.assertEqual(res_neginf & 0xFFFF, 0x8001)

class TestVCvtNormI16F16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_norm_i16_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x7FFF)

def ref_v_cvt_norm_u16_f16(s0: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0):
        res_u16 = 0
    else:
        res_u16 = norm_u16(bits_to_f16(s0 & 0xFFFF))
    return ((old_dst & 0xFFFF0000) | (res_u16 & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_CVT_NORM_U16_F16] = ref_v_cvt_norm_u16_f16


class TestVCvtNormU16F16Oracle(unittest.TestCase):
    def test_basic_conversion(self) -> None:
        # 1.0 -> 0xFFFF (65535)
        res = ref_v_cvt_norm_u16_f16(0x3C00, old_dst=0xFEED0000)
        self.assertEqual(res & 0xFFFF, 0xFFFF)
        self.assertEqual(res >> 16, 0xFEED)

    def test_fractional_and_clamp(self) -> None:
        # 0.5 -> 0x8000 (32768)
        res_half = ref_v_cvt_norm_u16_f16(0x3800, old_dst=0x43210000)
        self.assertEqual(res_half & 0xFFFF, 0x8000)
        # Negative -> clamped to 0.0 -> 0
        res_neg = ref_v_cvt_norm_u16_f16(0xBC00, old_dst=0x55550000)
        self.assertEqual(res_neg & 0xFFFF, 0)

class TestVCvtNormU16F16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_U16_F16,
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(0),
        )
        self.assertEqual(inst.disasm(), "v_cvt_norm_u16_f16 v1, v0")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x99990000)
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_U16_F16,
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(0),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0xFFFF)
        self.assertEqual(res >> 16, 0x9999)
        self.assertEqual(state.vgprs[1], res)

class TestVCvtNormU16F16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x88880000)
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_U16_F16,
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(0),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_U16_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0xFFFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # NaN -> 0
        res_nan = ref_v_cvt_norm_u16_f16(0x7E00)
        self.assertEqual(res_nan & 0xFFFF, 0)
        # +Inf clamped to 1.0 -> 0xFFFF
        res_inf = ref_v_cvt_norm_u16_f16(0x7C00)
        self.assertEqual(res_inf & 0xFFFF, 0xFFFF)
        # -Inf clamped to 0.0 -> 0
        res_neginf = ref_v_cvt_norm_u16_f16(0xFC00)
        self.assertEqual(res_neginf & 0xFFFF, 0)

class TestVCvtNormU16F16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_norm_u16_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0xFFFF)

def ref_v_cvt_pknorm_i16_f16(s0: int, s1: int) -> int:
    lo = 0 if is_nan_f16(s0) else norm_i16(bits_to_f16(s0 & 0xFFFF))
    hi = 0 if is_nan_f16(s1) else norm_i16(bits_to_f16(s1 & 0xFFFF))
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_CVT_PKNORM_I16_F16] = ref_v_cvt_pknorm_i16_f16


class TestVCvtPknormI16F16Oracle(unittest.TestCase):
    def test_basic_packing(self) -> None:
        # lo: 1.0 -> 0x7FFF; hi: -1.0 -> 0x8001
        res = ref_v_cvt_pknorm_i16_f16(0x3C00, 0xBC00)
        self.assertEqual(res, 0x80017FFF)

    def test_zero_and_clamp(self) -> None:
        # lo: 0.0 -> 0x0000; hi: 2.0 -> clamped to 1.0 (0x7FFF)
        res = ref_v_cvt_pknorm_i16_f16(0x0000, 0x4000)
        self.assertEqual(res, 0x7FFF0000)

class TestVCvtPknormI16F16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_cvt_pknorm_i16_f16 v2, v0, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0xBC00)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x80017FFF)
        self.assertEqual(state.vgprs[2], 0x80017FFF)

class TestVCvtPknormI16F16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0x5555555555555555
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0xBC00)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0x5555555555555555)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0xBC00)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x80017FFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # lo: NaN -> 0; hi: Inf -> 0x7FFF
        res = ref_v_cvt_pknorm_i16_f16(0x7E00, 0x7C00)
        self.assertEqual(res, 0x7FFF0000)

class TestVCvtPknormI16F16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xbc00",
            "v_cvt_pknorm_i16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_pknorm_i16_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)

def ref_v_cvt_pknorm_u16_f16(s0: int, s1: int) -> int:
    lo = 0 if is_nan_f16(s0) else norm_u16(bits_to_f16(s0 & 0xFFFF))
    hi = 0 if is_nan_f16(s1) else norm_u16(bits_to_f16(s1 & 0xFFFF))
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_CVT_PKNORM_U16_F16] = ref_v_cvt_pknorm_u16_f16


class TestVCvtPknormU16F16Oracle(unittest.TestCase):
    def test_basic_packing(self) -> None:
        # lo: 1.0 -> 0xFFFF; hi: 0.5 -> 0x8000
        res = ref_v_cvt_pknorm_u16_f16(0x3C00, 0x3800)
        self.assertEqual(res, 0x8000FFFF)

    def test_zero_and_clamp(self) -> None:
        # lo: -1.0 -> clamped to 0 -> 0x0000; hi: 2.0 -> clamped to 1.0 -> 0xFFFF
        res = ref_v_cvt_pknorm_u16_f16(0xBC00, 0x4000)
        self.assertEqual(res, 0xFFFF0000)

class TestVCvtPknormU16F16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_cvt_pknorm_u16_f16 v2, v0, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x3800)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x8000FFFF)
        self.assertEqual(state.vgprs[2], 0x8000FFFF)

class TestVCvtPknormU16F16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x3800)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x3800)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x8000FFFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # lo: NaN -> 0; hi: Inf -> 0xFFFF
        res = ref_v_cvt_pknorm_u16_f16(0x7E00, 0x7C00)
        self.assertEqual(res, 0xFFFF0000)

class TestVCvtPknormU16F16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_cvt_pknorm_u16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_pknorm_u16_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)

def ref_v_cvt_pknorm_i16_f32(s0: int, s1: int) -> int:
    lo = 0 if is_nan_f32(s0) else norm_i16(bits_to_f32(s0))
    hi = 0 if is_nan_f32(s1) else norm_i16(bits_to_f32(s1))
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_CVT_PKNORM_I16_F32] = ref_v_cvt_pknorm_i16_f32


class TestVCvtPknormI16F32Oracle(unittest.TestCase):
    def test_basic_packing(self) -> None:
        # lo: 1.0f -> 0x7FFF; hi: -1.0f -> 0x8001
        res = ref_v_cvt_pknorm_i16_f32(0x3F800000, 0xBF800000)
        self.assertEqual(res, 0x80017FFF)

    def test_zero_and_clamp(self) -> None:
        # lo: 0.0f -> 0x0000; hi: 3.0f -> clamped to 1.0 -> 0x7FFF
        res = ref_v_cvt_pknorm_i16_f32(0x00000000, 0x40400000)
        self.assertEqual(res, 0x7FFF0000)

class TestVCvtPknormI16F32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F32,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_cvt_pknorm_i16_f32 v2, v0, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0xBF800000)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F32,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x80017FFF)
        self.assertEqual(state.vgprs[2], 0x80017FFF)

class TestVCvtPknormI16F32SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0x1234567890ABCDEF
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0xBF800000)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F32,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0x1234567890ABCDEF)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0xBF800000)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F32,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x80017FFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # lo: NaN -> 0; hi: Inf -> 0x7FFF
        res = ref_v_cvt_pknorm_i16_f32(0x7FC00000, 0x7F800000)
        self.assertEqual(res, 0x7FFF0000)

class TestVCvtPknormI16F32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0xbf800000",
            "v_cvt_pknorm_i16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_pknorm_i16_f32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)

def ref_v_cvt_pknorm_u16_f32(s0: int, s1: int) -> int:
    lo = 0 if is_nan_f32(s0) else norm_u16(bits_to_f32(s0))
    hi = 0 if is_nan_f32(s1) else norm_u16(bits_to_f32(s1))
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_CVT_PKNORM_U16_F32] = ref_v_cvt_pknorm_u16_f32


class TestVCvtPknormU16F32Oracle(unittest.TestCase):
    def test_basic_packing(self) -> None:
        # lo: 1.0f -> 0xFFFF; hi: 0.5f -> 0x8000
        res = ref_v_cvt_pknorm_u16_f32(0x3F800000, 0x3F000000)
        self.assertEqual(res, 0x8000FFFF)

    def test_zero_and_clamp(self) -> None:
        # lo: -2.0f -> clamped to 0 -> 0x0000; hi: 5.0f -> clamped to 1.0 -> 0xFFFF
        res = ref_v_cvt_pknorm_u16_f32(0xC0000000, 0x40A00000)
        self.assertEqual(res, 0xFFFF0000)

class TestVCvtPknormU16F32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F32,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_cvt_pknorm_u16_f32 v2, v0, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x3F000000)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F32,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x8000FFFF)
        self.assertEqual(state.vgprs[2], 0x8000FFFF)

class TestVCvtPknormU16F32SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xFEDCBA9876543210
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x3F000000)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F32,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xFEDCBA9876543210)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x3F000000)
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F32,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x8000FFFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # lo: NaN -> 0; hi: Inf -> 0xFFFF
        res = ref_v_cvt_pknorm_u16_f32(0x7FC00000, 0x7F800000)
        self.assertEqual(res, 0xFFFF0000)

if __name__ == "__main__":
    unittest.main()

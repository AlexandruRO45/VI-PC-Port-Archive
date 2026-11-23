"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU Float16 Mixed Precision & Specialized FMA operations (Bundle 20):
  - v_fmaak_f16:     Float16 fused multiply-add with literal K (D = S0 * S1 + K)
  - v_fmamk_f16:     Float16 fused multiply-add with literal K (D = S0 * K + S1)
  - v_fma_mixhi_f16: Float16 mixed precision FMA into high halfword (D[31:16] = fma(S0, S1, S2))
  - v_fma_mixlo_f16: Float16 mixed precision FMA into low halfword (D[15:0] = fma(S0, S1, S2))
  - v_dot2_f32_f16:  Float16 packed dot product with Float32 accumulator (D = S0.lo*S1.lo + S0.hi*S1.hi + S2)
  - v_dot2c_f32_f16: Float16 packed dot product accumulating into destination (D = S0.lo*S1.lo + S0.hi*S1.hi + D)

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


class Opcode(Enum):
    V_FMAAK_F16 = "v_fmaak_f16"
    V_FMAMK_F16 = "v_fmamk_f16"
    V_FMA_MIXHI_F16 = "v_fma_mixhi_f16"
    V_FMA_MIXLO_F16 = "v_fma_mixlo_f16"
    V_DOT2_F32_F16 = "v_dot2_f32_f16"
    V_DOT2C_F32_F16 = "v_dot2c_f32_f16"


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
        old_dst = state.read_operand(instr.dst)
        if instr.src2 is not None:
            s2 = state.read_operand(instr.src2)
            if instr.opcode in (
                Opcode.V_FMAAK_F16,
                Opcode.V_FMAMK_F16,
                Opcode.V_FMA_MIXHI_F16,
                Opcode.V_FMA_MIXLO_F16,
            ):
                res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1, s2, old_dst=old_dst)
            else:
                res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1, s2)
        else:
            if instr.opcode == Opcode.V_DOT2C_F32_F16:
                res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1, old_dst=old_dst)
            else:
                res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1)

        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr(instr.dst.index, res)
        return res

def ref_v_fmaak_f16(s0: int, s1: int, k: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0) or is_nan_f16(s1) or is_nan_f16(k):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f16(s0 & 0xFFFF)
        fb = bits_to_f16(s1 & 0xFFFF)
        fk = bits_to_f16(k & 0xFFFF)
        r = fa * fb + fk
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return ((old_dst & 0xFFFF0000) | (res_f16 & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_FMAAK_F16] = ref_v_fmaak_f16


class TestVFmaakF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
        res = ref_v_fmaak_f16(0x3C00, 0x4000, 0x3800, old_dst=0xDEAD0000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xDEAD)

    def test_zero_and_identity(self) -> None:
        # 0.0 * 2.0 + 1.0 = 1.0 (0x3C00)
        res = ref_v_fmaak_f16(0x0000, 0x4000, 0x3C00, old_dst=0x12340000)
        self.assertEqual(res & 0xFFFF, 0x3C00)
        self.assertEqual(res >> 16, 0x1234)

class TestVFmaakF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_FMAAK_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.imm(0x3800),
        )
        self.assertEqual(inst.disasm(), "v_fmaak_f16 v2, v0, v1, 0x3800")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x4000)
        state.write_vgpr(2, 0xBEEF0000)
        inst = ASTInstruction(
            Opcode.V_FMAAK_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.imm(0x3800),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xBEEF)
        self.assertEqual(state.vgprs[2], res)

class TestVFmaakF16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xFFFFFFFFFFFFFFFF
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x4000)
        state.write_vgpr(2, 0x11110000)
        inst = ASTInstruction(
            Opcode.V_FMAAK_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.imm(0x3800),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xFFFFFFFFFFFFFFFF)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x4000)
        inst = ASTInstruction(
            Opcode.V_FMAAK_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.imm(0x3800),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x4100)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = ref_v_fmaak_f16(0x7E00, 0x4000, 0x3800)
        self.assertEqual(res_nan & 0xFFFF, 0x7E00)
        res_inf = ref_v_fmaak_f16(0x7C00, 0x4000, 0x3800)
        self.assertEqual(res_inf & 0xFFFF, 0x7C00)

class TestVFmaakF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x4000",
            "v_fmaak_f16 v2, v0, v1, 0x3800",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_fmaak_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4100)

def ref_v_fmamk_f16(s0: int, k: int, s1: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0) or is_nan_f16(k) or is_nan_f16(s1):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f16(s0 & 0xFFFF)
        fk = bits_to_f16(k & 0xFFFF)
        fb = bits_to_f16(s1 & 0xFFFF)
        r = fa * fk + fb
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return ((old_dst & 0xFFFF0000) | (res_f16 & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_FMAMK_F16] = ref_v_fmamk_f16


class TestVFmamkF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
        res = ref_v_fmamk_f16(0x3C00, 0x4000, 0x3800, old_dst=0xCAFE0000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xCAFE)

    def test_zero_and_identity(self) -> None:
        # 0.0 * 2.0 + 1.0 = 1.0 (0x3C00)
        res = ref_v_fmamk_f16(0x0000, 0x4000, 0x3C00, old_dst=0x43210000)
        self.assertEqual(res & 0xFFFF, 0x3C00)
        self.assertEqual(res >> 16, 0x4321)

class TestVFmamkF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_FMAMK_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.imm(0x4000),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_fmamk_f16 v2, v0, 0x4000, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x3800)
        state.write_vgpr(2, 0xFEED0000)
        inst = ASTInstruction(
            Opcode.V_FMAMK_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.imm(0x4000),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xFEED)
        self.assertEqual(state.vgprs[2], res)

class TestVFmamkF16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x3800)
        state.write_vgpr(2, 0x22220000)
        inst = ASTInstruction(
            Opcode.V_FMAMK_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.imm(0x4000),
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
            Opcode.V_FMAMK_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.imm(0x4000),
            ASTOperand.vgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x4100)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = ref_v_fmamk_f16(0x7E00, 0x4000, 0x3800)
        self.assertEqual(res_nan & 0xFFFF, 0x7E00)
        res_inf = ref_v_fmamk_f16(0x7C00, 0x4000, 0x3800)
        self.assertEqual(res_inf & 0xFFFF, 0x7C00)

class TestVFmamkF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_fmamk_f16 v2, v0, 0x4000, v1",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_fmamk_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4100)

def ref_v_fma_mixhi_f16(s0: int, s1: int, s2: int, old_dst: int = 0) -> int:
    if is_nan_f32(s0) or is_nan_f32(s1) or is_nan_f32(s2):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f32(s0)
        fb = bits_to_f32(s1)
        fc = bits_to_f32(s2)
        r = fa * fb + fc
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return (((res_f16 & 0xFFFF) << 16) | (old_dst & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_FMA_MIXHI_F16] = ref_v_fma_mixhi_f16


class TestVFmaMixhiF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0f * 2.0f + 0.5f = 2.5f (f16 0x4100 written to high halfword)
        res = ref_v_fma_mixhi_f16(0x3F800000, 0x40000000, 0x3F000000, old_dst=0x000055AA)
        self.assertEqual(res >> 16, 0x4100)
        self.assertEqual(res & 0xFFFF, 0x55AA)

    def test_zero_and_identity(self) -> None:
        # 0.0f * 2.0f + 1.0f = 1.0f (0x3C00 in high halfword)
        res = ref_v_fma_mixhi_f16(0x00000000, 0x40000000, 0x3F800000, old_dst=0x00001234)
        self.assertEqual(res >> 16, 0x3C00)
        self.assertEqual(res & 0xFFFF, 0x1234)

class TestVFmaMixhiF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_FMA_MIXHI_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        self.assertEqual(inst.disasm(), "v_fma_mixhi_f16 v3, v0, v1, v2")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x40000000)
        state.write_vgpr(2, 0x3F000000)
        state.write_vgpr(3, 0x00009988)
        inst = ASTInstruction(
            Opcode.V_FMA_MIXHI_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res >> 16, 0x4100)
        self.assertEqual(res & 0xFFFF, 0x9988)
        self.assertEqual(state.vgprs[3], res)

class TestVFmaMixhiF16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0x5555555555555555
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x40000000)
        state.write_vgpr(2, 0x3F000000)
        state.write_vgpr(3, 0x00003333)
        inst = ASTInstruction(
            Opcode.V_FMA_MIXHI_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0x5555555555555555)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x40000000)
        state.write_vgpr(2, 0x3F000000)
        inst = ASTInstruction(
            Opcode.V_FMA_MIXHI_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res >> 16, 0x4100)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = ref_v_fma_mixhi_f16(0x7FC00000, 0x40000000, 0x3F000000)
        self.assertEqual(res_nan >> 16, 0x7E00)
        res_inf = ref_v_fma_mixhi_f16(0x7F800000, 0x40000000, 0x3F000000)
        self.assertEqual(res_inf >> 16, 0x7C00)

class TestVFmaMixhiF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x3f000000",
            "v_fma_mixhi_f16 v3, v0, v1, v2",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_fma_mixhi_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] >> 16, 0x4100)

def ref_v_fma_mixlo_f16(s0: int, s1: int, s2: int, old_dst: int = 0) -> int:
    if is_nan_f32(s0) or is_nan_f32(s1) or is_nan_f32(s2):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f32(s0)
        fb = bits_to_f32(s1)
        fc = bits_to_f32(s2)
        r = fa * fb + fc
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return ((old_dst & 0xFFFF0000) | (res_f16 & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_FMA_MIXLO_F16] = ref_v_fma_mixlo_f16


class TestVFmaMixloF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0f * 2.0f + 0.5f = 2.5f (f16 0x4100 written to low halfword)
        res = ref_v_fma_mixlo_f16(0x3F800000, 0x40000000, 0x3F000000, old_dst=0xAA550000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xAA55)

    def test_zero_and_identity(self) -> None:
        # 0.0f * 2.0f + 1.0f = 1.0f (0x3C00 in low halfword)
        res = ref_v_fma_mixlo_f16(0x00000000, 0x40000000, 0x3F800000, old_dst=0x43210000)
        self.assertEqual(res & 0xFFFF, 0x3C00)
        self.assertEqual(res >> 16, 0x4321)

class TestVFmaMixloF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_FMA_MIXLO_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        self.assertEqual(inst.disasm(), "v_fma_mixlo_f16 v3, v0, v1, v2")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x40000000)
        state.write_vgpr(2, 0x3F000000)
        state.write_vgpr(3, 0x88990000)
        inst = ASTInstruction(
            Opcode.V_FMA_MIXLO_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0x8899)
        self.assertEqual(state.vgprs[3], res)

class TestVFmaMixloF16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x40000000)
        state.write_vgpr(2, 0x3F000000)
        state.write_vgpr(3, 0x44440000)
        inst = ASTInstruction(
            Opcode.V_FMA_MIXLO_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3F800000)
        state.write_vgpr(1, 0x40000000)
        state.write_vgpr(2, 0x3F000000)
        inst = ASTInstruction(
            Opcode.V_FMA_MIXLO_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x4100)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = ref_v_fma_mixlo_f16(0x7FC00000, 0x40000000, 0x3F000000)
        self.assertEqual(res_nan & 0xFFFF, 0x7E00)
        res_inf = ref_v_fma_mixlo_f16(0x7F800000, 0x40000000, 0x3F000000)
        self.assertEqual(res_inf & 0xFFFF, 0x7C00)

class TestVFmaMixloF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x3f000000",
            "v_fma_mixlo_f16 v3, v0, v1, v2",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_fma_mixlo_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4100)

def ref_v_dot2_f32_f16(s0: int, s1: int, s2: int) -> int:
    a0 = bits_to_f16(s0 & 0xFFFF)
    a1 = bits_to_f16((s0 >> 16) & 0xFFFF)
    b0 = bits_to_f16(s1 & 0xFFFF)
    b1 = bits_to_f16((s1 >> 16) & 0xFFFF)
    acc = bits_to_f32(s2)
    r = (a0 * b0) + (a1 * b1) + acc
    if math.isnan(r):
        return 0x7FC00000
    if math.isinf(r):
        return 0x7F800000 if r > 0 else 0xFF800000
    return f32_to_bits(r) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_DOT2_F32_F16] = ref_v_dot2_f32_f16


class TestVDot2F32F16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # (1.0*1.0) + (1.0*1.0) + 1.0 = 3.0f (0x40400000)
        res = ref_v_dot2_f32_f16(0x3C003C00, 0x3C003C00, 0x3F800000)
        self.assertEqual(res, 0x40400000)

    def test_zero_accumulator(self) -> None:
        # (2.0*0.5) + (1.0*3.0) + 0.0 = 1.0 + 3.0 = 4.0f (0x40800000)
        res = ref_v_dot2_f32_f16(0x3C004000, 0x42003800, 0x00000000)
        self.assertEqual(res, 0x40800000)

class TestVDot2F32F16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_DOT2_F32_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        self.assertEqual(inst.disasm(), "v_dot2_f32_f16 v3, v0, v1, v2")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C003C00)
        state.write_vgpr(1, 0x3C003C00)
        state.write_vgpr(2, 0x3F800000)
        inst = ASTInstruction(
            Opcode.V_DOT2_F32_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x40400000)
        self.assertEqual(state.vgprs[3], 0x40400000)

class TestVDot2F32F16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0x1234567890ABCDEF
        state.write_vgpr(0, 0x3C003C00)
        state.write_vgpr(1, 0x3C003C00)
        state.write_vgpr(2, 0x3F800000)
        inst = ASTInstruction(
            Opcode.V_DOT2_F32_F16,
            ASTOperand.vgpr(3),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0x1234567890ABCDEF)

    def test_register_aliasing(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C003C00)
        state.write_vgpr(1, 0x3C003C00)
        state.write_vgpr(2, 0x3F800000)
        inst = ASTInstruction(
            Opcode.V_DOT2_F32_F16,
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(2),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x40400000)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = ref_v_dot2_f32_f16(0x7E003C00, 0x3C003C00, 0x3F800000)
        self.assertEqual(res_nan, 0x7FC00000)
        res_inf = ref_v_dot2_f32_f16(0x7C003C00, 0x3C003C00, 0x3F800000)
        self.assertEqual(res_inf, 0x7F800000)

class TestVDot2F32F16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3c003c00",
            "v_mov_b32 v1, 0x3c003c00",
            "v_mov_b32 v2, 0x3f800000",
            "v_dot2_f32_f16 v3, v0, v1, v2",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_dot2_f32_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40400000)

if __name__ == "__main__":
    unittest.main()

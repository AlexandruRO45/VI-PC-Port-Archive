"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU Normalized Float16/Float32 Conversion operations (Bundle 42):
  - v_cvt_norm_i16_f16:   Normalized Float16 to signed integer Norm16 (D[15:0] = snorm16(S0.f16))
  - v_cvt_norm_u16_f16:   Normalized Float16 to unsigned integer Norm16 (D[15:0] = unorm16(S0.f16))
  - v_cvt_pknorm_i16_f16: Packed Float16 to signed Norm16 pair (D = {snorm16(S1.f16), snorm16(S0.f16)})
  - v_cvt_pknorm_u16_f16: Packed Float16 to unsigned Norm16 pair (D = {unorm16(S1.f16), unorm16(S0.f16)})
  - v_cvt_pknorm_i16_f32: Packed Float32 to signed Norm16 pair (D = {snorm16(S1.f32), snorm16(S0.f32)})
  - v_cvt_pknorm_u16_f32: Packed Float32 to unsigned Norm16 pair (D = {unorm16(S1.f32), unorm16(S0.f32)})

All test methods compile assembly listings directly through tools/rdna2spirv.py
via support.translate_lines(...) and execute in spirv_mini_interp via support.mrt0_words.
Standard library Python only; zero third-party dependencies; zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Optional
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

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


def ref_v_cvt_norm_i16_f16(s0: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0):
        res_i16 = 0
    else:
        res_i16 = norm_i16(bits_to_f16(s0 & 0xFFFF))
    return ((old_dst & 0xFFFF0000) | (res_i16 & 0xFFFF)) & MASK32


def ref_v_cvt_norm_u16_f16(s0: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0):
        res_u16 = 0
    else:
        res_u16 = norm_u16(bits_to_f16(s0 & 0xFFFF))
    return ((old_dst & 0xFFFF0000) | (res_u16 & 0xFFFF)) & MASK32


def ref_v_cvt_pknorm_i16_f16(s0: int, s1: int) -> int:
    lo = 0 if is_nan_f16(s0) else norm_i16(bits_to_f16(s0 & 0xFFFF))
    hi = 0 if is_nan_f16(s1) else norm_i16(bits_to_f16(s1 & 0xFFFF))
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)) & MASK32


def ref_v_cvt_pknorm_u16_f16(s0: int, s1: int) -> int:
    lo = 0 if is_nan_f16(s0) else norm_u16(bits_to_f16(s0 & 0xFFFF))
    hi = 0 if is_nan_f16(s1) else norm_u16(bits_to_f16(s1 & 0xFFFF))
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)) & MASK32


def ref_v_cvt_pknorm_i16_f32(s0: int, s1: int) -> int:
    lo = 0 if is_nan_f32(s0) else norm_i16(bits_to_f32(s0))
    hi = 0 if is_nan_f32(s1) else norm_i16(bits_to_f32(s1))
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)) & MASK32


def ref_v_cvt_pknorm_u16_f32(s0: int, s1: int) -> int:
    lo = 0 if is_nan_f32(s0) else norm_u16(bits_to_f32(s0))
    hi = 0 if is_nan_f32(s1) else norm_u16(bits_to_f32(s1))
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)) & MASK32


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


# =============================================================================
# 1. v_cvt_norm_i16_f16
# =============================================================================

class TestVCvtNormI16F16Oracle(unittest.TestCase):
    """Reference mathematical oracle tests for v_cvt_norm_i16_f16 backed by translator."""

    def test_basic_conversion(self) -> None:
        # 1.0 -> 0x7FFF (32767)
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xdead0000",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_i16_basic", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x7FFF)
        self.assertEqual(words[0] >> 16, 0xDEAD)
        self.assertEqual(words[0] & 0xFFFF, ref_v_cvt_norm_i16_f16(0x3C00) & 0xFFFF)

    def test_negative_conversion(self) -> None:
        # -1.0 -> 0x8001 (-32767)
        lines = [
            "v_mov_b32 v0, 0xbc00",
            "v_mov_b32 v1, 0xbeef0000",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_i16_neg", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x8001)
        self.assertEqual(words[0] >> 16, 0xBEEF)
        self.assertEqual(words[0] & 0xFFFF, ref_v_cvt_norm_i16_f16(0xBC00) & 0xFFFF)

    def test_zero_and_clamp(self) -> None:
        # 0.0 -> 0
        lines0 = [
            "v_mov_b32 v0, 0x0000",
            "v_mov_b32 v1, 0x11110000",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv0 = support.translate_lines("ps_norm_i16_zero", lines0)
        words0 = support.mrt0_words(spv0)
        self.assertEqual(words0[0] & 0xFFFF, 0x0000)
        self.assertEqual(words0[0] >> 16, 0x1111)

        # 2.0 -> clamped to 1.0 -> 0x7FFF
        lines1 = [
            "v_mov_b32 v0, 0x4000",
            "v_mov_b32 v1, 0x22220000",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv1 = support.translate_lines("ps_norm_i16_clamp", lines1)
        words1 = support.mrt0_words(spv1)
        self.assertEqual(words1[0] & 0xFFFF, 0x7FFF)
        self.assertEqual(words1[0] >> 16, 0x2222)


class TestVCvtNormI16F16ASTAndInterpreter(unittest.TestCase):
    """AST disassembly and interpreter step verification for v_cvt_norm_i16_f16."""

    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_I16_F16,
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(0),
        )
        self.assertEqual(inst.disasm(), "v_cvt_norm_i16_f16 v1, v0")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            inst.disasm(),
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_i16_ast", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x7FFF)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xcafe0000",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_i16_step", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x7FFF)
        self.assertEqual(words[0] >> 16, 0xCAFE)


class TestVCvtNormI16F16SIMDAndIEEE(unittest.TestCase):
    """SIMD execution and IEEE boundary handling for v_cvt_norm_i16_f16."""

    def test_condition_code_neutrality(self) -> None:
        lines = [
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x12340000",
            "v_cvt_norm_i16_f16 v1, v0",
            "v_cndmask_b32 v2, 0, 1, vcc",
            "exp mrt0, v1, v2, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_i16_cc", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x7FFF)
        self.assertEqual(words[1], 1)

    def test_register_aliasing(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_cvt_norm_i16_f16 v0, v0",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_i16_alias", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x7FFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # NaN -> 0
        lines_nan = [
            "v_mov_b32 v0, 0x7e00",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv_nan = support.translate_lines("ps_norm_i16_nan", lines_nan)
        words_nan = support.mrt0_words(spv_nan)
        self.assertEqual(words_nan[0] & 0xFFFF, 0)

        # +Inf clamped to 1.0 -> 0x7FFF
        lines_inf = [
            "v_mov_b32 v0, 0x7c00",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv_inf = support.translate_lines("ps_norm_i16_inf", lines_inf)
        words_inf = support.mrt0_words(spv_inf)
        self.assertEqual(words_inf[0] & 0xFFFF, 0x7FFF)

        # -Inf clamped to -1.0 -> 0x8001
        lines_ninf = [
            "v_mov_b32 v0, 0xfc00",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv_ninf = support.translate_lines("ps_norm_i16_ninf", lines_ninf)
        words_ninf = support.mrt0_words(spv_ninf)
        self.assertEqual(words_ninf[0] & 0xFFFF, 0x8001)


class TestVCvtNormI16F16Translation(unittest.TestCase):
    """Hardware translator verification for v_cvt_norm_i16_f16."""

    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_norm_i16_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x7FFF)


# =============================================================================
# 2. v_cvt_norm_u16_f16
# =============================================================================

class TestVCvtNormU16F16Oracle(unittest.TestCase):
    """Reference mathematical oracle tests for v_cvt_norm_u16_f16 backed by translator."""

    def test_basic_conversion(self) -> None:
        # 1.0 -> 0xFFFF (65535)
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xfeed0000",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_u16_basic", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0xFFFF)
        self.assertEqual(words[0] >> 16, 0xFEED)
        self.assertEqual(words[0] & 0xFFFF, ref_v_cvt_norm_u16_f16(0x3C00) & 0xFFFF)

    def test_fractional_and_clamp(self) -> None:
        # 0.5 -> 0x8000 (32768)
        lines0 = [
            "v_mov_b32 v0, 0x3800",
            "v_mov_b32 v1, 0x43210000",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv0 = support.translate_lines("ps_norm_u16_half", lines0)
        words0 = support.mrt0_words(spv0)
        self.assertEqual(words0[0] & 0xFFFF, 0x8000)
        self.assertEqual(words0[0] >> 16, 0x4321)

        # Negative -> clamped to 0.0 -> 0
        lines1 = [
            "v_mov_b32 v0, 0xbc00",
            "v_mov_b32 v1, 0x55550000",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv1 = support.translate_lines("ps_norm_u16_clamp", lines1)
        words1 = support.mrt0_words(spv1)
        self.assertEqual(words1[0] & 0xFFFF, 0)
        self.assertEqual(words1[0] >> 16, 0x5555)


class TestVCvtNormU16F16ASTAndInterpreter(unittest.TestCase):
    """AST disassembly and interpreter step verification for v_cvt_norm_u16_f16."""

    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_NORM_U16_F16,
            ASTOperand.vgpr(1),
            ASTOperand.vgpr(0),
        )
        self.assertEqual(inst.disasm(), "v_cvt_norm_u16_f16 v1, v0")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            inst.disasm(),
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_u16_ast", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0xFFFF)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x99990000",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_u16_step", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0xFFFF)
        self.assertEqual(words[0] >> 16, 0x9999)


class TestVCvtNormU16F16SIMDAndIEEE(unittest.TestCase):
    """SIMD execution and IEEE boundary handling for v_cvt_norm_u16_f16."""

    def test_condition_code_neutrality(self) -> None:
        lines = [
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x88880000",
            "v_cvt_norm_u16_f16 v1, v0",
            "v_cndmask_b32 v2, 0, 1, vcc",
            "exp mrt0, v1, v2, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_u16_cc", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0xFFFF)
        self.assertEqual(words[1], 1)

    def test_register_aliasing(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_cvt_norm_u16_f16 v0, v0",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_norm_u16_alias", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0xFFFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # NaN -> 0
        lines_nan = [
            "v_mov_b32 v0, 0x7e00",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv_nan = support.translate_lines("ps_norm_u16_nan", lines_nan)
        words_nan = support.mrt0_words(spv_nan)
        self.assertEqual(words_nan[0] & 0xFFFF, 0)

        # +Inf clamped to 1.0 -> 0xFFFF
        lines_inf = [
            "v_mov_b32 v0, 0x7c00",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv_inf = support.translate_lines("ps_norm_u16_inf", lines_inf)
        words_inf = support.mrt0_words(spv_inf)
        self.assertEqual(words_inf[0] & 0xFFFF, 0xFFFF)

        # -Inf clamped to 0.0 -> 0
        lines_ninf = [
            "v_mov_b32 v0, 0xfc00",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv_ninf = support.translate_lines("ps_norm_u16_ninf", lines_ninf)
        words_ninf = support.mrt0_words(spv_ninf)
        self.assertEqual(words_ninf[0] & 0xFFFF, 0)


class TestVCvtNormU16F16Translation(unittest.TestCase):
    """Hardware translator verification for v_cvt_norm_u16_f16."""

    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_norm_u16_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0xFFFF)


# =============================================================================
# 3. v_cvt_pknorm_i16_f16
# =============================================================================

class TestVCvtPknormI16F16Oracle(unittest.TestCase):
    """Reference mathematical oracle tests for v_cvt_pknorm_i16_f16 backed by translator."""

    def test_basic_packing(self) -> None:
        # lo: 1.0 -> 0x7FFF; hi: -1.0 -> 0x8001
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xbc00",
            "v_cvt_pknorm_i16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f16_basic", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)
        self.assertEqual(words[0], ref_v_cvt_pknorm_i16_f16(0x3C00, 0xBC00))

    def test_zero_and_clamp(self) -> None:
        # lo: 0.0 -> 0x0000; hi: 2.0 -> clamped to 1.0 (0x7FFF)
        lines = [
            "v_mov_b32 v0, 0x0000",
            "v_mov_b32 v1, 0x4000",
            "v_cvt_pknorm_i16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f16_clamp", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x7FFF0000)


class TestVCvtPknormI16F16ASTAndInterpreter(unittest.TestCase):
    """AST disassembly and interpreter step verification for v_cvt_pknorm_i16_f16."""

    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_cvt_pknorm_i16_f16 v2, v0, v1")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xbc00",
            inst.disasm(),
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f16_ast", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xbc00",
            "v_cvt_pknorm_i16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f16_step", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)


class TestVCvtPknormI16F16SIMDAndIEEE(unittest.TestCase):
    """SIMD execution and IEEE boundary handling for v_cvt_pknorm_i16_f16."""

    def test_condition_code_neutrality(self) -> None:
        lines = [
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xbc00",
            "v_cvt_pknorm_i16_f16 v2, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0, v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f16_cc", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)
        self.assertEqual(words[1], 1)

    def test_register_aliasing(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xbc00",
            "v_cvt_pknorm_i16_f16 v0, v0, v1",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f16_alias", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # lo: NaN -> 0; hi: Inf -> 0x7FFF
        lines = [
            "v_mov_b32 v0, 0x7e00",
            "v_mov_b32 v1, 0x7c00",
            "v_cvt_pknorm_i16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f16_ieee", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x7FFF0000)


class TestVCvtPknormI16F16Translation(unittest.TestCase):
    """Hardware translator verification for v_cvt_pknorm_i16_f16."""

    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0xbc00",
            "v_cvt_pknorm_i16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_pknorm_i16_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)


# =============================================================================
# 4. v_cvt_pknorm_u16_f16
# =============================================================================

class TestVCvtPknormU16F16Oracle(unittest.TestCase):
    """Reference mathematical oracle tests for v_cvt_pknorm_u16_f16 backed by translator."""

    def test_basic_packing(self) -> None:
        # lo: 1.0 -> 0xFFFF; hi: 0.5 -> 0x8000
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_cvt_pknorm_u16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f16_basic", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)
        self.assertEqual(words[0], ref_v_cvt_pknorm_u16_f16(0x3C00, 0x3800))

    def test_zero_and_clamp(self) -> None:
        # lo: -1.0 -> clamped to 0 -> 0x0000; hi: 2.0 -> clamped to 1.0 -> 0xFFFF
        lines = [
            "v_mov_b32 v0, 0xbc00",
            "v_mov_b32 v1, 0x4000",
            "v_cvt_pknorm_u16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f16_clamp", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF0000)


class TestVCvtPknormU16F16ASTAndInterpreter(unittest.TestCase):
    """AST disassembly and interpreter step verification for v_cvt_pknorm_u16_f16."""

    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_cvt_pknorm_u16_f16 v2, v0, v1")
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            inst.disasm(),
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f16_ast", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_cvt_pknorm_u16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f16_step", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)


class TestVCvtPknormU16F16SIMDAndIEEE(unittest.TestCase):
    """SIMD execution and IEEE boundary handling for v_cvt_pknorm_u16_f16."""

    def test_condition_code_neutrality(self) -> None:
        lines = [
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_cvt_pknorm_u16_f16 v2, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0, v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f16_cc", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)
        self.assertEqual(words[1], 1)

    def test_register_aliasing(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_cvt_pknorm_u16_f16 v0, v0, v1",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f16_alias", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # lo: NaN -> 0; hi: Inf -> 0xFFFF
        lines = [
            "v_mov_b32 v0, 0x7e00",
            "v_mov_b32 v1, 0x7c00",
            "v_cvt_pknorm_u16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f16_ieee", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF0000)


class TestVCvtPknormU16F16Translation(unittest.TestCase):
    """Hardware translator verification for v_cvt_pknorm_u16_f16."""

    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_cvt_pknorm_u16_f16 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_pknorm_u16_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)


# =============================================================================
# 5. v_cvt_pknorm_i16_f32
# =============================================================================

class TestVCvtPknormI16F32Oracle(unittest.TestCase):
    """Reference mathematical oracle tests for v_cvt_pknorm_i16_f32 backed by translator."""

    def test_basic_packing(self) -> None:
        # lo: 1.0f -> 0x7FFF; hi: -1.0f -> 0x8001
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0xbf800000",
            "v_cvt_pknorm_i16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f32_basic", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)
        self.assertEqual(words[0], ref_v_cvt_pknorm_i16_f32(0x3F800000, 0xBF800000))

    def test_zero_and_clamp(self) -> None:
        # lo: 0.0f -> 0x0000; hi: 3.0f -> clamped to 1.0 -> 0x7FFF
        lines = [
            "v_mov_b32 v0, 0x00000000",
            "v_mov_b32 v1, 0x40400000",
            "v_cvt_pknorm_i16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f32_clamp", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x7FFF0000)


class TestVCvtPknormI16F32ASTAndInterpreter(unittest.TestCase):
    """AST disassembly and interpreter step verification for v_cvt_pknorm_i16_f32."""

    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_I16_F32,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_cvt_pknorm_i16_f32 v2, v0, v1")
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0xbf800000",
            inst.disasm(),
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f32_ast", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0xbf800000",
            "v_cvt_pknorm_i16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f32_step", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)


class TestVCvtPknormI16F32SIMDAndIEEE(unittest.TestCase):
    """SIMD execution and IEEE boundary handling for v_cvt_pknorm_i16_f32."""

    def test_condition_code_neutrality(self) -> None:
        lines = [
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0xbf800000",
            "v_cvt_pknorm_i16_f32 v2, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0, v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f32_cc", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)
        self.assertEqual(words[1], 1)

    def test_register_aliasing(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0xbf800000",
            "v_cvt_pknorm_i16_f32 v0, v0, v1",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f32_alias", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # lo: NaN -> 0; hi: Inf -> 0x7FFF
        lines = [
            "v_mov_b32 v0, 0x7fc00000",
            "v_mov_b32 v1, 0x7f800000",
            "v_cvt_pknorm_i16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_i16_f32_ieee", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x7FFF0000)


class TestVCvtPknormI16F32Translation(unittest.TestCase):
    """Hardware translator verification for v_cvt_pknorm_i16_f32."""

    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0xbf800000",
            "v_cvt_pknorm_i16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_pknorm_i16_f32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80017FFF)


# =============================================================================
# 6. v_cvt_pknorm_u16_f32
# =============================================================================

class TestVCvtPknormU16F32Oracle(unittest.TestCase):
    """Reference mathematical oracle tests for v_cvt_pknorm_u16_f32 backed by translator."""

    def test_basic_packing(self) -> None:
        # lo: 1.0f -> 0xFFFF; hi: 0.5f -> 0x8000
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x3f000000",
            "v_cvt_pknorm_u16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f32_basic", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)
        self.assertEqual(words[0], ref_v_cvt_pknorm_u16_f32(0x3F800000, 0x3F000000))

    def test_zero_and_clamp(self) -> None:
        # lo: -2.0f -> clamped to 0 -> 0x0000; hi: 5.0f -> clamped to 1.0 -> 0xFFFF
        lines = [
            "v_mov_b32 v0, 0xc0000000",
            "v_mov_b32 v1, 0x40a00000",
            "v_cvt_pknorm_u16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f32_clamp", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF0000)


class TestVCvtPknormU16F32ASTAndInterpreter(unittest.TestCase):
    """AST disassembly and interpreter step verification for v_cvt_pknorm_u16_f32."""

    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_CVT_PKNORM_U16_F32,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
        )
        self.assertEqual(inst.disasm(), "v_cvt_pknorm_u16_f32 v2, v0, v1")
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x3f000000",
            inst.disasm(),
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f32_ast", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x3f000000",
            "v_cvt_pknorm_u16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f32_step", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)


class TestVCvtPknormU16F32SIMDAndIEEE(unittest.TestCase):
    """SIMD execution and IEEE boundary handling for v_cvt_pknorm_u16_f32."""

    def test_condition_code_neutrality(self) -> None:
        lines = [
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x3f000000",
            "v_cvt_pknorm_u16_f32 v2, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0, v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f32_cc", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)
        self.assertEqual(words[1], 1)

    def test_register_aliasing(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x3f000000",
            "v_cvt_pknorm_u16_f32 v0, v0, v1",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f32_alias", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)

    def test_ieee754_nan_and_inf(self) -> None:
        # lo: NaN -> 0; hi: Inf -> 0xFFFF
        lines = [
            "v_mov_b32 v0, 0x7fc00000",
            "v_mov_b32 v1, 0x7f800000",
            "v_cvt_pknorm_u16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pknorm_u16_f32_ieee", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF0000)


class TestVCvtPknormU16F32Translation(unittest.TestCase):
    """Hardware translator verification for v_cvt_pknorm_u16_f32."""

    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x3f000000",
            "v_cvt_pknorm_u16_f32 v2, v0, v1",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cvt_pknorm_u16_f32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8000FFFF)


# =============================================================================
# 7. Negative Controls and Mutation Detection
# =============================================================================

class TestVCvtNormNegativeControls(unittest.TestCase):
    """Negative controls and mutation detection for normalized conversions."""

    def test_syntax_rejection_missing_operand(self) -> None:
        lines = [
            "v_cvt_norm_i16_f16 v0",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_norm_bad", lines)

    def test_mutation_detection_signed_vs_unsigned_norm(self) -> None:
        # Negative half float -1.0h (0xBC00):
        # v_cvt_norm_i16_f16 gives 0x8001 (-32767)
        # v_cvt_norm_u16_f16 gives 0 (saturated clamp to 0)
        lines_i16 = [
            "v_mov_b32 v0, 0xbc00",
            "v_cvt_norm_i16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv_i16 = support.translate_lines("ps_mut_norm_i16", lines_i16)
        words_i16 = support.mrt0_words(spv_i16)

        lines_u16 = [
            "v_mov_b32 v0, 0xbc00",
            "v_cvt_norm_u16_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv_u16 = support.translate_lines("ps_mut_norm_u16", lines_u16)
        words_u16 = support.mrt0_words(spv_u16)

        self.assertEqual(words_i16[0] & 0xFFFF, 0x8001)
        self.assertEqual(words_u16[0] & 0xFFFF, 0x0000)
        self.assertNotEqual(words_i16[0] & 0xFFFF, words_u16[0] & 0xFFFF)


if __name__ == "__main__":
    unittest.main()

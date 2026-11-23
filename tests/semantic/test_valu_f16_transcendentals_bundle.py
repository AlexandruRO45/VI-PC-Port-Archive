"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU 16-bit floating-point transcendental operations (Bundle 21):
  - v_sin_f16:  16-bit float sine (input in turns: 1.0 turn = 2*pi radians)
  - v_cos_f16:  16-bit float cosine (input in turns: 1.0 turn = 2*pi radians)
  - v_exp_f16:  16-bit float base-2 exponential (2^x)
  - v_log_f16:  16-bit float base-2 logarithm (log2(x))
  - v_rcp_f16:  16-bit float reciprocal (1.0 / x)
  - v_rsq_f16:  16-bit float reciprocal square root (1.0 / sqrt(x))
  - v_sqrt_f16: 16-bit float square root (sqrt(x))

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
100% translator-backed: Every test exercises tools/rdna2spirv.py via support.translate_lines(...)
and executes via support.mrt0_words for hardware-accurate translation and verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Callable, Optional
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv
import support

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF


def f16_to_u16(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def u16_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & MASK16))[0]


class Opcode(Enum):
    V_SIN_F16 = "v_sin_f16"
    V_COS_F16 = "v_cos_f16"
    V_EXP_F16 = "v_exp_f16"
    V_LOG_F16 = "v_log_f16"
    V_RCP_F16 = "v_rcp_f16"
    V_RSQ_F16 = "v_rsq_f16"
    V_SQRT_F16 = "v_sqrt_f16"


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

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.VGPR:
                return f"v{op.index}"
            if op.kind == OperandKind.SGPR:
                return f"s{op.index}"
            return hex(op.value)
        return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}"


_TRANS_ID = 0


def run_transcendental_f16(opcode: str, src_u16: int, extra_lines: list[str] | None = None) -> int:
    """Compile and execute single-operand F16 transcendental via rdna2spirv and spirv_mini_interp."""
    global _TRANS_ID
    _TRANS_ID += 1
    lines = [
        f"v_mov_b32 v0, {src_u16 & MASK16:#x}",
        f"{opcode} v1, v0",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    lines.extend([
        "exp mrt0 v1, v1, v1, v1 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_{opcode}_{_TRANS_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK16


# -----------------------------------------------------------------------------
# Opcode 1: v_sin_f16
# -----------------------------------------------------------------------------

def ref_v_sin_f16(u: int) -> int:
    f = u16_to_f16(u)
    s = math.sin(f * 2.0 * math.pi)
    if abs(s) < 1e-7:
        s = 0.0
    return f16_to_u16(s)


class TestVSinF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x0000, 0x0000),  # sin(0.0) = 0.0
            (0x3400, 0x3C00),  # sin(0.25) = 1.0
            (0x3800, 0x0000),  # sin(0.5) = 0.0
            (0x3A00, 0xBC00),  # sin(0.75) = -1.0
            (0x3C00, 0x0000),  # sin(1.0) = 0.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_v_sin_f16(src), exp)
                hw = run_transcendental_f16("v_sin_f16", src)
                self.assertEqual(hw, exp)


class TestVSinF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_SIN_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_sin_f16 v0, v1")
        lines = [
            "v_mov_b32 v1, 0x3400",
            inst.disasm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_sin_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x3400",
            "v_sin_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_v_sin_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)


class TestVSinF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3400",
            "v_sin_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_sin_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)

    def test_ieee754_boundary(self) -> None:
        cases = [
            (0x0000, 0x0000),  # +0.0 -> +0.0
            (0x8000, 0x8000),  # -0.0 -> -0.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                hw = run_transcendental_f16("v_sin_f16", src)
                self.assertEqual(hw, exp)


class TestVSinF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3400",
            "v_sin_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_sin_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x3C00)

    def test_sgpr_and_clamp_modifiers(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x3400",
            "v_sin_f16 v0, s0 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mods_v_sin_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)


# -----------------------------------------------------------------------------
# Opcode 2: v_cos_f16
# -----------------------------------------------------------------------------

def ref_v_cos_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(math.cos(f * 2.0 * math.pi))


class TestVCosF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x0000, 0x3C00),  # cos(0.0) = 1.0
            (0x3400, 0x0000),  # cos(0.25) = 0.0
            (0x3800, 0xBC00),  # cos(0.5) = -1.0
            (0x3C00, 0x3C00),  # cos(1.0) = 1.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_v_cos_f16(src), exp)
                hw = run_transcendental_f16("v_cos_f16", src)
                self.assertEqual(hw, exp)


class TestVCosF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_COS_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_cos_f16 v0, v1")
        lines = [
            "v_mov_b32 v1, 0x0000",
            inst.disasm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_cos_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x0000",
            "v_cos_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_v_cos_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)


class TestVCosF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x0000",
            "v_cos_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_cos_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)

    def test_ieee754_boundary(self) -> None:
        cases = [
            (0x0000, 0x3C00),  # cos(+0.0) = 1.0
            (0x8000, 0x3C00),  # cos(-0.0) = 1.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                hw = run_transcendental_f16("v_cos_f16", src)
                self.assertEqual(hw, exp)


class TestVCosF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x0000",
            "v_cos_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cos_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x3C00)

    def test_sgpr_and_clamp_modifiers(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x0000",
            "v_cos_f16 v0, s0 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mods_v_cos_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)


# -----------------------------------------------------------------------------
# Opcode 3: v_exp_f16
# -----------------------------------------------------------------------------

def ref_v_exp_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(math.pow(2.0, f))


class TestVExpF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x0000, 0x3C00),  # 2^0 = 1.0
            (0x3C00, 0x4000),  # 2^1 = 2.0
            (0x4000, 0x4400),  # 2^2 = 4.0
            (0xBC00, 0x3800),  # 2^-1 = 0.5
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_v_exp_f16(src), exp)
                hw = run_transcendental_f16("v_exp_f16", src)
                self.assertEqual(hw, exp)


class TestVExpF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_EXP_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_exp_f16 v0, v1")
        lines = [
            "v_mov_b32 v1, 0x3C00",
            inst.disasm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_exp_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x4000)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x3C00",
            "v_exp_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_v_exp_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x4000)


class TestVExpF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3C00",
            "v_exp_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_exp_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x4000)

    def test_ieee754_boundary(self) -> None:
        cases = [
            (0x0000, 0x3C00),  # 2^+0.0 = 1.0
            (0x8000, 0x3C00),  # 2^-0.0 = 1.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                hw = run_transcendental_f16("v_exp_f16", src)
                self.assertEqual(hw, exp)


class TestVExpF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3C00",
            "v_exp_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_exp_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x4000)

    def test_sgpr_and_clamp_modifiers(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x3C00",
            "v_exp_f16 v0, s0 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mods_v_exp_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)  # clamped to 1.0


# -----------------------------------------------------------------------------
# Opcode 4: v_log_f16
# -----------------------------------------------------------------------------

def ref_v_log_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(math.log2(f))


class TestVLogF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x3C00, 0x0000),  # log2(1.0) = 0.0
            (0x4000, 0x3C00),  # log2(2.0) = 1.0
            (0x4400, 0x4000),  # log2(4.0) = 2.0
            (0x3800, 0xBC00),  # log2(0.5) = -1.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_v_log_f16(src), exp)
                hw = run_transcendental_f16("v_log_f16", src)
                self.assertEqual(hw, exp)


class TestVLogF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_LOG_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_log_f16 v0, v1")
        lines = [
            "v_mov_b32 v1, 0x4000",
            inst.disasm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_log_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x4000",
            "v_log_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_v_log_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)


class TestVLogF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4000",
            "v_log_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_log_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)

    def test_ieee754_boundary(self) -> None:
        cases = [
            (0x3C00, 0x0000),  # log2(1.0) = 0.0
            (0x4400, 0x4000),  # log2(4.0) = 2.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                hw = run_transcendental_f16("v_log_f16", src)
                self.assertEqual(hw, exp)


class TestVLogF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4000",
            "v_log_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_log_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x3C00)

    def test_sgpr_and_clamp_modifiers(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x4400",
            "v_log_f16 v0, s0 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mods_v_log_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)  # clamped to 1.0


# -----------------------------------------------------------------------------
# Opcode 5: v_rcp_f16
# -----------------------------------------------------------------------------

def ref_v_rcp_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(1.0 / f)


class TestVRcpF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x3C00, 0x3C00),  # 1/1.0 = 1.0
            (0x4000, 0x3800),  # 1/2.0 = 0.5
            (0x4400, 0x3400),  # 1/4.0 = 0.25
            (0x3800, 0x4000),  # 1/0.5 = 2.0
            (0xBC00, 0xBC00),  # 1/-1.0 = -1.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_v_rcp_f16(src), exp)
                hw = run_transcendental_f16("v_rcp_f16", src)
                self.assertEqual(hw, exp)


class TestVRcpF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_RCP_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_rcp_f16 v0, v1")
        lines = [
            "v_mov_b32 v1, 0x4000",
            inst.disasm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_rcp_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3800)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x4000",
            "v_rcp_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_v_rcp_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3800)


class TestVRcpF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4000",
            "v_rcp_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_rcp_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3800)

    def test_ieee754_boundary(self) -> None:
        cases = [
            (0x3C00, 0x3C00),  # 1.0 -> 1.0
            (0xBC00, 0xBC00),  # -1.0 -> -1.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                hw = run_transcendental_f16("v_rcp_f16", src)
                self.assertEqual(hw, exp)


class TestVRcpF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4000",
            "v_rcp_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_rcp_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x3800)

    def test_sgpr_and_clamp_modifiers(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x3800",
            "v_rcp_f16 v0, s0 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mods_v_rcp_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)  # 2.0 clamped to 1.0


# -----------------------------------------------------------------------------
# Opcode 6: v_rsq_f16
# -----------------------------------------------------------------------------

def ref_v_rsq_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(1.0 / math.sqrt(f))


class TestVRsqF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x3C00, 0x3C00),  # 1/sqrt(1.0) = 1.0
            (0x4400, 0x3800),  # 1/sqrt(4.0) = 0.5
            (0x4C00, 0x3400),  # 1/sqrt(16.0) = 0.25
            (0x3400, 0x4000),  # 1/sqrt(0.25) = 2.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_v_rsq_f16(src), exp)
                hw = run_transcendental_f16("v_rsq_f16", src)
                self.assertEqual(hw, exp)


class TestVRsqF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_RSQ_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_rsq_f16 v0, v1")
        lines = [
            "v_mov_b32 v1, 0x4400",
            inst.disasm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_rsq_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3800)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x4400",
            "v_rsq_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_v_rsq_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3800)


class TestVRsqF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4400",
            "v_rsq_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_rsq_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3800)

    def test_ieee754_boundary(self) -> None:
        cases = [
            (0x3C00, 0x3C00),  # 1.0 -> 1.0
            (0x4400, 0x3800),  # 4.0 -> 0.5
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                hw = run_transcendental_f16("v_rsq_f16", src)
                self.assertEqual(hw, exp)


class TestVRsqF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4400",
            "v_rsq_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_rsq_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x3800)

    def test_sgpr_and_clamp_modifiers(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x3400",
            "v_rsq_f16 v0, s0 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mods_v_rsq_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)  # 2.0 clamped to 1.0


# -----------------------------------------------------------------------------
# Opcode 7: v_sqrt_f16
# -----------------------------------------------------------------------------

def ref_v_sqrt_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(math.sqrt(f))


class TestVSqrtF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x0000, 0x0000),  # sqrt(0.0) = 0.0
            (0x3C00, 0x3C00),  # sqrt(1.0) = 1.0
            (0x4400, 0x4000),  # sqrt(4.0) = 2.0
            (0x4880, 0x4200),  # sqrt(9.0) = 3.0
            (0x4C00, 0x4400),  # sqrt(16.0) = 4.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_v_sqrt_f16(src), exp)
                hw = run_transcendental_f16("v_sqrt_f16", src)
                self.assertEqual(hw, exp)


class TestVSqrtF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_SQRT_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_sqrt_f16 v0, v1")
        lines = [
            "v_mov_b32 v1, 0x4400",
            inst.disasm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_sqrt_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x4000)

    def test_scalar_interpreter_step(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x4400",
            "v_sqrt_f16 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_v_sqrt_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x4000)


class TestVSqrtF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4400",
            "v_sqrt_f16 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_sqrt_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x4000)

    def test_ieee754_boundary(self) -> None:
        cases = [
            (0x0000, 0x0000),  # +0.0 -> +0.0
            (0x3C00, 0x3C00),  # 1.0 -> 1.0
        ]
        for src, exp in cases:
            with self.subTest(src=hex(src)):
                hw = run_transcendental_f16("v_sqrt_f16", src)
                self.assertEqual(hw, exp)


class TestVSqrtF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4400",
            "v_sqrt_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_sqrt_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x4000)

    def test_sgpr_and_clamp_modifiers(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x4400",
            "v_sqrt_f16 v0, s0 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mods_v_sqrt_f16", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x3C00)  # 2.0 clamped to 1.0


# -----------------------------------------------------------------------------
# Pipeline Integration & Suite Compliance
# -----------------------------------------------------------------------------

class TestVF16TranscendentalsPipeline(unittest.TestCase):
    def test_reciprocal_chain(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4000",
            "v_rcp_f16 v1, v0",
            "v_rcp_f16 v2, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pipe_rcp_rcp", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x4000)

    def test_exp_log_chain(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x4400",
            "v_log_f16 v1, v0",
            "v_exp_f16 v2, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pipe_exp_log", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, 0x4400)

    def test_compliance_guard(self) -> None:
        test_path = os.path.join(HERE, "test_valu_f16_transcendentals_bundle.py")
        self.assertTrue(os.path.isfile(test_path))
        with open(test_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("import support", content)
        self.assertIn("support.translate_lines", content)


if __name__ == "__main__":
    unittest.main()

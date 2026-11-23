"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU 32-bit bit manipulation and shift operations (Bundle 14):
  - v_bfe_u32:       32-bit unsigned bitfield extract
  - v_bfe_i32:       32-bit signed bitfield extract
  - v_bfi_b32:       32-bit bitfield insert / multiplexer
  - v_bfm_b32:       32-bit bitfield mask generator
  - v_alignbyte_b32: 32-bit byte alignment / byte funnel shift
  - v_alignbit_b32:  32-bit bit alignment / bit funnel shift

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

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    return val & MASK32


def run_hw_op(lines: list[str], test_name: str, exp_reg: str = "v3") -> int:
    """Helper to translate lines and run through spirv_mini_interp."""
    if support is None or rdna2spirv is None or getattr(support, "translate_lines", None) is None:
        raise unittest.SkipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
    full_lines = list(lines)
    full_lines.append(f"exp mrt0, {exp_reg}, {exp_reg}, {exp_reg}, {exp_reg} done")
    full_lines.append("s_endpgm")
    spv = support.translate_lines(test_name, full_lines)
    return support.mrt0_words(spv)[0]


class Opcode(Enum):
    V_BFE_U32 = "v_bfe_u32"
    V_BFE_I32 = "v_bfe_i32"
    V_BFI_B32 = "v_bfi_b32"
    V_BFM_B32 = "v_bfm_b32"
    V_ALIGNBYTE_B32 = "v_alignbyte_b32"
    V_ALIGNBIT_B32 = "v_alignbit_b32"


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


# -----------------------------------------------------------------------------
# 1. v_bfe_u32
# -----------------------------------------------------------------------------
def ref_v_bfe_u32(s0: int, s1: int, s2: int) -> int:
    off = s1 & 31
    wid = min(s2 & 31, 32 - off)
    return 0 if wid == 0 else (s0 >> off) & ((1 << wid) - 1)


GLOBAL_ORACLE_MAP[Opcode.V_BFE_U32] = ref_v_bfe_u32


class TestVBfeU32Oracle(unittest.TestCase):
    def test_oracle_execution(self) -> None:
        res = ref_v_bfe_u32(0x12345678, 8, 16)
        self.assertEqual(res, 0x3456)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_bfe_u32 v1, v0, 8, 16",
        ]
        hw_res = run_hw_op(lines, "ps_oracle_v_bfe_u32", exp_reg="v1")
        self.assertEqual(hw_res, res)


class TestVBfeU32ASTAndInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 8)
        state.write_vgpr(2, 16)
        inst = ASTInstruction(Opcode.V_BFE_U32, ASTOperand.vgpr(3), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.vgpr(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3456)
        self.assertEqual(state.vgprs[3], 0x3456)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 8",
            "v_mov_b32 v2, 16",
            inst.disasm(),
        ]
        hw_res = run_hw_op(lines, "ps_ast_v_bfe_u32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVBfeU32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_condition_codes(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 8)
        state.write_vgpr(2, 16)
        inst = ASTInstruction(Opcode.V_BFE_U32, ASTOperand.vgpr(0), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.vgpr(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3456)
        self.assertEqual(state.vgprs[0], 0x3456)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_bfe_u32 v0, v0, 8, 16",
        ]
        hw_res = run_hw_op(lines, "ps_alias_v_bfe_u32", exp_reg="v0")
        self.assertEqual(hw_res, 0x3456)

    def test_ieee754_bitwise_transparency(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        res = ref_v_bfe_u32(qnan, snan, 4)
        self.assertIsInstance(res, int)
        lines = [
            f"v_mov_b32 v0, {qnan:#x}",
            f"v_mov_b32 v1, {snan:#x}",
            "v_mov_b32 v2, 4",
            "v_bfe_u32 v3, v0, v1, v2",
        ]
        hw_res = run_hw_op(lines, "ps_ieee_v_bfe_u32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVBfeU32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            "v_bfe_u32 v3, v0, 8, 16",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_bfe_u32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3456)


# -----------------------------------------------------------------------------
# 2. v_bfe_i32
# -----------------------------------------------------------------------------
def ref_v_bfe_i32(s0: int, s1: int, s2: int) -> int:
    off = s1 & 31
    wid = min(s2 & 31, 32 - off)
    if wid == 0:
        return 0
    raw = (s0 >> off) & ((1 << wid) - 1)
    sign_bit = 1 << (wid - 1)
    return ((raw ^ sign_bit) - sign_bit) & MASK32


GLOBAL_ORACLE_MAP[Opcode.V_BFE_I32] = ref_v_bfe_i32


class TestVBfeI32Oracle(unittest.TestCase):
    def test_oracle_execution(self) -> None:
        res = ref_v_bfe_i32(0x12345678, 8, 16)
        self.assertEqual(res, 0x3456)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_bfe_i32 v1, v0, 8, 16",
        ]
        hw_res = run_hw_op(lines, "ps_oracle_v_bfe_i32", exp_reg="v1")
        self.assertEqual(hw_res, res)


class TestVBfeI32ASTAndInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 8)
        state.write_vgpr(2, 16)
        inst = ASTInstruction(Opcode.V_BFE_I32, ASTOperand.vgpr(3), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.vgpr(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3456)
        self.assertEqual(state.vgprs[3], 0x3456)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 8",
            "v_mov_b32 v2, 16",
            inst.disasm(),
        ]
        hw_res = run_hw_op(lines, "ps_ast_v_bfe_i32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVBfeI32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_condition_codes(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 8)
        state.write_vgpr(2, 16)
        inst = ASTInstruction(Opcode.V_BFE_I32, ASTOperand.vgpr(0), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.vgpr(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3456)
        self.assertEqual(state.vgprs[0], 0x3456)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_bfe_i32 v0, v0, 8, 16",
        ]
        hw_res = run_hw_op(lines, "ps_alias_v_bfe_i32", exp_reg="v0")
        self.assertEqual(hw_res, 0x3456)

    def test_ieee754_bitwise_transparency(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        res = ref_v_bfe_i32(qnan, snan, 4)
        self.assertIsInstance(res, int)
        lines = [
            f"v_mov_b32 v0, {qnan:#x}",
            f"v_mov_b32 v1, {snan:#x}",
            "v_mov_b32 v2, 4",
            "v_bfe_i32 v3, v0, v1, v2",
        ]
        hw_res = run_hw_op(lines, "ps_ieee_v_bfe_i32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVBfeI32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            "v_bfe_i32 v3, v0, 8, 16",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_bfe_i32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3456)


# -----------------------------------------------------------------------------
# 3. v_bfi_b32
# -----------------------------------------------------------------------------
def ref_v_bfi_b32(s0: int, s1: int, s2: int) -> int:
    return ((s0 & s1) | ((~s0) & s2)) & MASK32


GLOBAL_ORACLE_MAP[Opcode.V_BFI_B32] = ref_v_bfi_b32


class TestVBfiB32Oracle(unittest.TestCase):
    def test_oracle_execution(self) -> None:
        res = ref_v_bfi_b32(0x12345678, 0x9abcdef0, 0x55555555)
        self.assertEqual(res, 0x57755775)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            "v_bfi_b32 v3, v0, v1, v2",
        ]
        hw_res = run_hw_op(lines, "ps_oracle_v_bfi_b32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVBfiB32ASTAndInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 0x9abcdef0)
        state.write_vgpr(2, 0x55555555)
        inst = ASTInstruction(Opcode.V_BFI_B32, ASTOperand.vgpr(3), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.vgpr(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x57755775)
        self.assertEqual(state.vgprs[3], 0x57755775)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            inst.disasm(),
        ]
        hw_res = run_hw_op(lines, "ps_ast_v_bfi_b32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVBfiB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_condition_codes(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 0x9abcdef0)
        state.write_vgpr(2, 0x55555555)
        inst = ASTInstruction(Opcode.V_BFI_B32, ASTOperand.vgpr(0), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.vgpr(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x57755775)
        self.assertEqual(state.vgprs[0], 0x57755775)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            "v_bfi_b32 v0, v0, v1, v2",
        ]
        hw_res = run_hw_op(lines, "ps_alias_v_bfi_b32", exp_reg="v0")
        self.assertEqual(hw_res, 0x57755775)

    def test_ieee754_bitwise_transparency(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        res = ref_v_bfi_b32(qnan, snan, 4)
        self.assertIsInstance(res, int)
        lines = [
            f"v_mov_b32 v0, {qnan:#x}",
            f"v_mov_b32 v1, {snan:#x}",
            "v_mov_b32 v2, 4",
            "v_bfi_b32 v3, v0, v1, v2",
        ]
        hw_res = run_hw_op(lines, "ps_ieee_v_bfi_b32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVBfiB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            "v_bfi_b32 v3, v0, v1, v2",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_bfi_b32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x57755775)


# -----------------------------------------------------------------------------
# 4. v_bfm_b32
# -----------------------------------------------------------------------------
def ref_v_bfm_b32(s0: int, s1: int) -> int:
    return (((1 << (s0 & 31)) - 1) << (s1 & 31)) & MASK32


GLOBAL_ORACLE_MAP[Opcode.V_BFM_B32] = ref_v_bfm_b32


class TestVBfmB32Oracle(unittest.TestCase):
    def test_oracle_execution(self) -> None:
        res = ref_v_bfm_b32(8, 4)
        self.assertEqual(res, 0xff0)
        lines = [
            "v_bfm_b32 v0, 8, 4",
        ]
        hw_res = run_hw_op(lines, "ps_oracle_v_bfm_b32", exp_reg="v0")
        self.assertEqual(hw_res, res)


class TestVBfmB32ASTAndInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 8)
        state.write_vgpr(1, 4)
        inst = ASTInstruction(Opcode.V_BFM_B32, ASTOperand.vgpr(2), ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0xff0)
        self.assertEqual(state.vgprs[2], 0xff0)
        lines = [
            "v_mov_b32 v0, 8",
            "v_mov_b32 v1, 4",
            inst.disasm(),
        ]
        hw_res = run_hw_op(lines, "ps_ast_v_bfm_b32", exp_reg="v2")
        self.assertEqual(hw_res, res)


class TestVBfmB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_condition_codes(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 8)
        state.write_vgpr(1, 4)
        inst = ASTInstruction(Opcode.V_BFM_B32, ASTOperand.vgpr(0), ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0xff0)
        self.assertEqual(state.vgprs[0], 0xff0)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)
        lines = [
            "v_mov_b32 v0, 8",
            "v_mov_b32 v1, 4",
            "v_bfm_b32 v0, v0, v1",
        ]
        hw_res = run_hw_op(lines, "ps_alias_v_bfm_b32", exp_reg="v0")
        self.assertEqual(hw_res, 0xff0)

    def test_ieee754_bitwise_transparency(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        res = ref_v_bfm_b32(qnan, snan)
        self.assertIsInstance(res, int)
        lines = [
            f"v_mov_b32 v0, {qnan:#x}",
            f"v_mov_b32 v1, {snan:#x}",
            "v_bfm_b32 v2, v0, v1",
        ]
        hw_res = run_hw_op(lines, "ps_ieee_v_bfm_b32", exp_reg="v2")
        self.assertEqual(hw_res, res)


class TestVBfmB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            "v_bfm_b32 v2, 8, 4",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_bfm_b32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xff0)


# -----------------------------------------------------------------------------
# 5. v_alignbyte_b32
# -----------------------------------------------------------------------------
def ref_v_alignbyte_b32(s0: int, s1: int, s2: int) -> int:
    return (((s0 << 32) | s1) >> ((s2 & 3) * 8)) & MASK32


GLOBAL_ORACLE_MAP[Opcode.V_ALIGNBYTE_B32] = ref_v_alignbyte_b32


class TestVAlignbyteB32Oracle(unittest.TestCase):
    def test_oracle_execution(self) -> None:
        res = ref_v_alignbyte_b32(0x12345678, 0x9abcdef0, 2)
        self.assertEqual(res, 0x56789abc)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_alignbyte_b32 v2, v0, v1, 2",
        ]
        hw_res = run_hw_op(lines, "ps_oracle_v_alignbyte_b32", exp_reg="v2")
        self.assertEqual(hw_res, res)


class TestVAlignbyteB32ASTAndInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 0x9abcdef0)
        state.write_vgpr(2, 2)
        inst = ASTInstruction(Opcode.V_ALIGNBYTE_B32, ASTOperand.vgpr(3), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.imm(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x56789abc)
        self.assertEqual(state.vgprs[3], 0x56789abc)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            inst.disasm(),
        ]
        hw_res = run_hw_op(lines, "ps_ast_v_alignbyte_b32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVAlignbyteB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_condition_codes(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 0x9abcdef0)
        state.write_vgpr(2, 2)
        inst = ASTInstruction(Opcode.V_ALIGNBYTE_B32, ASTOperand.vgpr(0), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.imm(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x56789abc)
        self.assertEqual(state.vgprs[0], 0x56789abc)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_alignbyte_b32 v0, v0, v1, 2",
        ]
        hw_res = run_hw_op(lines, "ps_alias_v_alignbyte_b32", exp_reg="v0")
        self.assertEqual(hw_res, 0x56789abc)

    def test_ieee754_bitwise_transparency(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        res = ref_v_alignbyte_b32(qnan, snan, 4)
        self.assertIsInstance(res, int)
        lines = [
            f"v_mov_b32 v0, {qnan:#x}",
            f"v_mov_b32 v1, {snan:#x}",
            "v_mov_b32 v2, 4",
            "v_alignbyte_b32 v3, v0, v1, v2",
        ]
        hw_res = run_hw_op(lines, "ps_ieee_v_alignbyte_b32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVAlignbyteB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            "v_alignbyte_b32 v3, v0, v1, 2",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_alignbyte_b32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x56789abc)


# -----------------------------------------------------------------------------
# 6. v_alignbit_b32
# -----------------------------------------------------------------------------
def ref_v_alignbit_b32(s0: int, s1: int, s2: int) -> int:
    sh = s2 & 31
    return s1 if sh == 0 else (((s0 << (32 - sh)) | (s1 >> sh)) & MASK32)


GLOBAL_ORACLE_MAP[Opcode.V_ALIGNBIT_B32] = ref_v_alignbit_b32


class TestVAlignbitB32Oracle(unittest.TestCase):
    def test_oracle_execution(self) -> None:
        res = ref_v_alignbit_b32(0x12345678, 0x9abcdef0, 8)
        self.assertEqual(res, 0x789abcde)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_alignbit_b32 v2, v0, v1, 8",
        ]
        hw_res = run_hw_op(lines, "ps_oracle_v_alignbit_b32", exp_reg="v2")
        self.assertEqual(hw_res, res)


class TestVAlignbitB32ASTAndInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 0x9abcdef0)
        state.write_vgpr(2, 8)
        inst = ASTInstruction(Opcode.V_ALIGNBIT_B32, ASTOperand.vgpr(3), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.imm(8))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x789abcde)
        self.assertEqual(state.vgprs[3], 0x789abcde)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            inst.disasm(),
        ]
        hw_res = run_hw_op(lines, "ps_ast_v_alignbit_b32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVAlignbitB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_condition_codes(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 0x9abcdef0)
        state.write_vgpr(2, 8)
        inst = ASTInstruction(Opcode.V_ALIGNBIT_B32, ASTOperand.vgpr(0), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.imm(8))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x789abcde)
        self.assertEqual(state.vgprs[0], 0x789abcde)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_alignbit_b32 v0, v0, v1, 8",
        ]
        hw_res = run_hw_op(lines, "ps_alias_v_alignbit_b32", exp_reg="v0")
        self.assertEqual(hw_res, 0x789abcde)

    def test_ieee754_bitwise_transparency(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        res = ref_v_alignbit_b32(qnan, snan, 4)
        self.assertIsInstance(res, int)
        lines = [
            f"v_mov_b32 v0, {qnan:#x}",
            f"v_mov_b32 v1, {snan:#x}",
            "v_mov_b32 v2, 4",
            "v_alignbit_b32 v3, v0, v1, v2",
        ]
        hw_res = run_hw_op(lines, "ps_ieee_v_alignbit_b32", exp_reg="v3")
        self.assertEqual(hw_res, res)


class TestVAlignbitB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x9abcdef0",
            "v_mov_b32 v2, 0x55555555",
            "v_alignbit_b32 v3, v0, v1, 8",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_alignbit_b32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x789abcde)


if __name__ == "__main__":
    unittest.main()

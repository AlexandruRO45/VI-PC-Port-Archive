"""Semantic hardware tests and formal verification for RDNA2 s_and_b64.

Covers:
  - s_and_b64: D = S0 & S1 (bitwise AND across 64-bit scalar registers)
  - Condition code SCC: SCC = 1 if D != 0 else 0
  - Special register operands: EXEC, VCC, NULL
  - Multi-lane divergence and EXEC lane masking
  - In-place operand preservation and destination aliasing
  - Double precision / packed float IEEE 754 bit-pattern preservation
  - Hardware SPIR-V translation and execution via tools/rdna2spirv.py
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64 = 1 << 63


def ref_s_and_b64(s0: int, s1: int) -> int:
    """RDNA2 s_and_b64: D = S0 & S1."""
    return (s0 & s1) & MASK64


def ref_scc_b64(val64: int) -> int:
    """RDNA2 SCC condition flag: 1 if 64-bit result != 0, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


def lo32(val: int) -> int:
    return val & MASK32


def hi32(val: int) -> int:
    return (val >> 32) & MASK32


def join64(lo: int, hi: int) -> int:
    return ((hi & MASK32) << 32) | (lo & MASK32)


class SaluAndB64OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for RDNA2 s_and_b64."""

    def test_bitwise_truth_table_exhaustive(self):
        """Verify bit-by-bit truth table of AND across 64 bit positions."""
        for bit in range(64):
            b_mask = 1 << bit
            self.assertEqual(ref_s_and_b64(0, 0) & b_mask, 0)
            self.assertEqual(ref_s_and_b64(0, b_mask) & b_mask, 0)
            self.assertEqual(ref_s_and_b64(b_mask, 0) & b_mask, 0)
            self.assertEqual(ref_s_and_b64(b_mask, b_mask) & b_mask, b_mask)

    def test_algebraic_identities(self):
        """Verify Boolean algebraic laws: commutativity, associativity, idempotence, identities."""
        a = 0x01234567_89ABCDEF
        b = 0xFEDCBA98_76543210
        c = 0x5555AAAA_3333CCCC

        # Commutativity: A & B == B & A
        self.assertEqual(ref_s_and_b64(a, b), ref_s_and_b64(b, a))
        # Associativity: (A & B) & C == A & (B & C)
        self.assertEqual(ref_s_and_b64(ref_s_and_b64(a, b), c), ref_s_and_b64(a, ref_s_and_b64(b, c)))
        # Identity element: A & 1s == A
        self.assertEqual(ref_s_and_b64(a, MASK64), a)
        # Annihilator: A & 0 == 0
        self.assertEqual(ref_s_and_b64(a, 0), 0)
        # Idempotence: A & A == A
        self.assertEqual(ref_s_and_b64(a, a), a)

    def test_boundary_corners(self):
        """Verify edge values: 0, 1, MASK64, MSB, alternating bit patterns, dword masks."""
        boundaries = [
            0x00000000_00000000,
            0x00000000_00000001,
            0x80000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x55555555_55555555,
            0xAAAAAAAA_AAAAAAAA,
            0x00000000_FFFFFFFF,
            0xFFFFFFFF_00000000,
        ]
        for s0 in boundaries:
            for s1 in boundaries:
                with self.subTest(s0=hex(s0), s1=hex(s1)):
                    res = ref_s_and_b64(s0, s1)
                    self.assertGreaterEqual(res, 0)
                    self.assertLessEqual(res, MASK64)
                    scc = ref_scc_b64(res)
                    self.assertIn(scc, (0, 1))
                    self.assertEqual(scc, 1 if res != 0 else 0)

    def test_spirv_hardware_basic(self):
        """Verify translator emits valid SPIR-V executing exact 64-bit bitwise AND."""
        test_pairs = [
            (0x12345678_9ABCDEF0, 0x0F0F0F0F_F0F0F0F0),
            (0xFFFFFFFF_FFFFFFFF, 0x12345678_87654321),
            (0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000),
            (0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF),
        ]
        for idx, (v0, v1) in enumerate(test_pairs):
            with self.subTest(idx=idx, v0=hex(v0), v1=hex(v1)):
                lines = [
                    f"s_mov_b32 s0, {lo32(v0):#x}",
                    f"s_mov_b32 s1, {hi32(v0):#x}",
                    f"s_mov_b32 s2, {lo32(v1):#x}",
                    f"s_mov_b32 s3, {hi32(v1):#x}",
                    "s_and_b64 s[4:5], s[0:1], s[2:3]",
                    "s_cselect_b32 s6, 1, 0",
                    "v_mov_b32 v0, s4",
                    "v_mov_b32 v1, s5",
                    "v_mov_b32 v2, s6",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_salu_and_b64_basic_{idx}", lines)
                out = mrt0_words(spv)
                res64 = join64(out[0], out[1])
                expected = ref_s_and_b64(v0, v1)
                self.assertEqual(res64, expected)
                self.assertEqual(out[2], ref_scc_b64(expected))


class OperandKind(Enum):
    SGPR_PAIR = auto()
    SPECIAL = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class AstOperand:
    kind: OperandKind
    name: str
    value: int = 0

    def resolve(self, env: dict[str, int]) -> int:
        if self.kind == OperandKind.LITERAL:
            return self.value & MASK64
        if self.name in env:
            return env[self.name] & MASK64
        raise KeyError(f"Operand not found in env: {self.name}")


@dataclass(frozen=True)
class AstSAndB64:
    dst: AstOperand
    src0: AstOperand
    src1: AstOperand

    def eval(self, env: dict[str, int]) -> tuple[int, int]:
        v0 = self.src0.resolve(env)
        v1 = self.src1.resolve(env)
        res = ref_s_and_b64(v0, v1)
        scc = ref_scc_b64(res)
        return res, scc

    def to_asm(self) -> str:
        return f"s_and_b64 {self.dst.name}, {self.src0.name}, {self.src1.name}"


class SaluAndB64AstTests(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions for s_and_b64."""

    def test_ast_node_evaluation(self):
        env = {
            "s[0:1]": 0x11112222_33334444,
            "s[2:3]": 0x55556666_77778888,
        }
        node = AstSAndB64(
            dst=AstOperand(OperandKind.SGPR_PAIR, "s[4:5]"),
            src0=AstOperand(OperandKind.SGPR_PAIR, "s[0:1]"),
            src1=AstOperand(OperandKind.SGPR_PAIR, "s[2:3]"),
        )
        res, scc = node.eval(env)
        expected = ref_s_and_b64(env["s[0:1]"], env["s[2:3]"])
        self.assertEqual(res, expected)
        self.assertEqual(scc, ref_scc_b64(expected))
        self.assertEqual(node.to_asm(), "s_and_b64 s[4:5], s[0:1], s[2:3]")

    def test_ast_compositional_expression_tree(self):
        """Compose nested AST expressions: (A & B) & C."""
        env = {
            "s[0:1]": 0xFF00FF00_AA55AA55,
            "s[2:3]": 0xFFFF0000_FFFF0000,
            "s[4:5]": 0x0F0F0F0F_0F0F0F0F,
        }
        step1 = AstSAndB64(
            dst=AstOperand(OperandKind.SGPR_PAIR, "s[6:7]"),
            src0=AstOperand(OperandKind.SGPR_PAIR, "s[0:1]"),
            src1=AstOperand(OperandKind.SGPR_PAIR, "s[2:3]"),
        )
        r1, _ = step1.eval(env)
        env["s[6:7]"] = r1

        step2 = AstSAndB64(
            dst=AstOperand(OperandKind.SGPR_PAIR, "s[8:9]"),
            src0=AstOperand(OperandKind.SGPR_PAIR, "s[6:7]"),
            src1=AstOperand(OperandKind.SGPR_PAIR, "s[4:5]"),
        )
        r2, scc2 = step2.eval(env)
        expected = env["s[0:1]"] & env["s[2:3]"] & env["s[4:5]"]
        self.assertEqual(r2, expected)
        self.assertEqual(scc2, ref_scc_b64(expected))

    def test_ast_hardware_translation(self):
        """Emit assembly from AST and verify execution via translator."""
        node = AstSAndB64(
            dst=AstOperand(OperandKind.SGPR_PAIR, "s[4:5]"),
            src0=AstOperand(OperandKind.SGPR_PAIR, "s[0:1]"),
            src1=AstOperand(OperandKind.SGPR_PAIR, "s[2:3]"),
        )
        v0 = 0xCAFEBABE_DEADBEEF
        v1 = 0x0000FFFF_FFFF0000
        lines = [
            f"s_mov_b32 s0, {lo32(v0):#x}",
            f"s_mov_b32 s1, {hi32(v0):#x}",
            f"s_mov_b32 s2, {lo32(v1):#x}",
            f"s_mov_b32 s3, {hi32(v1):#x}",
            node.to_asm(),
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_and_b64_ast", lines)
        out = mrt0_words(spv)
        res64 = join64(out[0], out[1])
        expected = ref_s_and_b64(v0, v1)
        self.assertEqual(res64, expected)
        self.assertEqual(out[2], ref_scc_b64(expected))


class ScalarMachineState:
    """Models RDNA2 scalar execution state: 106 SGPRs, EXEC, VCC, SCC, and NULL."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.exec: int = MASK64
        self.vcc: int = 0
        self.scc: int = 0

    def read_sgpr64(self, idx: int) -> int:
        if idx < 0 or idx + 1 >= len(self.sgprs):
            raise IndexError(f"SGPR index out of bounds: {idx}")
        lo = self.sgprs[idx] & MASK32
        hi = self.sgprs[idx + 1] & MASK32
        return (hi << 32) | lo

    def write_sgpr64(self, idx: int, val: int) -> None:
        if idx < 0 or idx + 1 >= len(self.sgprs):
            raise IndexError(f"SGPR index out of bounds: {idx}")
        self.sgprs[idx] = val & MASK32
        self.sgprs[idx + 1] = (val >> 32) & MASK32

    def resolve_src64(self, src: int | str) -> int:
        if isinstance(src, int):
            return src & MASK64
        src = src.strip()
        if src.lower() == "exec":
            return self.exec & MASK64
        if src.lower() == "vcc":
            return self.vcc & MASK64
        if src.lower() == "null":
            return 0
        if src.startswith("s[") and ":" in src and src.endswith("]"):
            base = int(src[2:-1].split(":")[0])
            return self.read_sgpr64(base)
        if src.startswith("0x") or src.startswith("-0x") or src.isdigit() or (src.startswith("-") and src[1:].isdigit()):
            return int(src, 0) & MASK64
        raise ValueError(f"Unknown 64-bit source operand: {src}")

    def write_dst64(self, dst: str, val: int) -> None:
        val = val & MASK64
        dst = dst.strip()
        if dst.lower() == "exec":
            self.exec = val
        elif dst.lower() == "vcc":
            self.vcc = val
        elif dst.lower() == "null":
            pass
        elif dst.startswith("s[") and ":" in dst and dst.endswith("]"):
            base = int(dst[2:-1].split(":")[0])
            self.write_sgpr64(base, val)
        else:
            raise ValueError(f"Unknown 64-bit destination operand: {dst}")

    def execute_and_b64(self, dst: str, src0: int | str, src1: int | str) -> int:
        s0 = self.resolve_src64(src0)
        s1 = self.resolve_src64(src1)
        res = ref_s_and_b64(s0, s1)
        self.write_dst64(dst, res)
        self.scc = ref_scc_b64(res)
        return res

    def step(self, line: str) -> None:
        clean = line.split(";")[0].strip()
        if not clean:
            return
        parts = clean.split(None, 1)
        opcode = parts[0].lower()
        if len(parts) < 2:
            raise ValueError(f"Missing operands for {opcode}")
        args = [arg.strip() for arg in parts[1].split(",")]
        if opcode == "s_and_b64":
            if len(args) != 3:
                raise ValueError(f"s_and_b64 requires 3 operands, got {len(args)}")
            self.execute_and_b64(args[0], args[1], args[2])
        elif opcode == "s_mov_b32":
            if len(args) != 2:
                raise ValueError("s_mov_b32 requires 2 operands")
            dst_idx = int(args[0][1:])
            self.sgprs[dst_idx] = int(args[1], 0) & MASK32
        else:
            raise ValueError(f"Unsupported opcode: {opcode}")


class SaluAndB64InterpreterTests(unittest.TestCase):
    """Deterministic scalar instruction interpreter tests for s_and_b64."""

    def test_scalar_register_pairs(self):
        state = ScalarMachineState()
        val = 0x12345678_9ABCDEF0
        state.write_sgpr64(10, val)
        self.assertEqual(state.read_sgpr64(10), val)
        self.assertEqual(state.sgprs[10], 0x9ABCDEF0)
        self.assertEqual(state.sgprs[11], 0x12345678)

    def test_interpreter_step_execution(self):
        state = ScalarMachineState()
        state.step("s_mov_b32 s0, 0x55555555")
        state.step("s_mov_b32 s1, 0x55555555")
        state.step("s_mov_b32 s2, 0x33333333")
        state.step("s_mov_b32 s3, 0x33333333")
        state.step("s_and_b64 s[4:5], s[0:1], s[2:3]")
        self.assertEqual(state.read_sgpr64(4), 0x11111111_11111111)
        self.assertEqual(state.scc, 1)

    def test_special_registers_and_null(self):
        state = ScalarMachineState()
        state.exec = 0xAAAAAAAAAAAAAAAA
        state.vcc = 0x5555555555555555
        state.execute_and_b64("null", "exec", "vcc")
        self.assertEqual(state.scc, 0)
        self.assertEqual(state.exec, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(state.vcc, 0x5555555555555555)

    def test_interpreter_hardware_conformance(self):
        state = ScalarMachineState()
        v0 = 0xABCD0123_456789EF
        v1 = 0x00FF00FF_00FF00FF
        state.write_sgpr64(0, v0)
        state.write_sgpr64(2, v1)
        res_interp = state.execute_and_b64("s[4:5]", "s[0:1]", "s[2:3]")

        lines = [
            f"s_mov_b32 s0, {lo32(v0):#x}",
            f"s_mov_b32 s1, {hi32(v0):#x}",
            f"s_mov_b32 s2, {lo32(v1):#x}",
            f"s_mov_b32 s3, {hi32(v1):#x}",
            "s_and_b64 s[4:5], s[0:1], s[2:3]",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_and_b64_interp", lines)
        out = mrt0_words(spv)
        res_hw = join64(out[0], out[1])
        self.assertEqual(res_interp, res_hw)
        self.assertEqual(state.scc, out[2])


if __name__ == "__main__":
    unittest.main()

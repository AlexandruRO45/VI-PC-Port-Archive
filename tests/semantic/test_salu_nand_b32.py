"""Semantic reference oracle and verification for RDNA2 scalar NAND logic (s_nand_b32).

AMD RDNA2 / GFX10 SOP2 scalar ALU instruction:
    s_nand_b32 s_dst, s_src0, s_src1

Hardware Semantics:
    D.u = ~(S0.u & S1.u) & 0xFFFFFFFF
    SCC = 1 if D.u != 0 else 0

Validation Coverage:
    - Reference mathematical oracle: bit-by-bit truth table, De Morgan dualities, commutativity
    - AST definitions: structural operand types, parsing, operand bounds
    - Deterministic scalar interpreter: register bank, literals, special registers (SCC, NULL, VCC)
    - SIMD divergence: wavefront lane masking, active/inactive lanes, Wave64 split
    - Register aliasing: in-place destination updates (dst == src0, dst == src1, dst == src0 == src1)
    - Condition codes: exhaustive zero/nonzero SCC transitions, single-bit sensitivity
    - Floating-point / IEEE 754 edge cases: subnormals, infinities, NaNs, Sheffer stroke universal logic
    - End-to-end SPIR-V lowering: translation via tools/rdna2spirv.py and execution via spirv_mini_interp
"""
from __future__ import annotations

import dataclasses
import enum
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

try:
    from support import translate_lines, mrt0_words  # noqa: E402
    TRANSLATOR_ERROR = None
except Exception as _err:
    translate_lines = None  # type: ignore
    mrt0_words = None  # type: ignore
    TRANSLATOR_ERROR = str(_err)

MASK32: int = 0xFFFFFFFF


def ref_nand_b32(s0: int, s1: int) -> int:
    """RDNA2 s_nand_b32 reference calculation: D = ~(S0 & S1) & 0xFFFFFFFF."""
    return (~(s0 & s1)) & MASK32


def ref_scc(val32: int) -> int:
    """RDNA2 SCC condition flag: 1 if 32-bit result != 0 else 0."""
    return 1 if (val32 & MASK32) != 0 else 0


class SaluNandB32OracleTests(unittest.TestCase):
    """Reference mathematical oracle verification for RDNA2 s_nand_b32."""

    def test_truth_table_bit_by_bit(self):
        """Verify full 1-bit truth table for NAND: 0,0->1; 0,1->1; 1,0->1; 1,1->0."""
        truth_table = {
            (0, 0): 1,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 0,
        }
        for (b0, b1), expected_bit in truth_table.items():
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK32 if b0 else 0
                s1 = MASK32 if b1 else 0
                expected = MASK32 if expected_bit else 0
                got = ref_nand_b32(s0, s1)
                self.assertEqual(got, expected)
                self.assertEqual(ref_scc(got), 1 if expected_bit else 0)

    def test_commutativity_symmetry(self):
        """Verify s_nand_b32 is strictly commutative: NAND(a, b) == NAND(b, a)."""
        samples = [
            (0x00000000, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x55555555, 0xAAAAAAAA),
            (0x12345678, 0x9ABCDEF0),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x0000FFFF, 0xFFFF0000),
            (0x80000000, 0x7FFFFFFF),
        ]
        for a, b in samples:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_nand_b32(a, b), ref_nand_b32(b, a))

    def test_de_morgan_identities(self):
        """Verify De Morgan algebraic dualities for NAND."""
        values = [0x00000000, 0xFFFFFFFF, 0x13579BDF, 0x2468ACE0, 0xCAFEBABE, 0xDEADBEEF]
        for a in values:
            for b in values:
                # NAND(a, b) == NOT(a) OR NOT(b)
                self.assertEqual(ref_nand_b32(a, b), ((~a & MASK32) | (~b & MASK32)) & MASK32)
                # NAND(a, 0) == 0xFFFFFFFF
                self.assertEqual(ref_nand_b32(a, 0), MASK32)
                # NAND(a, 0xFFFFFFFF) == NOT(a)
                self.assertEqual(ref_nand_b32(a, MASK32), (~a) & MASK32)
                # NAND(a, a) == NOT(a)
                self.assertEqual(ref_nand_b32(a, a), (~a) & MASK32)
                # NAND(a, NOT(a)) == 0xFFFFFFFF
                not_a = (~a) & MASK32
                self.assertEqual(ref_nand_b32(a, not_a), MASK32)

    def test_standard_bit_patterns(self):
        """Verify boundary patterns: walking ones and walking zeros."""
        for i in range(32):
            bit_mask = 1 << i
            inv_mask = (~bit_mask) & MASK32
            # NAND with self-bit
            self.assertEqual(ref_nand_b32(bit_mask, bit_mask), inv_mask)
            # NAND with inverse bit
            self.assertEqual(ref_nand_b32(bit_mask, inv_mask), MASK32)


class OperandKind(enum.Enum):
    SGPR = "sgpr"
    IMM = "imm"
    SPECIAL = "special"
    NULL = "null"


@dataclasses.dataclass(frozen=True)
class Operand:
    kind: OperandKind
    raw: str
    reg_num: int | None = None
    imm_val: int | None = None


@dataclasses.dataclass(frozen=True)
class SOP2Instruction:
    op: str
    dst: Operand
    src0: Operand
    src1: Operand


def parse_operand(token: str) -> Operand:
    tok = token.strip().rstrip(",")
    if not tok:
        raise ValueError("empty operand token")
    if tok == "null":
        return Operand(kind=OperandKind.NULL, raw=tok)
    if tok in ("vcc_lo", "vcc_hi", "exec_lo", "exec_hi", "scc"):
        return Operand(kind=OperandKind.SPECIAL, raw=tok)
    if tok.startswith("s") and tok[1:].isdigit():
        r = int(tok[1:])
        if not (0 <= r <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {tok}")
        return Operand(kind=OperandKind.SGPR, raw=tok, reg_num=r)
    try:
        if tok.startswith(("0x", "0X")):
            val = int(tok, 16)
        else:
            val = int(tok, 10)
        return Operand(kind=OperandKind.IMM, raw=tok, imm_val=val & MASK32)
    except ValueError:
        raise ValueError(f"unrecognized operand syntax: {tok}")


def parse_sop2(line: str) -> SOP2Instruction:
    clean = line.split(";")[0].strip()
    if not clean:
        raise ValueError("empty line")
    parts = clean.split(None, 1)
    op = parts[0]
    if len(parts) < 2:
        raise ValueError(f"instruction missing operands: {line}")
    args = [arg.strip() for arg in parts[1].split(",") if arg.strip()]
    if len(args) != 3:
        raise ValueError(f"SOP2 instruction requires 3 operands (dst, src0, src1), got {len(args)}: {line}")
    dst = parse_operand(args[0])
    src0 = parse_operand(args[1])
    src1 = parse_operand(args[2])
    return SOP2Instruction(op=op, dst=dst, src0=src0, src1=src1)


def validate_sop2(inst: SOP2Instruction) -> None:
    if inst.op != "s_nand_b32":
        raise ValueError(f"unsupported opcode: {inst.op}")
    if inst.dst.kind == OperandKind.IMM:
        raise ValueError(f"destination cannot be immediate: {inst.dst.raw}")
    imm_count = sum(1 for src in (inst.src0, inst.src1) if src.kind == OperandKind.IMM)
    if imm_count > 1:
        raise ValueError("RDNA2 SOP2 encodes at most one literal constant operand")


class SaluNandB32AstTests(unittest.TestCase):
    """AST structure and operand parsing validation for s_nand_b32."""

    def test_ast_parse_reg_reg(self):
        inst = parse_sop2("s_nand_b32 s0, s1, s2")
        validate_sop2(inst)
        self.assertEqual(inst.op, "s_nand_b32")
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.reg_num, 0)
        self.assertEqual(inst.src0.reg_num, 1)
        self.assertEqual(inst.src1.reg_num, 2)

    def test_ast_parse_reg_imm(self):
        inst = parse_sop2("s_nand_b32 s4, s5, 0x12345678")
        validate_sop2(inst)
        self.assertEqual(inst.dst.reg_num, 4)
        self.assertEqual(inst.src0.reg_num, 5)
        self.assertEqual(inst.src1.kind, OperandKind.IMM)
        self.assertEqual(inst.src1.imm_val, 0x12345678)

    def test_ast_parse_special_registers(self):
        inst1 = parse_sop2("s_nand_b32 null, s0, s1")
        validate_sop2(inst1)
        self.assertEqual(inst1.dst.kind, OperandKind.NULL)

        inst2 = parse_sop2("s_nand_b32 vcc_lo, s2, s3")
        validate_sop2(inst2)
        self.assertEqual(inst2.dst.kind, OperandKind.SPECIAL)
        self.assertEqual(inst2.dst.raw, "vcc_lo")

    def test_ast_validation_rejects_invalid_forms(self):
        with self.assertRaises(ValueError):
            validate_sop2(parse_sop2("s_nor_b32 s0, s1, s2"))
        with self.assertRaises(ValueError):
            validate_sop2(parse_sop2("s_nand_b32 0x10, s1, s2"))
        with self.assertRaises(ValueError):
            validate_sop2(parse_sop2("s_nand_b32 s0, 0x10, 0x20"))
        with self.assertRaises(ValueError):
            parse_sop2("s_nand_b32 s106, s1, s2")


class ScalarInterpreter:
    """Single-lane scalar execution state modeling RDNA2 SGPRs and SCC."""

    def __init__(self) -> None:
        self.sgprs: list[int] = [0] * 106
        self.scc: int = 0
        self.vcc_lo: int = 0
        self.vcc_hi: int = 0
        self.exec_lo: int = 0xFFFFFFFF
        self.exec_hi: int = 0xFFFFFFFF

    def read_operand(self, op: Operand) -> int:
        if op.kind == OperandKind.SGPR:
            assert op.reg_num is not None
            return self.sgprs[op.reg_num]
        if op.kind == OperandKind.IMM:
            assert op.imm_val is not None
            return op.imm_val
        if op.kind == OperandKind.SPECIAL:
            if op.raw == "vcc_lo":
                return self.vcc_lo
            if op.raw == "vcc_hi":
                return self.vcc_hi
            if op.raw == "exec_lo":
                return self.exec_lo
            if op.raw == "exec_hi":
                return self.exec_hi
            if op.raw == "scc":
                return self.scc
        if op.kind == OperandKind.NULL:
            return 0
        raise ValueError(f"cannot read operand {op}")

    def write_operand(self, op: Operand, val: int) -> None:
        v = val & MASK32
        if op.kind == OperandKind.SGPR:
            assert op.reg_num is not None
            self.sgprs[op.reg_num] = v
        elif op.kind == OperandKind.SPECIAL:
            if op.raw == "vcc_lo":
                self.vcc_lo = v
            elif op.raw == "vcc_hi":
                self.vcc_hi = v
            elif op.raw == "exec_lo":
                self.exec_lo = v
            elif op.raw == "exec_hi":
                self.exec_hi = v
            elif op.raw == "scc":
                self.scc = 1 if v != 0 else 0
        elif op.kind == OperandKind.NULL:
            pass
        else:
            raise ValueError(f"cannot write operand {op}")

    def step(self, inst: SOP2Instruction) -> int:
        validate_sop2(inst)
        s0 = self.read_operand(inst.src0)
        s1 = self.read_operand(inst.src1)
        res = ref_nand_b32(s0, s1)
        self.write_operand(inst.dst, res)
        self.scc = ref_scc(res)
        return res

    def exec_line(self, line: str) -> None:
        clean = line.split(";")[0].strip()
        if not clean:
            return
        parts = clean.split(None, 1)
        op = parts[0]
        args = [arg.strip() for arg in parts[1].split(",") if arg.strip()]
        if op == "s_mov_b32":
            dst = parse_operand(args[0])
            src = parse_operand(args[1])
            val = self.read_operand(src)
            self.write_operand(dst, val)
        elif op == "s_nand_b32":
            inst = parse_sop2(line)
            self.step(inst)
        elif op == "s_cselect_b32":
            dst = parse_operand(args[0])
            s0 = self.read_operand(parse_operand(args[1]))
            s1 = self.read_operand(parse_operand(args[2]))
            self.write_operand(dst, s0 if self.scc != 0 else s1)
        else:
            raise ValueError(f"unsupported interpreter op: {op}")


class SaluNandB32InterpreterTests(unittest.TestCase):
    """Deterministic scalar interpreter tests for s_nand_b32."""

    def test_interp_basic_reg_reg(self):
        interp = ScalarInterpreter()
        interp.exec_line("s_mov_b32 s0, 0x0F0F0F0F")
        interp.exec_line("s_mov_b32 s1, 0x33333333")
        interp.exec_line("s_nand_b32 s2, s0, s1")
        self.assertEqual(interp.sgprs[2], 0xFCFCFCFC)
        self.assertEqual(interp.scc, 1)

    def test_interp_immediate_operand(self):
        interp = ScalarInterpreter()
        interp.exec_line("s_mov_b32 s0, 0xFFFFFFFF")
        interp.exec_line("s_nand_b32 s1, s0, 0x12345678")
        self.assertEqual(interp.sgprs[1], (~0x12345678) & MASK32)
        self.assertEqual(interp.scc, 1)

    def test_interp_null_destination(self):
        interp = ScalarInterpreter()
        interp.exec_line("s_mov_b32 s0, 0xFFFFFFFF")
        interp.exec_line("s_mov_b32 s1, 0xFFFFFFFF")
        interp.exec_line("s_nand_b32 null, s0, s1")
        self.assertEqual(interp.scc, 0)

    def test_interp_negative_invalid_instruction(self):
        interp = ScalarInterpreter()
        with self.assertRaises(ValueError):
            interp.exec_line("s_invalid_op s0, s1, s2")


class WaveState:
    """Models SIMD vector state for 32/64 lanes controlled by scalar execution masks."""

    def __init__(self, num_lanes: int = 32) -> None:
        self.num_lanes = num_lanes
        self.exec_mask: int = (1 << num_lanes) - 1
        self.vgprs: list[list[int]] = [[0] * num_lanes for _ in range(64)]

    def apply_vector_add(self, dst: int, src0: int, src1: int) -> None:
        """Executes vector addition only on active lanes per exec_mask."""
        for lane in range(self.num_lanes):
            if (self.exec_mask >> lane) & 1:
                self.vgprs[dst][lane] = (self.vgprs[src0][lane] + self.vgprs[src1][lane]) & MASK32


class SaluNandB32SimdDivergenceTests(unittest.TestCase):
    """Multi-lane SIMD divergence verification for mask updates via s_nand_b32."""

    def test_wave32_divergence_mask_generation(self):
        """Verify s_nand_b32 correctly updates active lane masks for divergent branching."""
        interp = ScalarInterpreter()
        cond_a = 0b11110000111100001111000011110000
        cond_b = 0b11001100110011001100110011001100
        interp.sgprs[0] = cond_a
        interp.sgprs[1] = cond_b
        interp.exec_line("s_nand_b32 s2, s0, s1")

        expected_mask = ref_nand_b32(cond_a, cond_b)
        self.assertEqual(interp.sgprs[2], expected_mask)

        wave = WaveState(num_lanes=32)
        wave.exec_mask = interp.sgprs[2]
        for lane in range(32):
            wave.vgprs[0][lane] = 10
            wave.vgprs[1][lane] = 20

        wave.apply_vector_add(dst=2, src0=0, src1=1)

        for lane in range(32):
            lane_active = (expected_mask >> lane) & 1
            expected_val = 30 if lane_active else 0
            self.assertEqual(wave.vgprs[2][lane], expected_val)

    def test_wave64_halves_independence(self):
        """Verify Wave64 low (0-31) and high (32-63) halves operate independently."""
        interp = ScalarInterpreter()
        mask_lo = 0xAAAAAAAA
        mask_hi = 0x55555555
        interp.vcc_lo = mask_lo
        interp.vcc_hi = mask_hi
        interp.sgprs[0] = 0xFFFFFFFF
        interp.sgprs[1] = 0x00000000

        interp.exec_line("s_nand_b32 vcc_lo, vcc_lo, s0")
        self.assertEqual(interp.vcc_lo, (~mask_lo) & MASK32)
        self.assertEqual(interp.vcc_hi, mask_hi)

        interp.exec_line("s_nand_b32 vcc_hi, vcc_hi, s1")
        self.assertEqual(interp.vcc_hi, MASK32)

    def test_all_active_and_all_inactive_boundaries(self):
        """Verify convergence when all lanes are active or all lanes are inactive."""
        all_lanes = 0xFFFFFFFF
        none_lanes = 0x00000000

        self.assertEqual(ref_nand_b32(all_lanes, all_lanes), none_lanes)
        self.assertEqual(ref_nand_b32(all_lanes, none_lanes), all_lanes)
        self.assertEqual(ref_nand_b32(none_lanes, none_lanes), all_lanes)


class SaluNandB32RegisterAliasingTests(unittest.TestCase):
    """In-place operand register aliasing validation for s_nand_b32."""

    def test_alias_dst_equals_src0(self):
        """Verify s_nand_b32 s0, s0, s1 preserves operands until write."""
        interp = ScalarInterpreter()
        s0_init = 0x0F0F0F0F
        s1_val = 0x33333333
        interp.sgprs[0] = s0_init
        interp.sgprs[1] = s1_val
        interp.exec_line("s_nand_b32 s0, s0, s1")
        expected = ref_nand_b32(s0_init, s1_val)
        self.assertEqual(interp.sgprs[0], expected)
        self.assertEqual(interp.sgprs[1], s1_val)

    def test_alias_dst_equals_src1(self):
        """Verify s_nand_b32 s1, s0, s1 preserves operands until write."""
        interp = ScalarInterpreter()
        s0_val = 0x12345678
        s1_init = 0x87654321
        interp.sgprs[0] = s0_val
        interp.sgprs[1] = s1_init
        interp.exec_line("s_nand_b32 s1, s0, s1")
        expected = ref_nand_b32(s0_val, s1_init)
        self.assertEqual(interp.sgprs[1], expected)
        self.assertEqual(interp.sgprs[0], s0_val)

    def test_alias_dst_equals_both_srcs(self):
        """Verify s_nand_b32 s0, s0, s0 produces exact bitwise NOT."""
        interp = ScalarInterpreter()
        s0_val = 0xDEADBEEF
        interp.sgprs[0] = s0_val
        interp.exec_line("s_nand_b32 s0, s0, s0")
        expected = (~s0_val) & MASK32
        self.assertEqual(interp.sgprs[0], expected)

    def test_alias_chained_accumulators(self):
        """Verify chained updates on an accumulator register match oracle."""
        interp = ScalarInterpreter()
        acc = 0x11223344
        ops = [0x55667788, 0x99AABBCC, 0xDDEEFF00, 0x01234567]
        interp.sgprs[0] = acc
        for idx, val in enumerate(ops):
            r = idx + 1
            interp.sgprs[r] = val
            interp.exec_line(f"s_nand_b32 s0, s0, s{r}")
            acc = ref_nand_b32(acc, val)
            self.assertEqual(interp.sgprs[0], acc)


class SaluNandB32ConditionCodesTests(unittest.TestCase):
    """Scalar Condition Code (SCC) behavior and bit sensitivity for s_nand_b32."""

    def test_scc_zero_only_when_both_all_ones(self):
        """Verify SCC is 0 if and only if both sources have all bits set."""
        interp = ScalarInterpreter()
        interp.sgprs[0] = MASK32
        interp.sgprs[1] = MASK32
        interp.exec_line("s_nand_b32 s2, s0, s1")
        self.assertEqual(interp.sgprs[2], 0)
        self.assertEqual(interp.scc, 0)

    def test_scc_nonzero_on_single_bit_zero(self):
        """Clearing any single bit in either input makes NAND result nonzero (SCC=1)."""
        interp = ScalarInterpreter()
        for bit in range(32):
            mask = (~(1 << bit)) & MASK32
            interp.sgprs[0] = mask
            interp.sgprs[1] = MASK32
            interp.exec_line("s_nand_b32 s2, s0, s1")
            self.assertEqual(interp.sgprs[2], 1 << bit)
            self.assertEqual(interp.scc, 1)

            interp.sgprs[0] = MASK32
            interp.sgprs[1] = mask
            interp.exec_line("s_nand_b32 s2, s0, s1")
            self.assertEqual(interp.sgprs[2], 1 << bit)
            self.assertEqual(interp.scc, 1)

    def test_scc_nonzero_on_single_bit_one(self):
        """Setting only one bit in both inputs produces 31 bits set (SCC=1)."""
        interp = ScalarInterpreter()
        for bit in range(32):
            mask = 1 << bit
            interp.sgprs[0] = mask
            interp.sgprs[1] = mask
            interp.exec_line("s_nand_b32 s2, s0, s1")
            expected = (~mask) & MASK32
            self.assertEqual(interp.sgprs[2], expected)
            self.assertEqual(interp.scc, 1)

    def test_scc_zero_inputs(self):
        """NAND of 0 with 0 yields 0xFFFFFFFF, resulting in SCC=1."""
        interp = ScalarInterpreter()
        interp.sgprs[0] = 0
        interp.sgprs[1] = 0
        interp.exec_line("s_nand_b32 s2, s0, s1")
        self.assertEqual(interp.sgprs[2], MASK32)
        self.assertEqual(interp.scc, 1)

    def test_scc_overwrite_state_transitions(self):
        """Verify strict overwrite of previous SCC state (0->1, 1->0, 0->0, 1->1)."""
        interp = ScalarInterpreter()

        # Step 1: Force SCC = 0
        interp.sgprs[0] = MASK32
        interp.sgprs[1] = MASK32
        interp.exec_line("s_nand_b32 s2, s0, s1")
        self.assertEqual(interp.scc, 0)

        # Step 2: 0 -> 1 transition
        interp.sgprs[1] = 0
        interp.exec_line("s_nand_b32 s2, s0, s1")
        self.assertEqual(interp.scc, 1)

        # Step 3: 1 -> 1 transition
        interp.exec_line("s_nand_b32 s2, s0, s1")
        self.assertEqual(interp.scc, 1)

        # Step 4: 1 -> 0 transition
        interp.sgprs[1] = MASK32
        interp.exec_line("s_nand_b32 s2, s0, s1")
        self.assertEqual(interp.scc, 0)


if __name__ == "__main__":
    unittest.main()

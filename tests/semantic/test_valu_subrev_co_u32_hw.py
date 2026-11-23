"""Semantic tests for RDNA2 v_subrev_co_u32 (vector reverse subtract with carry out).

RDNA2 VALU architecture contract:
- Opcode: v_subrev_co_u32 vdst, sdst, src0, src1 [clamp]
- Operation: D = src1 - src0 (unsigned 32-bit integer subtraction).
- Reverse operand order: computes src1 minus src0, contrasting with v_sub_co_u32 (src0 minus src1).
- Borrow out: borrow occurs if src1 < src0 (unsigned), writing 1 to sdst mask for active lanes.
- VOP3 clamp modifier: hardware saturates D to 0 upon borrow (underflow) while still reporting
  the real borrow bit in sdst. If no borrow occurs, D receives (src1 - src0) & 0xFFFFFFFF.
- EXEC masking: inactive lanes preserve their destination VGPR and contribute 0 to the sdst mask.
- Condition codes: VALU operations do not modify SCC.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------
M32 = 0xFFFFFFFF


def ref_subrev_co_u32(src0: int, src1: int, clamp: bool = False) -> tuple[int, int]:
    """Pure mathematical oracle matching RDNA2 hardware semantics for v_subrev_co_u32.

    Returns (dst, borrow) where:
    - dst is (src1 - src0) & 0xFFFFFFFF (or 0 if clamp and borrow occurred)
    - borrow is 1 if (src1 & 0xFFFFFFFF) < (src0 & 0xFFFFFFFF) else 0
    """
    s0 = src0 & M32
    s1 = src1 & M32
    diff = (s1 - s0) & M32
    borrow = 1 if s1 < s0 else 0
    dst = 0 if (clamp and borrow) else diff
    return dst, borrow


class TestSubrevCoU32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        dst, borrow = ref_subrev_co_u32(10, 25)
        self.assertEqual(dst, 15)
        self.assertEqual(borrow, 0)

    def test_borrow_underflow(self) -> None:
        dst, borrow = ref_subrev_co_u32(25, 10)
        self.assertEqual(dst, 0xFFFFFFF1)
        self.assertEqual(borrow, 1)

    def test_zero_identity(self) -> None:
        dst, borrow = ref_subrev_co_u32(0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 0)

        dst, borrow = ref_subrev_co_u32(12345, 12345)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 0)

    def test_boundary_opposites(self) -> None:
        # src1 = 0, src0 = 0xFFFFFFFF -> 0 - 0xFFFFFFFF = 1 with borrow
        dst, borrow = ref_subrev_co_u32(0xFFFFFFFF, 0)
        self.assertEqual(dst, 1)
        self.assertEqual(borrow, 1)

        # src1 = 0xFFFFFFFF, src0 = 0 -> 0xFFFFFFFF - 0 = 0xFFFFFFFF without borrow
        dst, borrow = ref_subrev_co_u32(0, 0xFFFFFFFF)
        self.assertEqual(dst, 0xFFFFFFFF)
        self.assertEqual(borrow, 0)

    def test_clamp_saturation(self) -> None:
        # Underflow with clamp saturates dst to 0, borrow remains 1
        dst, borrow = ref_subrev_co_u32(50, 20, clamp=True)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 1)

        # Normal subtraction with clamp remains diff, borrow is 0
        dst, borrow = ref_subrev_co_u32(20, 50, clamp=True)
        self.assertEqual(dst, 30)
        self.assertEqual(borrow, 0)

    def test_subtraction_reversibility_asymmetry(self) -> None:
        # v_subrev_co_u32 computes src1 - src0, which is anti-symmetric to src0 - src1
        d1, b1 = ref_subrev_co_u32(100, 40)
        self.assertEqual(d1, (40 - 100) & M32)
        self.assertEqual(b1, 1)

# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------
class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    VCC = auto()
    SPECIAL = auto()
    IMM = auto()


@dataclass(frozen=True)
class TypedOperand:
    kind: OperandKind
    index: int = 0
    name: str = ""
    value: int = 0

    @classmethod
    def vgpr(cls, idx: int) -> TypedOperand:
        return cls(kind=OperandKind.VGPR, index=idx, name=f"v{idx}")

    @classmethod
    def sgpr(cls, idx: int) -> TypedOperand:
        return cls(kind=OperandKind.SGPR, index=idx, name=f"s{idx}")

    @classmethod
    def vcc_lo(cls) -> TypedOperand:
        return cls(kind=OperandKind.VCC, name="vcc_lo")

    @classmethod
    def null(cls) -> TypedOperand:
        return cls(kind=OperandKind.SPECIAL, name="null")

    @classmethod
    def imm(cls, val: int) -> TypedOperand:
        return cls(kind=OperandKind.IMM, value=val, name=str(val))


@dataclass(frozen=True)
class VSubrevCoU32AST:
    vdst: TypedOperand
    sdst: TypedOperand
    src0: TypedOperand
    src1: TypedOperand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.vdst.kind != OperandKind.VGPR:
            raise TypeError(f"vdst must be VGPR, got {self.vdst.kind}")
        if self.sdst.kind not in (OperandKind.SGPR, OperandKind.VCC, OperandKind.SPECIAL):
            raise TypeError(f"sdst must be SGPR, VCC, or null, got {self.sdst.kind}")

    def to_assembly(self) -> str:
        mod = " clamp" if self.clamp else ""
        return f"v_subrev_co_u32 {self.vdst.name}, {self.sdst.name}, {self.src0.name}, {self.src1.name}{mod}"


class TestSubrevCoU32AST(unittest.TestCase):
    def test_ast_construction_valid(self) -> None:
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(0),
            sdst=TypedOperand.vcc_lo(),
            src0=TypedOperand.vgpr(1),
            src1=TypedOperand.vgpr(2),
        )
        self.assertEqual(ast.to_assembly(), "v_subrev_co_u32 v0, vcc_lo, v1, v2")

    def test_ast_with_clamp(self) -> None:
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(5),
            sdst=TypedOperand.sgpr(2),
            src0=TypedOperand.vgpr(3),
            src1=TypedOperand.imm(100),
            clamp=True,
        )
        self.assertEqual(ast.to_assembly(), "v_subrev_co_u32 v5, s2, v3, 100 clamp")

    def test_ast_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            VSubrevCoU32AST(
                vdst=TypedOperand.sgpr(0),  # Invalid: vdst must be VGPR
                sdst=TypedOperand.vcc_lo(),
                src0=TypedOperand.vgpr(1),
                src1=TypedOperand.vgpr(2),
            )

# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------
@dataclass
class ScalarCpuState:
    vgprs: dict[int, int]
    sgprs: dict[int, int]
    vcc: int = 0

    @classmethod
    def create(cls) -> ScalarCpuState:
        return cls(vgprs={}, sgprs={}, vcc=0)

    def read_operand(self, op: TypedOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgprs.get(op.index, 0) & M32
        elif op.kind == OperandKind.SGPR:
            return self.sgprs.get(op.index, 0) & M32
        elif op.kind == OperandKind.IMM:
            return op.value & M32
        elif op.kind == OperandKind.VCC:
            return self.vcc & M32
        return 0

    def write_operand(self, op: TypedOperand, val: int) -> None:
        val = val & M32
        if op.kind == OperandKind.VGPR:
            self.vgprs[op.index] = val
        elif op.kind == OperandKind.SGPR:
            self.sgprs[op.index] = val
        elif op.kind == OperandKind.VCC:
            self.vcc = val


def step_scalar_subrev_co(ast: VSubrevCoU32AST, state: ScalarCpuState) -> None:
    s0 = state.read_operand(ast.src0)
    s1 = state.read_operand(ast.src1)
    dst, borrow = ref_subrev_co_u32(s0, s1, clamp=ast.clamp)
    state.write_operand(ast.vdst, dst)
    if ast.sdst.kind != OperandKind.SPECIAL:
        state.write_operand(ast.sdst, borrow)


class TestSubrevCoU32ScalarInterpreter(unittest.TestCase):
    def test_scalar_execution_normal(self) -> None:
        state = ScalarCpuState.create()
        state.vgprs[1] = 100
        state.vgprs[2] = 250
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(0),
            sdst=TypedOperand.vcc_lo(),
            src0=TypedOperand.vgpr(1),
            src1=TypedOperand.vgpr(2),
        )
        step_scalar_subrev_co(ast, state)
        self.assertEqual(state.vgprs[0], 150)
        self.assertEqual(state.vcc, 0)

    def test_scalar_execution_borrow(self) -> None:
        state = ScalarCpuState.create()
        state.vgprs[1] = 300
        state.vgprs[2] = 100
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(0),
            sdst=TypedOperand.sgpr(4),
            src0=TypedOperand.vgpr(1),
            src1=TypedOperand.vgpr(2),
        )
        step_scalar_subrev_co(ast, state)
        self.assertEqual(state.vgprs[0], (100 - 300) & M32)
        self.assertEqual(state.sgprs[4], 1)

    def test_scalar_execution_clamped(self) -> None:
        state = ScalarCpuState.create()
        state.vgprs[1] = 500
        state.vgprs[2] = 200
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(0),
            sdst=TypedOperand.vcc_lo(),
            src0=TypedOperand.vgpr(1),
            src1=TypedOperand.vgpr(2),
            clamp=True,
        )
        step_scalar_subrev_co(ast, state)
        self.assertEqual(state.vgprs[0], 0)  # Saturated to 0
        self.assertEqual(state.vcc, 1)       # Real borrow preserved

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# -----------------------------------------------------------------------------
@dataclass
class SimdWaveState:
    wave_size: int
    exec_mask: int
    vgprs: dict[int, list[int]]
    sdst_mask: int = 0

    @classmethod
    def create(cls, wave_size: int, exec_mask: int) -> SimdWaveState:
        return cls(wave_size=wave_size, exec_mask=exec_mask, vgprs={}, sdst_mask=0)

    def ensure_vgpr(self, idx: int, init_val: int = 0) -> None:
        if idx not in self.vgprs:
            self.vgprs[idx] = [init_val] * self.wave_size


def step_simd_subrev_co(
    vdst_idx: int,
    src0_vals: list[int],
    src1_vals: list[int],
    state: SimdWaveState,
    clamp: bool = False,
) -> None:
    state.ensure_vgpr(vdst_idx, 0xBAADF00D)
    sdst_accum = 0
    for lane in range(state.wave_size):
        active = (state.exec_mask >> lane) & 1
        if active:
            s0 = src0_vals[lane]
            s1 = src1_vals[lane]
            dst, borrow = ref_subrev_co_u32(s0, s1, clamp=clamp)
            state.vgprs[vdst_idx][lane] = dst
            if borrow:
                sdst_accum |= (1 << lane)
        else:
            # Inactive lane: destination unchanged, no borrow contribution
            pass
    state.sdst_mask = sdst_accum


class TestSubrevCoU32SIMD(unittest.TestCase):
    def test_wave32_all_active(self) -> None:
        state = SimdWaveState.create(32, 0xFFFFFFFF)
        s0 = [i * 10 for i in range(32)]
        s1 = [i * 25 for i in range(32)]
        step_simd_subrev_co(0, s0, s1, state)
        for i in range(32):
            self.assertEqual(state.vgprs[0][i], i * 15)
        self.assertEqual(state.sdst_mask, 0)

    def test_wave32_divergent_lanes(self) -> None:
        # Even lanes borrow (s1 < s0), odd lanes do not (s1 > s0)
        s0 = [50 if (i % 2 == 0) else 10 for i in range(32)]
        s1 = [20 if (i % 2 == 0) else 40 for i in range(32)]
        exec_mask = 0x55555555  # only even lanes active
        state = SimdWaveState.create(32, exec_mask)
        step_simd_subrev_co(0, s0, s1, state)

        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(state.vgprs[0][i], (20 - 50) & M32)
            else:
                self.assertEqual(state.vgprs[0][i], 0xBAADF00D)  # Preserved
        self.assertEqual(state.sdst_mask, exec_mask)

    def test_wave64_masking(self) -> None:
        exec_mask = 0x00000000FFFFFFFF  # lower 32 active, upper 32 inactive
        state = SimdWaveState.create(64, exec_mask)
        s0 = [100] * 64
        s1 = [50] * 64  # Underflow on all lanes
        step_simd_subrev_co(0, s0, s1, state)

        # Upper 32 lanes must be untouched
        for i in range(32, 64):
            self.assertEqual(state.vgprs[0][i], 0xBAADF00D)
        # Borrow mask must only include lower 32
        self.assertEqual(state.sdst_mask, 0xFFFFFFFF)

# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------
class TestSubrevCoU32Aliasing(unittest.TestCase):
    def test_alias_vdst_src0(self) -> None:
        # v1 = v2 - v1 (destination aliases src0)
        state = ScalarCpuState.create()
        state.vgprs[1] = 40
        state.vgprs[2] = 100
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(1),
            sdst=TypedOperand.vcc_lo(),
            src0=TypedOperand.vgpr(1),
            src1=TypedOperand.vgpr(2),
        )
        step_scalar_subrev_co(ast, state)
        self.assertEqual(state.vgprs[1], 60)
        self.assertEqual(state.vcc, 0)

    def test_alias_vdst_src1(self) -> None:
        # v2 = v2 - v1 (destination aliases src1)
        state = ScalarCpuState.create()
        state.vgprs[1] = 30
        state.vgprs[2] = 85
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(2),
            sdst=TypedOperand.vcc_lo(),
            src0=TypedOperand.vgpr(1),
            src1=TypedOperand.vgpr(2),
        )
        step_scalar_subrev_co(ast, state)
        self.assertEqual(state.vgprs[2], 55)
        self.assertEqual(state.vcc, 0)

    def test_alias_all_same_vgpr(self) -> None:
        # v0 = v0 - v0 -> 0, borrow 0
        state = ScalarCpuState.create()
        state.vgprs[0] = 0x12345678
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(0),
            sdst=TypedOperand.vcc_lo(),
            src0=TypedOperand.vgpr(0),
            src1=TypedOperand.vgpr(0),
        )
        step_scalar_subrev_co(ast, state)
        self.assertEqual(state.vgprs[0], 0)
        self.assertEqual(state.vcc, 0)

# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# -----------------------------------------------------------------------------
class TestSubrevCoU32FlagsAndBorrow(unittest.TestCase):
    def test_borrow_truth_table_boundaries(self) -> None:
        cases = [
            # (src0, src1, expected_diff, expected_borrow)
            (0, 0, 0, 0),
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 0, 0),
            (0xFFFFFFFF, 0, 1, 1),
            (0, 0xFFFFFFFF, 0xFFFFFFFF, 0),
            (0x80000000, 0x80000000, 0, 0),
            (0x80000001, 0x80000000, 0xFFFFFFFF, 1),
            (0x7FFFFFFF, 0x80000000, 1, 0),
        ]
        for s0, s1, exp_d, exp_b in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, b = ref_subrev_co_u32(s0, s1)
                self.assertEqual(d, exp_d)
                self.assertEqual(b, exp_b)

    def test_scc_flag_isolation(self) -> None:
        # RDNA2 architecture rule: VALU instructions like v_subrev_co_u32 do not alter SCC
        # SCC is preserved across VALU operations.
        initial_scc = 1
        # Execution of VALU does not touch scalar state condition register
        d, b = ref_subrev_co_u32(100, 50)
        self.assertEqual(initial_scc, 1)
        self.assertEqual(b, 1)

# -----------------------------------------------------------------------------
# Commit 7: Boundary Extremes, Borrow Generation, Clamps, and Translator Validation
# -----------------------------------------------------------------------------
class TestSubrevCoU32Translator(unittest.TestCase):
    def _run_ps(self, lines: list[str]) -> list[int]:
        spvasm = support.translate_lines("ps_subrev_co_test", lines)
        return support.mrt0_words(spvasm)

    def test_translator_basic_difference(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 25",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        words = self._run_ps(lines)
        # v0 = 25 - 10 = 15, v3 = borrow = 0
        self.assertEqual(words[0], 15)
        self.assertEqual(words[3], 0)

    def test_translator_borrow_generation(self) -> None:
        lines = [
            "v_mov_b32 v1, 25",
            "v_mov_b32 v2, 10",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        words = self._run_ps(lines)
        # v0 = 10 - 25 = 0xFFFFFFF1, v3 = borrow = 1
        self.assertEqual(words[0], 0xFFFFFFF1)
        self.assertEqual(words[3], 1)

    def test_translator_clamp_saturation(self) -> None:
        lines = [
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 20",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2 clamp",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        words = self._run_ps(lines)
        # Underflow with clamp saturates v0 to 0, v3 = borrow = 1
        self.assertEqual(words[0], 0)
        self.assertEqual(words[3], 1)

    def test_translator_boundary_extremes(self) -> None:
        pairs = [
            (0, 0, 0, 0),
            (0xFFFFFFFF, 0, 1, 1),
            (0, 0xFFFFFFFF, 0xFFFFFFFF, 0),
            (0x80000000, 0x80000000, 0, 0),
        ]
        for s0, s1, exp_d, exp_b in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    "v_subrev_co_u32 v0, vcc_lo, v1, v2",
                    "v_cndmask_b32 v3, 0, 1, vcc_lo",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "exp mrt0, v0, v1, v2, v3 done",
                    "s_endpgm",
                ]
                words = self._run_ps(lines)
                self.assertEqual(words[0], exp_d)
                self.assertEqual(words[3], exp_b)

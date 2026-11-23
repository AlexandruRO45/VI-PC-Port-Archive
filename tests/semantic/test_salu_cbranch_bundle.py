from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 SALU Conditional Branch Execution Suite.

Covers:
- s_cbranch_execnz: conditional branch taken if execution mask EXEC is non-zero (any lane active).
- s_cbranch_execz: conditional branch taken if execution mask EXEC is zero (all lanes inactive).
- s_cbranch_scc0: conditional branch taken if scalar condition code SCC is 0.
- s_cbranch_scc1: conditional branch taken if scalar condition code SCC is 1.
- s_cbranch_vccnz: conditional branch taken if vector condition code VCC is non-zero (wave-wide).
- s_cbranch_vccz: conditional branch taken if vector condition code VCC is zero (wave-wide).

Zero game data, standard library Python only (python -I, struct, math, unittest).
"""

from dataclasses import dataclass
import enum
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(TESTS_DIR)
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT_DIR, "tools")
for p in (HERE, TESTS_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from support import STUB_DXBC, translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


@dataclass
class SaluMachineState:
    pc: int = 0
    scc: int = 0
    exec_mask: int = 0xFFFFFFFFFFFFFFFF
    vcc: int = 0
    sgprs: list[int] = None

    def __post_init__(self):
        if self.sgprs is None:
            self.sgprs = [0] * 106


def eval_branch_condition(mnemonic: str, state: SaluMachineState) -> bool:
    if mnemonic == "s_cbranch_execnz":
        return (state.exec_mask & MASK64) != 0
    if mnemonic == "s_cbranch_execz":
        return (state.exec_mask & MASK64) == 0
    if mnemonic == "s_cbranch_scc0":
        return (state.scc & 1) == 0
    if mnemonic == "s_cbranch_scc1":
        return (state.scc & 1) == 1
    if mnemonic == "s_cbranch_vccnz":
        return (state.vcc & MASK64) != 0
    if mnemonic == "s_cbranch_vccz":
        return (state.vcc & MASK64) == 0
    raise ValueError(f"Unknown branch mnemonic: {mnemonic}")


class BranchRegKind(enum.Enum):
    LABEL = "label"
    SIMM16 = "simm16"


@dataclass(frozen=True)
class BranchInstructionAST:
    mnemonic: str
    target_offset: int


def step_branch(instr: BranchInstructionAST, state: SaluMachineState) -> int:
    if eval_branch_condition(instr.mnemonic, state):
        state.pc = state.pc + 1 + instr.target_offset
    else:
        state.pc = state.pc + 1
    return state.pc


def translate_cs_branch(name: str, lines: list[tuple[str, int]]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "disasm"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "fssm"), exist_ok=True)
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(f"; {name} synthetic branch test\n")
            for text, size in lines:
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
        return spv


def run_cs_branch(spv: str) -> int:
    pc = [0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28
    mem = si.run_workgroup(
        spv, 64, 64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory={}
    )
    return mem.get(0, 0)

# -----------------------------------------------------------------------------
# Opcode 1: s_cbranch_execnz
# -----------------------------------------------------------------------------

def ref_oracle_execnz(exec_mask: int) -> bool:
    return (exec_mask & MASK64) != 0


class SCbranchExecnzOracleTests(unittest.TestCase):
    def test_all_lanes_active(self):
        self.assertTrue(ref_oracle_execnz(0xFFFFFFFFFFFFFFFF))

    def test_single_lane_active(self):
        self.assertTrue(ref_oracle_execnz(1))
        self.assertTrue(ref_oracle_execnz(1 << 31))
        self.assertTrue(ref_oracle_execnz(1 << 63))

    def test_all_lanes_inactive(self):
        self.assertFalse(ref_oracle_execnz(0))

class SCbranchExecnzAstTests(unittest.TestCase):
    def test_ast_construction(self):
        ast = BranchInstructionAST(
            mnemonic="s_cbranch_execnz",
            target_offset=4,
        )
        self.assertEqual(ast.mnemonic, "s_cbranch_execnz")
        self.assertEqual(ast.target_offset, 4)

class SCbranchExecnzScalarInterpreterTests(unittest.TestCase):
    def test_branch_taken_advances_pc(self):
        state = SaluMachineState(pc=10, exec_mask=0x1)
        instr = BranchInstructionAST("s_cbranch_execnz", target_offset=5)
        new_pc = step_branch(instr, state)
        self.assertEqual(new_pc, 16)  # 10 + 1 + 5

    def test_branch_not_taken_falls_through(self):
        state = SaluMachineState(pc=10, exec_mask=0x0)
        instr = BranchInstructionAST("s_cbranch_execnz", target_offset=5)
        new_pc = step_branch(instr, state)
        self.assertEqual(new_pc, 11)  # 10 + 1

class SCbranchExecnzTranslatorTests(unittest.TestCase):
    def test_branches_registry(self):
        self.assertIn("s_cbranch_execnz", rdna2spirv.BRANCHES)

    def test_branch_condition_emission(self):
        g = rdna2spirv.Gen("cs")
        cond = rdna2spirv.branch_condition(g, "s_cbranch_execnz")
        self.assertTrue(g.uses_vote)
        body = "\n".join(g.body)
        self.assertIn("OpGroupNonUniformAny", body)

    def test_execution_taken(self):
        lines = [
            ("s_mov_b32 s20, 7", 4),
            ("s_cbranch_execnz 1", 4),
            ("s_mov_b32 s20, 9", 4),
            ("v_mov_b32_e32 v6, s20", 4),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv = translate_cs_branch("cs_cbranch_execnz_taken", lines)
        val = run_cs_branch(spv)
        self.assertEqual(val, 7)

# -----------------------------------------------------------------------------
# Opcode 2: s_cbranch_execz
# -----------------------------------------------------------------------------

def ref_oracle_execz(exec_mask: int) -> bool:
    return (exec_mask & MASK64) == 0


class SCbranchExeczOracleTests(unittest.TestCase):
    def test_all_lanes_inactive(self):
        self.assertTrue(ref_oracle_execz(0))

    def test_active_lanes(self):
        self.assertFalse(ref_oracle_execz(1))
        self.assertFalse(ref_oracle_execz(0xFFFFFFFFFFFFFFFF))

class SCbranchExeczAstTests(unittest.TestCase):
    def test_ast_construction(self):
        ast = BranchInstructionAST(
            mnemonic="s_cbranch_execz",
            target_offset=-2,
        )
        self.assertEqual(ast.mnemonic, "s_cbranch_execz")
        self.assertEqual(ast.target_offset, -2)

class SCbranchExeczScalarInterpreterTests(unittest.TestCase):
    def test_branch_taken_when_exec_zero(self):
        state = SaluMachineState(pc=20, exec_mask=0)
        instr = BranchInstructionAST("s_cbranch_execz", target_offset=3)
        self.assertEqual(step_branch(instr, state), 24)

    def test_branch_not_taken_when_exec_nonzero(self):
        state = SaluMachineState(pc=20, exec_mask=0xFF)
        instr = BranchInstructionAST("s_cbranch_execz", target_offset=3)
        self.assertEqual(step_branch(instr, state), 21)

class SCbranchExeczTranslatorTests(unittest.TestCase):
    def test_branches_registry(self):
        self.assertIn("s_cbranch_execz", rdna2spirv.BRANCHES)

    def test_branch_condition_emission(self):
        g = rdna2spirv.Gen("cs")
        cond = rdna2spirv.branch_condition(g, "s_cbranch_execz")
        self.assertTrue(g.uses_vote)
        body = "\n".join(g.body)
        self.assertIn("OpGroupNonUniformAny", body)
        self.assertIn("OpLogicalNot", body)

    def test_execution_not_taken(self):
        lines = [
            ("s_mov_b32 s20, 7", 4),
            ("s_cbranch_execz 1", 4),
            ("s_mov_b32 s20, 9", 4),
            ("v_mov_b32_e32 v6, s20", 4),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv = translate_cs_branch("cs_cbranch_execz_fallthrough", lines)
        val = run_cs_branch(spv)
        self.assertEqual(val, 9)  # Fallthrough executes s_mov_b32 s20, 9

if __name__ == '__main__':
    unittest.main()

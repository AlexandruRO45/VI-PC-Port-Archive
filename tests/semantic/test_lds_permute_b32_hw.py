"""Semantic interpreter tests and reference mathematical oracle for ds_permute_b32.

RDNA2 / GFX10 LDS cross-lane forward (scatter) permute:
    ds_permute_b32 vdst, vaddr, vdata [offset:offset16]

Semantics:
    Every source lane i active in EXEC calculates:
        effective_address = (vaddr + offset16) & 0xFFFFFFFF
        target_lane = ((effective_address >> 2) & 31) (within its 32-lane half)
    and sends its vdata to target_lane.
    - Destination lanes that receive no data become 0 (when active in EXEC).
    - Destination lanes that are inactive in EXEC retain their old vdst.
    - In wave64, operations occur independently within half 0 (lanes 0..31) and half 1 (lanes 32..63).
    - Collisions: If multiple active lanes target the same destination, the RDNA2 ISA states
      the result is one of the writers. Leonida translator resolves this with highest-numbered writer winning.
"""
from __future__ import annotations

import math
import os
import re
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF
OFFSET16_MASK: int = 0xFFFF
OLD_VDST: int = 0x5A5A1234
ALL32: int = 0xFFFFFFFF
ALL64: int = (1 << 64) - 1


def calc_target_lane(vaddr: int, offset: int = 0) -> int:
    """Calculate target lane index within a 32-lane half."""
    eff = (vaddr + (offset & OFFSET16_MASK)) & MASK32
    return (eff >> 2) & 31


def ref_ds_permute_scalar(vaddr: int, vdata: int, offset: int = 0) -> tuple[int, int]:
    """Pure mathematical single-lane target and masked 32-bit data."""
    return calc_target_lane(vaddr, offset), vdata & MASK32


def ref_ds_permute_wave(
    addrs: list[int],
    datas: list[int],
    exec_mask: int = ALL32,
    offset: int = 0,
    lanes: int = 32,
    old_dsts: list[int] | None = None,
) -> list[int]:
    """Pure reference simulation of ds_permute_b32 across wave32 or wave64.

    Matches Leonida translator semantics where highest active source lane wins on collision.
    """
    if old_dsts is None:
        old_dsts = [OLD_VDST] * lanes
    out = list(old_dsts)

    for half_base in range(0, lanes, 32):
        half_len = min(32, lanes - half_base)
        writers: dict[int, list[int]] = {d: [] for d in range(32)}

        for i in range(half_len):
            src_lane = half_base + i
            if (exec_mask >> src_lane) & 1:
                target = calc_target_lane(addrs[src_lane], offset)
                writers[target].append(datas[src_lane] & MASK32)

        for i in range(half_len):
            dst_lane = half_base + i
            if (exec_mask >> dst_lane) & 1:
                if writers[i]:
                    out[dst_lane] = writers[i][-1]
                else:
                    out[dst_lane] = 0
            else:
                out[dst_lane] = old_dsts[dst_lane]

    return out


class ArchitecturalPermuteModel:
    """Architectural model verifying forward cross-lane permute and collision sets."""

    def __init__(self, lanes: int = 32) -> None:
        self.lanes = lanes

    def execute(
        self,
        addrs: list[int],
        datas: list[int],
        exec_mask: int,
        offset: int = 0,
        old_dsts: list[int] | None = None,
    ) -> list[int]:
        return ref_ds_permute_wave(addrs, datas, exec_mask, offset, self.lanes, old_dsts)

    def allowed_outcomes(
        self,
        addrs: list[int],
        datas: list[int],
        exec_mask: int,
        offset: int = 0,
        old_dsts: list[int] | None = None,
    ) -> list[set[int]]:
        """Returns the set of all valid values per lane permitted by the RDNA2 ISA."""
        if old_dsts is None:
            old_dsts = [OLD_VDST] * self.lanes
        out: list[set[int]] = []

        for half_base in range(0, self.lanes, 32):
            half_len = min(32, self.lanes - half_base)
            writers: dict[int, list[int]] = {d: [] for d in range(32)}

            for i in range(half_len):
                src_lane = half_base + i
                if (exec_mask >> src_lane) & 1:
                    target = calc_target_lane(addrs[src_lane], offset)
                    writers[target].append(datas[src_lane] & MASK32)

            for i in range(half_len):
                dst_lane = half_base + i
                if (exec_mask >> dst_lane) & 1:
                    if writers[i]:
                        out.append(set(writers[i]))
                    else:
                        out.append({0})
                else:
                    out.append({old_dsts[dst_lane]})

        return out


class TestDsPermuteOracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Mathematical reference oracle and architectural model tests."""

    def test_target_lane_calculation_basic(self) -> None:
        # Address in bytes: dword 0 is byte 0, dword 1 is byte 4, etc.
        self.assertEqual(calc_target_lane(0), 0)
        self.assertEqual(calc_target_lane(4), 1)
        self.assertEqual(calc_target_lane(8), 2)
        self.assertEqual(calc_target_lane(124), 31)
        # Modulo 32 wrapping: dword 32 is byte 128 -> lane 0
        self.assertEqual(calc_target_lane(128), 0)
        self.assertEqual(calc_target_lane(132), 1)

    def test_target_lane_offset_addition(self) -> None:
        # Address 0 with offset 4 -> lane 1
        self.assertEqual(calc_target_lane(0, offset=4), 1)
        # Address 4 with offset 8 -> byte 12 -> lane 3
        self.assertEqual(calc_target_lane(4, offset=8), 3)
        # Unaligned low 2 bits are ignored by >> 2
        self.assertEqual(calc_target_lane(1, offset=0), 0)
        self.assertEqual(calc_target_lane(2, offset=0), 0)
        self.assertEqual(calc_target_lane(3, offset=0), 0)
        self.assertEqual(calc_target_lane(5, offset=0), 1)

    def test_reference_oracle_unwritten_lanes_zero(self) -> None:
        # Only lane 0 writes to lane 5; all other lanes remain unwritten
        addrs = [20 if i == 0 else 0 for i in range(32)]
        datas = [0xABCD if i == 0 else 0 for i in range(32)]
        # EXEC has all 32 lanes on, but only lane 0 is an active writer (exec_mask=0x1)
        res = ref_ds_permute_wave(addrs, datas, exec_mask=0x1, offset=0, lanes=32)
        # Lane 5 receives 0xABCD, unwritten lanes (0..4, 6..31) are off in EXEC so they keep OLD_VDST
        self.assertEqual(res[5], OLD_VDST)  # since lane 5 is off in EXEC (exec_mask=1 only has bit 0 on)
        self.assertEqual(res[0], 0)         # lane 0 is active, nobody wrote to it, gets 0

        # Now all lanes on in EXEC:
        res_all = ref_ds_permute_wave(addrs, datas, exec_mask=ALL32, offset=0, lanes=32)
        self.assertEqual(res_all[5], 0xABCD)
        for d in range(32):
            if d != 5 and d != 0:
                self.assertEqual(res_all[d], 0)

    def test_reference_oracle_collision_highest_writer_wins(self) -> None:
        # All 32 lanes write to lane 0 with data = 100 + lane
        addrs = [0] * 32
        datas = [100 + i for i in range(32)]
        res = ref_ds_permute_wave(addrs, datas, exec_mask=ALL32, offset=0, lanes=32)
        self.assertEqual(res[0], 100 + 31)  # lane 31 is highest writer
        for d in range(1, 32):
            self.assertEqual(res[d], 0)

    def test_architectural_model_wave64_halves_isolation(self) -> None:
        model = ArchitecturalPermuteModel(lanes=64)
        # Lower half lanes write to lane 10; upper half lanes write to lane 10 of upper half (lane 42)
        addrs = [40] * 64  # byte 40 -> lane 10
        datas = [1000 + i for i in range(64)]
        res = model.execute(addrs, datas, exec_mask=ALL64, offset=0)
        self.assertEqual(res[10], 1000 + 31)
        self.assertEqual(res[42], 1000 + 63)
        self.assertEqual(res[0], 0)
        self.assertEqual(res[32], 0)


class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class ASTNode:
    pass


@dataclass(frozen=True)
class RegisterNode(ASTNode):
    name: str
    index: int
    is_vector: bool = True

    def __str__(self) -> str:
        return f"{'v' if self.is_vector else 's'}{self.index}"


@dataclass(frozen=True)
class ImmediateNode(ASTNode):
    value: int

    def __str__(self) -> str:
        return hex(self.value) if self.value > 9 else str(self.value)


@dataclass(frozen=True)
class DsPermuteInsn(ASTNode):
    dst: RegisterNode
    addr: RegisterNode
    data: RegisterNode
    offset: int = 0

    def __str__(self) -> str:
        mod = f" offset:{self.offset}" if self.offset else ""
        return f"ds_permute_b32 {self.dst}, {self.addr}, {self.data}{mod}"


def parse_ds_permute_line(line: str) -> DsPermuteInsn:
    """Parse a ds_permute_b32 assembly line into a symbolic AST node."""
    clean = line.strip().split(";")[0].strip()
    if not clean.startswith("ds_permute_b32"):
        raise ValueError(f"Not a ds_permute_b32 instruction: {line}")

    body = clean[len("ds_permute_b32"):].strip()
    parts = [p.strip() for p in body.split(",") if p.strip()]
    if len(parts) < 3:
        raise ValueError(f"Malformed ds_permute_b32 instruction, expected 3 operands: {line}")

    def parse_reg(tok: str) -> RegisterNode:
        m = re.match(r"^([vs])(\d+)$", tok.strip())
        if not m:
            raise ValueError(f"Invalid register operand: {tok}")
        return RegisterNode(name=tok.strip(), index=int(m.group(2)), is_vector=(m.group(1) == "v"))

    dst = parse_reg(parts[0])
    addr = parse_reg(parts[1])

    data_tok = parts[2]
    offset = 0
    if "offset:" in data_tok:
        subparts = data_tok.split()
        data = parse_reg(subparts[0])
        for sp in subparts[1:]:
            if sp.startswith("offset:"):
                off_val_str = sp.split(":")[1]
                offset = int(off_val_str, 0)
    else:
        data = parse_reg(data_tok)

    return DsPermuteInsn(dst=dst, addr=addr, data=data, offset=offset)


class TestDsPermuteSymbolicAST(unittest.TestCase):
    """Part 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_immutability(self) -> None:
        dst = RegisterNode("v10", 10)
        addr = RegisterNode("v1", 1)
        data = RegisterNode("v2", 2)
        insn = DsPermuteInsn(dst=dst, addr=addr, data=data, offset=16)

        self.assertEqual(insn.dst.index, 10)
        self.assertEqual(insn.addr.index, 1)
        self.assertEqual(insn.data.index, 2)
        self.assertEqual(insn.offset, 16)
        self.assertEqual(str(insn), "ds_permute_b32 v10, v1, v2 offset:16")

    def test_parse_ds_permute_standard(self) -> None:
        insn = parse_ds_permute_line("ds_permute_b32 v0, v1, v2")
        self.assertEqual(insn.dst.index, 0)
        self.assertEqual(insn.addr.index, 1)
        self.assertEqual(insn.data.index, 2)
        self.assertEqual(insn.offset, 0)

    def test_parse_ds_permute_with_offset(self) -> None:
        insn = parse_ds_permute_line("ds_permute_b32 v15, v3, v4 offset:32")
        self.assertEqual(insn.dst.index, 15)
        self.assertEqual(insn.addr.index, 3)
        self.assertEqual(insn.data.index, 4)
        self.assertEqual(insn.offset, 32)

    def test_parse_ds_permute_hex_offset_and_comments(self) -> None:
        insn = parse_ds_permute_line("  ds_permute_b32 v10, v20, v30 offset:0x40 ; scatter pass")
        self.assertEqual(insn.dst.index, 10)
        self.assertEqual(insn.addr.index, 20)
        self.assertEqual(insn.data.index, 30)
        self.assertEqual(insn.offset, 64)


class ScalarPermuteState:
    """State tracking for scalar / single-lane instruction interpretation."""

    def __init__(self) -> None:
        self.registers: dict[int, int] = {}

    def set_reg(self, idx: int, val: int) -> None:
        self.registers[idx] = val & MASK32

    def get_reg(self, idx: int) -> int:
        return self.registers.get(idx, 0)


class ScalarPermuteInterpreter:
    """Interpreter modeling deterministic single-lane address calculation and message emission."""

    def __init__(self, state: ScalarPermuteState) -> None:
        self.state = state

    def eval_instruction(self, insn: DsPermuteInsn) -> tuple[int, int]:
        vaddr = self.state.get_reg(insn.addr.index)
        vdata = self.state.get_reg(insn.data.index)
        target = calc_target_lane(vaddr, insn.offset)
        return target, vdata


class TestDsPermuteScalarInterpreter(unittest.TestCase):
    """Part 3: Deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarPermuteState()
        self.interp = ScalarPermuteInterpreter(self.state)

    def test_scalar_identity_mapping(self) -> None:
        self.state.set_reg(1, 0)       # addr = 0 -> lane 0
        self.state.set_reg(2, 0x1234)  # data = 0x1234
        insn = DsPermuteInsn(RegisterNode("v0", 0), RegisterNode("v1", 1), RegisterNode("v2", 2))
        target, data = self.interp.eval_instruction(insn)
        self.assertEqual(target, 0)
        self.assertEqual(data, 0x1234)

    def test_scalar_target_lane_with_offset(self) -> None:
        self.state.set_reg(1, 16)      # addr = 16 -> dword 4
        self.state.set_reg(2, 0xCAFE)
        insn = DsPermuteInsn(RegisterNode("v0", 0), RegisterNode("v1", 1), RegisterNode("v2", 2), offset=8)
        # (16 + 8) = 24 -> 24 >> 2 = 6 -> lane 6
        target, data = self.interp.eval_instruction(insn)
        self.assertEqual(target, 6)
        self.assertEqual(data, 0xCAFE)

    def test_scalar_target_lane_wrapping(self) -> None:
        self.state.set_reg(1, 128)     # addr = 128 -> lane 32 % 32 = 0
        self.state.set_reg(2, 0x9999)
        insn = DsPermuteInsn(RegisterNode("v0", 0), RegisterNode("v1", 1), RegisterNode("v2", 2))
        target, data = self.interp.eval_instruction(insn)
        self.assertEqual(target, 0)
        self.assertEqual(data, 0x9999)


class SIMDPermuteEngine:
    """Architectural SIMD engine simulating vector execution under 32-bit and 64-bit EXEC masks."""

    def __init__(self, lanes: int = 32) -> None:
        self.lanes = lanes
        self.vgprs: dict[int, list[int]] = {r: [0] * lanes for r in range(256)}
        self.exec_mask: int = (1 << lanes) - 1

    def set_vgpr(self, r: int, values: list[int]) -> None:
        self.vgprs[r] = [v & MASK32 for v in values[:self.lanes]]

    def get_vgpr(self, r: int) -> list[int]:
        return list(self.vgprs.get(r, [0] * self.lanes))

    def step_ds_permute(self, dst_idx: int, addr_idx: int, data_idx: int, offset: int = 0) -> None:
        addrs = self.get_vgpr(addr_idx)
        datas = self.get_vgpr(data_idx)
        old_dsts = self.get_vgpr(dst_idx)

        res = ref_ds_permute_wave(
            addrs=addrs,
            datas=datas,
            exec_mask=self.exec_mask,
            offset=offset,
            lanes=self.lanes,
            old_dsts=old_dsts,
        )
        self.vgprs[dst_idx] = res


class TestDsPermuteSIMDExecution(unittest.TestCase):
    """Part 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_full_exec_wave32_rotation(self) -> None:
        engine = SIMDPermuteEngine(lanes=32)
        # Each lane i sends its data (1000 + i) to lane (i + 1) % 32
        # Address in bytes: (i + 1) * 4
        addrs = [((i + 1) % 32) * 4 for i in range(32)]
        datas = [1000 + i for i in range(32)]
        engine.set_vgpr(1, addrs)
        engine.set_vgpr(2, datas)
        engine.set_vgpr(0, [OLD_VDST] * 32)

        engine.step_ds_permute(dst_idx=0, addr_idx=1, data_idx=2)
        res = engine.get_vgpr(0)

        # Lane 0 receives lane 31's data; lane 1 receives lane 0's data
        self.assertEqual(res[0], 1000 + 31)
        for i in range(1, 32):
            self.assertEqual(res[i], 1000 + i - 1)

    def test_divergent_alternating_exec_even_active(self) -> None:
        engine = SIMDPermuteEngine(lanes=32)
        engine.exec_mask = 0x55555555  # Even lanes active (0, 2, 4, ...)
        # Even lane i writes to i; odd lane i writes to i (but is inactive)
        addrs = [i * 4 for i in range(32)]
        datas = [700 + i for i in range(32)]
        engine.set_vgpr(1, addrs)
        engine.set_vgpr(2, datas)
        engine.set_vgpr(0, [OLD_VDST] * 32)

        engine.step_ds_permute(dst_idx=0, addr_idx=1, data_idx=2)
        res = engine.get_vgpr(0)

        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(res[i], 700 + i)
            else:
                # Inactive destination lane preserves old vdst
                self.assertEqual(res[i], OLD_VDST)

    def test_wave64_divergent_halves(self) -> None:
        engine = SIMDPermuteEngine(lanes=64)
        # Lower half: full EXEC. Upper half: low 16 active (0xFFFF0000FFFF)
        engine.exec_mask = (0x0000FFFF << 32) | 0xFFFFFFFF
        addrs = [(i * 4) for i in range(64)]
        datas = [2000 + i for i in range(64)]
        engine.set_vgpr(1, addrs)
        engine.set_vgpr(2, datas)
        engine.set_vgpr(0, [OLD_VDST] * 64)

        engine.step_ds_permute(dst_idx=0, addr_idx=1, data_idx=2)
        res = engine.get_vgpr(0)

        # Lower half all active
        for i in range(32):
            self.assertEqual(res[i], 2000 + i)
        # Upper half lanes 32..47 active
        for i in range(32, 48):
            self.assertEqual(res[i], 2000 + i)
        # Upper half lanes 48..63 inactive -> preserve OLD_VDST
        for i in range(48, 64):
            self.assertEqual(res[i], OLD_VDST)


class TestDsPermuteRegisterAliasing(unittest.TestCase):
    """Part 5: Register aliasing and in-place operand preservation."""

    def test_aliasing_vdst_equals_vaddr(self) -> None:
        engine = SIMDPermuteEngine(lanes=32)
        # Rotate by 1: lane i sends to lane (i + 1) % 32
        addrs = [((i + 1) % 32) * 4 for i in range(32)]
        datas = [500 + i for i in range(32)]
        engine.set_vgpr(1, addrs)  # v1 is addr AND dst
        engine.set_vgpr(2, datas)  # v2 is data

        # Instruction: ds_permute_b32 v1, v1, v2
        engine.step_ds_permute(dst_idx=1, addr_idx=1, data_idx=2)
        res = engine.get_vgpr(1)

        # All lanes must sample their address before v1 is overwritten
        self.assertEqual(res[0], 500 + 31)
        for i in range(1, 32):
            self.assertEqual(res[i], 500 + i - 1)

    def test_aliasing_vdst_equals_vdata(self) -> None:
        engine = SIMDPermuteEngine(lanes=32)
        addrs = [((i + 1) % 32) * 4 for i in range(32)]
        datas = [600 + i for i in range(32)]
        engine.set_vgpr(1, addrs)  # v1 is addr
        engine.set_vgpr(2, datas)  # v2 is data AND dst

        # Instruction: ds_permute_b32 v2, v1, v2
        engine.step_ds_permute(dst_idx=2, addr_idx=1, data_idx=2)
        res = engine.get_vgpr(2)

        self.assertEqual(res[0], 600 + 31)
        for i in range(1, 32):
            self.assertEqual(res[i], 600 + i - 1)

    def test_aliasing_all_operands_same_register(self) -> None:
        engine = SIMDPermuteEngine(lanes=32)
        # Identity address: lane i has (i * 4) as both addr and data
        vals = [i * 4 for i in range(32)]
        engine.set_vgpr(0, vals)

        # Instruction: ds_permute_b32 v0, v0, v0
        engine.step_ds_permute(dst_idx=0, addr_idx=0, data_idx=0)
        res = engine.get_vgpr(0)

        for i in range(32):
            self.assertEqual(res[i], i * 4)


class TestDsPermuteCollisionsAndUnwritten(unittest.TestCase):
    """Part 6: Collision resolution and unwritten lane semantics."""

    def test_all_lanes_target_single_lane(self) -> None:
        engine = SIMDPermuteEngine(lanes=32)
        # All 32 lanes target lane 7 (byte address 28)
        engine.set_vgpr(1, [28] * 32)
        engine.set_vgpr(2, [100 + i for i in range(32)])
        engine.set_vgpr(0, [OLD_VDST] * 32)

        engine.step_ds_permute(dst_idx=0, addr_idx=1, data_idx=2)
        res = engine.get_vgpr(0)

        # Lane 7 receives the highest active writer (lane 31)
        self.assertEqual(res[7], 100 + 31)
        # All other active destination lanes receive 0
        for i in range(32):
            if i != 7:
                self.assertEqual(res[i], 0)

    def test_two_to_one_pairing_collision(self) -> None:
        engine = SIMDPermuteEngine(lanes=32)
        # Lanes 2i and 2i+1 both target lane i (for i in 0..15)
        addrs = [(i // 2) * 4 for i in range(32)]
        datas = [1000 + i for i in range(32)]
        engine.set_vgpr(1, addrs)
        engine.set_vgpr(2, datas)
        engine.set_vgpr(0, [OLD_VDST] * 32)

        engine.step_ds_permute(dst_idx=0, addr_idx=1, data_idx=2)
        res = engine.get_vgpr(0)

        # Lanes 0..15 receive the higher odd writer (2i + 1)
        for i in range(16):
            self.assertEqual(res[i], 1000 + 2 * i + 1)
        # Lanes 16..31 are unwritten and receive 0
        for i in range(16, 32):
            self.assertEqual(res[i], 0)

    def test_inactive_higher_source_does_not_win(self) -> None:
        engine = SIMDPermuteEngine(lanes=32)
        # Lane 0 and lane 1 both target lane 0
        addrs = [0] * 32
        datas = [50 + i for i in range(32)]
        # EXEC mask has lane 0 active, but lane 1 inactive (exec_mask = 0x1)
        engine.exec_mask = 0x1
        engine.set_vgpr(1, addrs)
        engine.set_vgpr(2, datas)
        engine.set_vgpr(0, [OLD_VDST] * 32)

        engine.step_ds_permute(dst_idx=0, addr_idx=1, data_idx=2)
        res = engine.get_vgpr(0)

        # Since lane 1 is inactive, active lane 0 wins!
        self.assertEqual(res[0], 50)

if __name__ == "__main__":
    unittest.main()

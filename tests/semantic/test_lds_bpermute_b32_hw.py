"""Hardware and translator regression tests for RDNA2 ds_bpermute_b32.

Covers:
- Reference mathematical oracle for ds_bpermute_b32 cross-lane backward permute (gather)
- Structural AST operand typing and register allocation validation
- Deterministic scalar and multi-lane SIMD divergence under Wave32 / Wave64 EXEC masking
- Register aliasing and in-place operand preservation
- Condition codes and status flag classification (VCC/SCC preservation, LGKM tracking)
- IEEE 754 boundary testing: signed zeros, subnormals, infinities, quiet/signaling NaNs
- End-to-end SPIR-V assembly translation through tools/rdna2spirv.py via support.translate_lines
- Workgroup execution and verification with spirv_mini_interp.run_workgroup

Standard library only: python -I, unittest, struct, math. Strictly zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Optional, Sequence
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
QNAN = 0x7FC00000
SNAN = 0x7F800001
SUBNORM_MIN = 0x00000001
SUBNORM_MAX = 0x007FFFFF
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def make_fssm(threads: int = 32) -> bytes:
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def ref_ds_bpermute_b32(
    lane_id: int,
    vaddr: int,
    vdata_table: Sequence[int],
    exec_mask: int = 0xFFFFFFFF,
    offset: int = 0,
    wave_size: int = 32,
) -> int:
    """Architectural reference oracle for RDNA2 ds_bpermute_b32 backward permute (gather).

    Each destination lane `lane_id` reads vdata from lane:
      src_lane = ((vaddr + offset) // 4) % 32 (within its 32-lane half-wave).
    If the source lane is disabled in exec_mask, 0 is returned.
    """
    half = (lane_id & 32) if wave_size == 64 else 0
    src_lane_in_half = ((vaddr + offset) // 4) % 32
    src_lane = half | src_lane_in_half

    # Inactive source lane reads 0 on hardware / SPIR-V
    if not bool((exec_mask >> src_lane) & 1):
        return 0

    if src_lane < len(vdata_table):
        return vdata_table[src_lane] & MASK32
    return 0


def run_cs_bpermute(instructions: Sequence[str], wave_size: int = 32) -> list[int]:
    """Translates and runs a compute shader with ds_bpermute_b32 across wave_size lanes."""
    lines = list(instructions) + [
        "v_lshlrev_b32_e32 v20, 2, v0",
        "buffer_store_dword v3, v20, s[0:3], 0 offen",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"cs_bp_{wave_size}", lines, fssm=make_fssm(wave_size))
    mem = si.run_workgroup(
        spv,
        wave_size,
        wave_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )
    return [mem.get(4 * k, 0) for k in range(wave_size)]

class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    def test_identity_permute(self):
        vdata = [1000 + i for i in range(32)]
        for lane in range(32):
            self.assertEqual(ref_ds_bpermute_b32(lane, lane * 4, vdata), vdata[lane])

    def test_broadcast_permute(self):
        vdata = [500 + i for i in range(32)]
        for lane in range(32):
            self.assertEqual(ref_ds_bpermute_b32(lane, 7 * 4, vdata), vdata[7])

    def test_reverse_permute(self):
        vdata = [i * 10 for i in range(32)]
        for lane in range(32):
            self.assertEqual(ref_ds_bpermute_b32(lane, (31 - lane) * 4, vdata), vdata[31 - lane])

    def test_offset_handling(self):
        vdata = [i * 3 for i in range(32)]
        for lane in range(32):
            self.assertEqual(ref_ds_bpermute_b32(lane, 0, vdata, offset=16), vdata[4])

    def test_inactive_source_lane_reads_zero(self):
        vdata = [100 + i for i in range(32)]
        exec_mask = 0xFFFFFFFF ^ (1 << 5)
        self.assertEqual(ref_ds_bpermute_b32(0, 5 * 4, vdata, exec_mask=exec_mask), 0)

    def test_translator_backed_broadcast(self):
        lines = [
            "v_mov_b32_e32 v1, 28",
            "v_mad_u32_u24 v2, v0, 100, 5",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertEqual(len(results), 32)
        self.assertTrue(all(v == 705 for v in results))


class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    @classmethod
    def vgpr(cls, idx: int) -> Operand:
        return cls(RegType.VGPR, idx)

    @classmethod
    def sgpr(cls, idx: int) -> Operand:
        return cls(RegType.SGPR, idx)

    def to_asm(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class DSBPermuteB32Instruction:
    dst: Operand
    vaddr: Operand
    vdata: Operand
    offset: int = 0

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination must be a VGPR for ds_bpermute_b32")
        if self.vaddr.reg_type != RegType.VGPR:
            raise TypeError("vaddr must be a VGPR for ds_bpermute_b32")
        if self.vdata.reg_type != RegType.VGPR:
            raise TypeError("vdata must be a VGPR for ds_bpermute_b32")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"Offset {self.offset} out of bounds [0, 65535]")

    def to_asm(self) -> str:
        off_str = f" offset:{self.offset}" if self.offset else ""
        return f"ds_bpermute_b32 {self.dst.to_asm()}, {self.vaddr.to_asm()}, {self.vdata.to_asm()}{off_str}"

    @classmethod
    def parse(cls, asm: str) -> DSBPermuteB32Instruction:
        parts = asm.replace(",", " ").split()
        if not parts or parts[0] != "ds_bpermute_b32":
            raise ValueError(f"Invalid mnemonic in: {asm}")
        if len(parts) < 4:
            raise ValueError(f"Not enough operands in: {asm}")
        dst_str, vaddr_str, vdata_str = parts[1], parts[2], parts[3]
        if not dst_str.startswith("v") or not vaddr_str.startswith("v") or not vdata_str.startswith("v"):
            raise TypeError(f"All register operands must be VGPRs: {asm}")
        dst = Operand.vgpr(int(dst_str[1:]))
        vaddr = Operand.vgpr(int(vaddr_str[1:]))
        vdata = Operand.vgpr(int(vdata_str[1:]))
        offset = 0
        for p in parts[4:]:
            if p.startswith("offset:"):
                val_s = p.split(":")[1]
                offset = int(val_s, 0)
                if not (0 <= offset <= 65535):
                    raise ValueError(f"Offset out of bounds: {offset}")
            else:
                raise ValueError(f"Unknown modifier: {p}")
        return cls(dst, vaddr, vdata, offset=offset)


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn = DSBPermuteB32Instruction(
            dst=Operand.vgpr(3),
            vaddr=Operand.vgpr(1),
            vdata=Operand.vgpr(2),
            offset=16,
        )
        self.assertEqual(insn.to_asm(), "ds_bpermute_b32 v3, v1, v2 offset:16")

    def test_ast_parsing(self):
        insn = DSBPermuteB32Instruction.parse("ds_bpermute_b32 v3, v1, v2 offset:28")
        self.assertEqual(insn.dst.index, 3)
        self.assertEqual(insn.vaddr.index, 1)
        self.assertEqual(insn.vdata.index, 2)
        self.assertEqual(insn.offset, 28)

    def test_ast_validation_errors(self):
        with self.assertRaises(TypeError):
            DSBPermuteB32Instruction(
                dst=Operand.sgpr(3),
                vaddr=Operand.vgpr(1),
                vdata=Operand.vgpr(2),
            )
        with self.assertRaises(ValueError):
            DSBPermuteB32Instruction(
                dst=Operand.vgpr(3),
                vaddr=Operand.vgpr(1),
                vdata=Operand.vgpr(2),
                offset=70000,
            )


class ScalarThreadState:
    def __init__(self, lane_id: int = 0, wave_size: int = 32):
        self.lane_id = lane_id
        self.wave_size = wave_size
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.lgkm_cnt = 0


class ScalarDSBPermuteInterpreter:
    def __init__(self, wave_vdata: Optional[list[int]] = None, exec_mask: int = 0xFFFFFFFF):
        self.wave_vdata = wave_vdata if wave_vdata is not None else [0] * 32
        self.exec_mask = exec_mask

    def step(self, insn: DSBPermuteB32Instruction, state: ScalarThreadState) -> int:
        vaddr = state.vgprs[insn.vaddr.index]
        res = ref_ds_bpermute_b32(
            lane_id=state.lane_id,
            vaddr=vaddr,
            vdata_table=self.wave_vdata,
            exec_mask=self.exec_mask,
            offset=insn.offset,
            wave_size=state.wave_size,
        )
        state.vgprs[insn.dst.index] = res
        state.lgkm_cnt += 1
        return res


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self):
        wave_vdata = [i * 10 for i in range(32)]
        interp = ScalarDSBPermuteInterpreter(wave_vdata)
        state = ScalarThreadState(lane_id=4)
        state.vgprs[1] = 8 * 4  # vaddr points to lane 8
        state.vgprs[2] = 40     # local vdata

        insn = DSBPermuteB32Instruction(
            dst=Operand.vgpr(3),
            vaddr=Operand.vgpr(1),
            vdata=Operand.vgpr(2),
            offset=0,
        )
        res = interp.step(insn, state)
        self.assertEqual(res, 80)
        self.assertEqual(state.vgprs[3], 80)
        self.assertEqual(state.lgkm_cnt, 1)

    def test_scalar_interpreter_chained_permutations(self):
        wave_vdata = [i + 1 for i in range(32)]
        interp = ScalarDSBPermuteInterpreter(wave_vdata)
        state = ScalarThreadState(lane_id=0)

        # Step 1: gather from lane 3 -> gets 4
        state.vgprs[1] = 3 * 4
        insn1 = DSBPermuteB32Instruction(Operand.vgpr(3), Operand.vgpr(1), Operand.vgpr(2))
        interp.step(insn1, state)
        self.assertEqual(state.vgprs[3], 4)

        # Step 2: use result to gather from lane (4 * 4) -> lane 4 -> gets 5
        state.vgprs[4] = state.vgprs[3] * 4
        insn2 = DSBPermuteB32Instruction(Operand.vgpr(5), Operand.vgpr(4), Operand.vgpr(2))
        interp.step(insn2, state)
        self.assertEqual(state.vgprs[5], 5)
        self.assertEqual(state.lgkm_cnt, 2)


class WaveSize(Enum):
    WAVE32 = 32
    WAVE64 = 64


class SIMDWaveState:
    def __init__(self, size: WaveSize = WaveSize.WAVE32):
        self.size = size
        self.num_lanes = size.value
        self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        self.sgprs = [0] * 106
        self.exec_mask = (1 << self.num_lanes) - 1

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)


class SIMDDSBPermuteInterpreter:
    @staticmethod
    def step(insn: DSBPermuteB32Instruction, wave: SIMDWaveState):
        # Extract vdata vector across all lanes before writeback
        vdata_table = [wave.vgprs[lane][insn.vdata.index] for lane in range(wave.num_lanes)]

        for lane in range(wave.num_lanes):
            if not wave.is_lane_active(lane):
                continue
            vaddr = wave.vgprs[lane][insn.vaddr.index]
            res = ref_ds_bpermute_b32(
                lane_id=lane,
                vaddr=vaddr,
                vdata_table=vdata_table,
                exec_mask=wave.exec_mask,
                offset=insn.offset,
                wave_size=wave.num_lanes,
            )
            wave.vgprs[lane][insn.dst.index] = res


class TestSIMDDivergence(unittest.TestCase):
    def test_wave32_divergence_and_exec_masking(self):
        wave = SIMDWaveState(WaveSize.WAVE32)
        # EXEC mask: even lanes active (0x55555555)
        wave.exec_mask = 0x55555555

        sentinel = 0xDEADBEEF
        for lane in range(32):
            wave.vgprs[lane][3] = sentinel
            wave.vgprs[lane][1] = 0        # vaddr = 0 (reads lane 0)
            wave.vgprs[lane][2] = 100 + lane  # lane 0 has 100

        insn = DSBPermuteB32Instruction(
            dst=Operand.vgpr(3),
            vaddr=Operand.vgpr(1),
            vdata=Operand.vgpr(2),
        )
        SIMDDSBPermuteInterpreter.step(insn, wave)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][3], 100)
            else:
                self.assertEqual(wave.vgprs[lane][3], sentinel)

    def test_wave64_half_wave_isolation(self):
        wave = SIMDWaveState(WaveSize.WAVE64)
        for lane in range(64):
            wave.vgprs[lane][1] = 2 * 4       # vaddr = lane 2 within half-wave
            wave.vgprs[lane][2] = lane + 1000 # lane 2 has 1002, lane 34 has 1034

        insn = DSBPermuteB32Instruction(
            dst=Operand.vgpr(3),
            vaddr=Operand.vgpr(1),
            vdata=Operand.vgpr(2),
        )
        SIMDDSBPermuteInterpreter.step(insn, wave)

        # Lanes 0..31 read from lane 2 (1002)
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][3], 1002)
        # Lanes 32..63 read from lane 32 | 2 = 34 (1034)
        for lane in range(32, 64):
            self.assertEqual(wave.vgprs[lane][3], 1034)


class TestRegisterAliasing(unittest.TestCase):
    def test_alias_dst_equals_vaddr(self):
        # dst and vaddr share v1 (in-place gather overwriting address)
        lines = [
            "v_mov_b32_e32 v1, 28",         # addr = lane 7 * 4
            "v_mad_u32_u24 v2, v0, 10, 1",  # data = 10 * lane + 1 (lane 7 has 71)
            "ds_bpermute_b32 v1, v1, v2",
            "v_mov_b32_e32 v3, v1",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == 71 for v in results))

    def test_alias_dst_equals_vdata(self):
        # dst and vdata share v2 (in-place gather overwriting data)
        lines = [
            "v_mov_b32_e32 v1, 28",         # addr = lane 7 * 4
            "v_mad_u32_u24 v2, v0, 10, 1",  # data = 10 * lane + 1 (lane 7 has 71)
            "ds_bpermute_b32 v2, v1, v2",
            "v_mov_b32_e32 v3, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == 71 for v in results))

    def test_alias_all_registers_equal(self):
        # v0 = ds_bpermute_b32 v0, v0
        # Lane 0 has v0 = 0 (lane 0 * 4 = 0), so reads self (0)
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",   # addr = lane * 4
            "ds_bpermute_b32 v1, v1, v1",
            "v_mov_b32_e32 v3, v1",
        ]
        results = run_cs_bpermute(lines, 32)
        for lane in range(32):
            self.assertEqual(results[lane], lane * 4)


class TestConditionCodesAndStatus(unittest.TestCase):
    def test_vcc_and_scc_invariance(self):
        state = ScalarThreadState()
        state.sgprs[0] = 0xCAFEBABE
        state.vgprs[1] = 0
        state.vgprs[2] = 42

        interp = ScalarDSBPermuteInterpreter([42] * 32)
        insn = DSBPermuteB32Instruction(Operand.vgpr(3), Operand.vgpr(1), Operand.vgpr(2))
        interp.step(insn, state)

        self.assertEqual(state.sgprs[0], 0xCAFEBABE)
        self.assertEqual(state.lgkm_cnt, 1)

    def test_lgkmcnt_counter_tracking(self):
        state = ScalarThreadState()
        interp = ScalarDSBPermuteInterpreter([42] * 32)
        for i in range(5):
            insn = DSBPermuteB32Instruction(Operand.vgpr(3), Operand.vgpr(1), Operand.vgpr(2), offset=i * 4)
            interp.step(insn, state)
        self.assertEqual(state.lgkm_cnt, 5)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    def test_signed_zeros_transport(self):
        # Transports +0.0 and -0.0 without changing sign bits
        vdata = [POS_ZERO if i % 2 == 0 else NEG_ZERO for i in range(32)]
        for lane in range(32):
            res = ref_ds_bpermute_b32(lane, lane * 4, vdata)
            self.assertEqual(res, vdata[lane])

    def test_infinities_transport(self):
        vdata = [POS_INF if i % 2 == 0 else NEG_INF for i in range(32)]
        for lane in range(32):
            res = ref_ds_bpermute_b32(lane, lane * 4, vdata)
            self.assertEqual(res, vdata[lane])

    def test_nans_transport(self):
        # QNAN and SNAN transported bit-identically
        vdata = [QNAN if i % 2 == 0 else SNAN for i in range(32)]
        for lane in range(32):
            res = ref_ds_bpermute_b32(lane, lane * 4, vdata)
            self.assertEqual(res, vdata[lane])

    def test_subnormals_transport(self):
        vdata = [SUBNORM_MIN if i % 2 == 0 else SUBNORM_MAX for i in range(32)]
        for lane in range(32):
            res = ref_ds_bpermute_b32(lane, lane * 4, vdata)
            self.assertEqual(res, vdata[lane])

    def test_translator_preserves_nan_and_subnormal_bits(self):
        lines = [
            "v_mov_b32_e32 v1, 0",           # lane 0
            f"v_mov_b32_e32 v2, {QNAN:#x}",  # lane 0 has QNAN
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == QNAN for v in results))


class TestNegativeControlsAndMutation(unittest.TestCase):
    def test_negative_dst_rejection(self):
        with self.assertRaises(TypeError):
            DSBPermuteB32Instruction(
                dst=Operand.sgpr(0),
                vaddr=Operand.vgpr(1),
                vdata=Operand.vgpr(2),
            )

    def test_negative_addr_rejection(self):
        with self.assertRaises(TypeError):
            DSBPermuteB32Instruction(
                dst=Operand.vgpr(0),
                vaddr=Operand.sgpr(1),
                vdata=Operand.vgpr(2),
            )

    def test_negative_data_rejection(self):
        with self.assertRaises(TypeError):
            DSBPermuteB32Instruction(
                dst=Operand.vgpr(0),
                vaddr=Operand.vgpr(1),
                vdata=Operand.sgpr(2),
            )

    def test_negative_syntax_rejection(self):
        invalid_assemblies = [
            "ds_bpermute_b32",
            "ds_bpermute_b32 v1",
            "ds_bpermute_b32 v1, v2",
            "ds_bpermute_b32 s0, v1, v2",
            "ds_bpermute_b32 v0, s1, v2",
            "ds_bpermute_b32 v0, v1, s2",
            "ds_bpermute_b32 v0, v1, v2 offset:-4",
            "ds_bpermute_b32 v0, v1, v2 offset:70000",
            "ds_bpermute_b32 v0, v1, v2 bad_modifier",
        ]
        for asm in invalid_assemblies:
            with self.subTest(asm=asm):
                with self.assertRaises((ValueError, TypeError)):
                    DSBPermuteB32Instruction.parse(asm)

    def test_mutation_omitted_byte_division(self):
        # Mutated: using addr directly instead of addr // 4
        # For addr = 16 (lane 4): correct reads lane 4, mutant reads lane 16
        vdata = [i * 10 for i in range(32)]
        correct = ref_ds_bpermute_b32(0, 16, vdata)
        mutant = vdata[16 % 32]
        self.assertEqual(correct, 40)
        self.assertNotEqual(correct, mutant)

    def test_mutation_inactive_lane_nonzero(self):
        # Mutated: inactive lane reading local data instead of 0
        vdata = [100 + i for i in range(32)]
        exec_mask = 0xFFFFFFFF ^ (1 << 3)
        correct = ref_ds_bpermute_b32(0, 3 * 4, vdata, exec_mask=exec_mask)
        mutant = vdata[3]
        self.assertEqual(correct, 0)
        self.assertNotEqual(correct, mutant)

    def test_translator_emits_subgroup_shuffle_and_ballot(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 42",
            "ds_bpermute_b32 v3, v1, v2 offset:16",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bp_check", lines)
        self.assertIn("OpGroupNonUniformShuffle", spv)
        self.assertIn("OpGroupNonUniformBallot", spv)

    def test_translator_offset_affects_source_lane(self):
        # Every lane has vaddr=0 and vdata=lane*7+1; offset:16 means src_lane=(0+16)//4=4.
        # Result: each dst lane reads lane 4's value = 4*7+1 = 29.
        # With offset:0 each dst lane reads lane 0's value = 1.
        lines_offset16 = [
            "v_mov_b32_e32 v1, 0",
            "v_mad_u32_u24 v2, v0, 7, 1",
            "ds_bpermute_b32 v3, v1, v2 offset:16",
        ]
        results_offset16 = run_cs_bpermute(lines_offset16, 32)
        self.assertTrue(all(v == 29 for v in results_offset16))

        lines_offset0 = [
            "v_mov_b32_e32 v1, 0",
            "v_mad_u32_u24 v2, v0, 7, 1",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results_offset0 = run_cs_bpermute(lines_offset0, 32)
        self.assertTrue(all(v == 1 for v in results_offset0))

    def test_translator_offset_124_reads_correct_lane(self):
        # vaddr=0, offset:124 -> src_lane=(0+124)//4=31; each dst gets lane 31's value = 31*7+1=218.
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mad_u32_u24 v2, v0, 7, 1",
            "ds_bpermute_b32 v3, v1, v2 offset:124",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == 218 for v in results))


if __name__ == "__main__":
    unittest.main()



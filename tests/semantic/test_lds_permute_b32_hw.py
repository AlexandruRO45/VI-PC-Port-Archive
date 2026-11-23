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

100% of tests translate synthetic assembly via tools/rdna2spirv.py and execute via spirv_mini_interp.
Zero in-file mock models, zero game data, standard library Python only.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFFFFFF
OFFSET16_MASK: int = 0xFFFF
OLD_VDST: int = 0x5A5A1234
VSHARP: list[int] = [0, 0, 0x1000, 0x31016FAC]


def make_fssm(threads: int) -> bytes:
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


FSSM32 = make_fssm(32)
FSSM64 = make_fssm(64)


def calc_target_lane(vaddr: int, offset: int = 0) -> int:
    """Calculate target lane index within a 32-lane half."""
    eff = (vaddr + (offset & OFFSET16_MASK)) & MASK32
    return (eff >> 2) & 31


def run_permute_cs(name: str, lines: list[str], threads: int = 32) -> dict[int, int]:
    """Translate and run a compute shader across workgroup threads, returning memory map."""
    fssm = FSSM64 if threads == 64 else FSSM32
    prog = list(lines) + ["s_endpgm"]
    spv = support.translate_lines(name, prog, fssm=fssm)
    mem = si.run_workgroup(
        spv,
        threads,
        threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )
    return mem


class TestDsPermuteOracleAndModel(unittest.TestCase):
    """Part 1: Mathematical reference calculations and translator execution."""

    def test_target_lane_calculation_basic(self) -> None:
        self.assertEqual(calc_target_lane(0), 0)
        self.assertEqual(calc_target_lane(4), 1)
        self.assertEqual(calc_target_lane(8), 2)
        self.assertEqual(calc_target_lane(124), 31)
        self.assertEqual(calc_target_lane(128), 0)
        self.assertEqual(calc_target_lane(132), 1)

        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_lshlrev_b32_e32 v1, 2, v0",  # addr = 4 * lane -> lane maps to itself
            "v_add_nc_u32_e32 v2, 100, v0",  # data = 100 + lane
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_perm_calc_basic", lines, threads=32)
        for i in range(32):
            self.assertEqual(mem.get(4 * i), 100 + i)

    def test_target_lane_offset_addition(self) -> None:
        self.assertEqual(calc_target_lane(0, offset=4), 1)
        self.assertEqual(calc_target_lane(4, offset=8), 3)

        # offset:4 shifts destination by +1 lane
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 200, v0",
            "ds_permute_b32 v10, v1, v2 offset:4",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_perm_off4", lines, threads=32)
        self.assertEqual(mem.get(0), 231)  # lane 0 receives lane 31
        for i in range(1, 32):
            self.assertEqual(mem.get(4 * i), 200 + i - 1)

    def test_unwritten_lanes_zero_under_exec(self) -> None:
        # Only lane 0 is active in EXEC, sending to lane 5.
        # Lane 0 receives no writes -> 0.
        # Lane 5 is inactive in EXEC -> retains OLD_VDST.
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "s_mov_b32 exec_lo, 0x1",
            "v_mov_b32_e32 v1, 20",  # byte 20 -> lane 5
            "v_mov_b32_e32 v2, 0xABCD",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, 0xFFFFFFFF",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_perm_unwritten", lines, threads=32)
        self.assertEqual(mem.get(0), 0)  # active lane 0 received nothing -> 0
        self.assertEqual(mem.get(20), OLD_VDST)  # lane 5 inactive in EXEC -> kept OLD_VDST

    def test_collision_highest_writer_wins(self) -> None:
        # All 32 lanes write to lane 0 with data = 100 + lane.
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_mov_b32_e32 v1, 0",  # target lane 0
            "v_add_nc_u32_e32 v2, 100, v0",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_perm_collision", lines, threads=32)
        self.assertEqual(mem.get(0), 100 + 31)  # lane 31 is highest writer
        for d in range(1, 32):
            self.assertEqual(mem.get(4 * d), 0)  # unwritten lanes get 0

    def test_wave64_halves_isolation(self) -> None:
        lines = [
            "s_mov_b64 exec, -1",
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_mov_b32_e32 v1, 40",  # byte 40 -> lane 10 of each half
            "v_add_nc_u32_e32 v2, 1000, v0",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_perm_w64_iso", lines, threads=64)
        # Half 0: lane 10 gets highest writer of half 0 (lane 31 -> 1031)
        self.assertEqual(mem.get(40), 1031)
        # Half 1: lane 42 gets highest writer of half 1 (lane 63 -> 1063)
        self.assertEqual(mem.get(168), 1063)
        self.assertEqual(mem.get(0), 0)
        self.assertEqual(mem.get(128), 0)


class TestDsPermuteASTAndLowering(unittest.TestCase):
    """Part 2: Symbolic opcode translation and SPIR-V lowering."""

    def test_ast_spirv_instructions(self) -> None:
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, 0xABCDEF01",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_ast_perm", lines, fssm=FSSM32)
        self.assertTrue("OpGroupNonUniformBallot" in spv or "OpGroupNonUniformShuffle" in spv)
        mem = run_permute_cs("cs_ast_perm_exec", lines[:-1], threads=32)
        self.assertEqual(mem.get(0), 0xABCDEF01)

    def test_ast_with_offset(self) -> None:
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x12345678",
            "ds_permute_b32 v10, v1, v2 offset:16",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_ast_perm_off", lines, fssm=FSSM32)
        self.assertIn("OpIAdd", spv)
        mem = run_permute_cs("cs_ast_perm_off_exec", lines[:-1], threads=32)
        # offset:16 -> byte 16 -> lane 4
        self.assertEqual(mem.get(16), 0x12345678)


class TestDsPermuteSIMDExecution(unittest.TestCase):
    """Part 3: SIMD multi-lane divergence and execution masking."""

    def test_full_exec_wave32_rotation(self) -> None:
        # Each lane i sends to (i + 1) % 32
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_add_nc_u32_e32 v1, 1, v0",
            "v_and_b32_e32 v1, 31, v1",
            "v_lshlrev_b32_e32 v1, 2, v1",
            "v_add_nc_u32_e32 v2, 1000, v0",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_simd_rot", lines, threads=32)
        self.assertEqual(mem.get(0), 1000 + 31)
        for i in range(1, 32):
            self.assertEqual(mem.get(4 * i), 1000 + i - 1)

    def test_divergent_alternating_exec_even_active(self) -> None:
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "s_mov_b32 exec_lo, 0x55555555",  # Even lanes active (0, 2, 4...)
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 700, v0",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, 0xFFFFFFFF",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_simd_alt_exec", lines, threads=32)
        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(mem.get(4 * i), 700 + i)
            else:
                self.assertEqual(mem.get(4 * i), OLD_VDST)

    def test_source_gating_by_exec(self) -> None:
        # Lane 0 is inactive in EXEC, all other lanes active.
        # Lane i targets (i + 1) % 32. Lane 1 is active, but its only sender (lane 0) is inactive!
        # Destination lane 1 must receive 0 (unwritten active lane), not lane 0's data!
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_add_nc_u32_e32 v1, 1, v0",
            "v_and_b32_e32 v1, 31, v1",
            "v_lshlrev_b32_e32 v1, 2, v1",
            "v_add_nc_u32_e32 v2, 500, v0",
            "s_mov_b32 exec_lo, 0xFFFFFFFE",  # lane 0 inactive
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, 0xFFFFFFFF",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_simd_src_gating", lines, threads=32)
        self.assertEqual(mem.get(0), OLD_VDST)  # destination lane 0 was inactive in EXEC
        self.assertEqual(mem.get(4), 0)  # destination lane 1 active, but its source was inactive -> 0
        self.assertEqual(mem.get(8), 501)  # destination lane 2 received active lane 1's data

    def test_wave64_divergent_halves(self) -> None:
        lines = [
            "s_mov_b64 exec, -1",
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 2000, v0",
            # Mask upper half: low 16 of upper half active (lanes 32..47 active, 48..63 inactive)
            "s_mov_b32 exec_hi, 0x0000FFFF",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_w64_div", lines, threads=64)
        for i in range(32):
            self.assertEqual(mem.get(4 * i), 2000 + i)
        for i in range(32, 48):
            self.assertEqual(mem.get(4 * i), 2000 + i)
        for i in range(48, 64):
            self.assertEqual(mem.get(4 * i), OLD_VDST)


class TestDsPermuteRegisterAliasing(unittest.TestCase):
    """Part 4: Register aliasing and in-place operand preservation."""

    def test_aliasing_vdst_equals_vaddr(self) -> None:
        lines = [
            "v_add_nc_u32_e32 v1, 1, v0",
            "v_and_b32_e32 v1, 31, v1",
            "v_lshlrev_b32_e32 v1, 2, v1",  # v1 is addr AND dst
            "v_add_nc_u32_e32 v2, 500, v0",  # v2 is data
            "ds_permute_b32 v1, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v1, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_alias_dst_addr", lines, threads=32)
        self.assertEqual(mem.get(0), 500 + 31)
        for i in range(1, 32):
            self.assertEqual(mem.get(4 * i), 500 + i - 1)

    def test_aliasing_vdst_equals_vdata(self) -> None:
        lines = [
            "v_add_nc_u32_e32 v1, 1, v0",
            "v_and_b32_e32 v1, 31, v1",
            "v_lshlrev_b32_e32 v1, 2, v1",  # v1 is addr
            "v_add_nc_u32_e32 v2, 600, v0",  # v2 is data AND dst
            "ds_permute_b32 v2, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_alias_dst_data", lines, threads=32)
        self.assertEqual(mem.get(0), 600 + 31)
        for i in range(1, 32):
            self.assertEqual(mem.get(4 * i), 600 + i - 1)

    def test_aliasing_all_operands_same_register(self) -> None:
        lines = [
            "v_lshlrev_b32_e32 v5, 2, v0",  # v5 is addr, data, and dst
            "ds_permute_b32 v5, v5, v5",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v5, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_alias_all_same", lines, threads=32)
        for i in range(32):
            self.assertEqual(mem.get(4 * i), 4 * i)


class TestDsPermuteCollisionsAndUnwritten(unittest.TestCase):
    """Part 5: Collision resolution and unwritten lane semantics."""

    def test_all_lanes_target_single_lane(self) -> None:
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_mov_b32_e32 v1, 28",  # target lane 7
            "v_add_nc_u32_e32 v2, 100, v0",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_collision_lane7", lines, threads=32)
        self.assertEqual(mem.get(28), 100 + 31)  # lane 7 gets highest active writer (lane 31)
        for i in range(32):
            if i != 7:
                self.assertEqual(mem.get(4 * i), 0)

    def test_two_to_one_pairing_collision(self) -> None:
        # Lanes 2i and 2i+1 both target lane i. Odd writer 2i+1 is higher and wins.
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_lshrrev_b32_e32 v1, 1, v0",  # i = lane // 2
            "v_lshlrev_b32_e32 v1, 2, v1",  # byte address = i * 4
            "v_add_nc_u32_e32 v2, 1000, v0",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_collision_pairs", lines, threads=32)
        for i in range(16):
            self.assertEqual(mem.get(4 * i), 1000 + 2 * i + 1)
        for i in range(16, 32):
            self.assertEqual(mem.get(4 * i), 0)


class TestDsPermuteBoundaryOffsetsAndWrapping(unittest.TestCase):
    """Part 6: Boundary offsets and address wrapping extremes."""

    def test_max_aligned_offset_0xfffc(self) -> None:
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x8888",
            "ds_permute_b32 v10, v1, v2 offset:0xFFFC",  # 0xFFFC >> 2 = 16383 % 32 = 31
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_off_fffc", lines, threads=32)
        self.assertEqual(mem.get(124), 0x8888)

    def test_unaligned_subword_offsets(self) -> None:
        self.assertEqual(calc_target_lane(0, offset=1), 0)
        self.assertEqual(calc_target_lane(0, offset=2), 0)
        self.assertEqual(calc_target_lane(0, offset=3), 0)
        self.assertEqual(calc_target_lane(0, offset=4), 1)

    def test_address_wrapping_at_32bit_boundary(self) -> None:
        # 0xFFFFFFFF + 5 = 4 (modulo 2^32); 4 >> 2 = 1 -> lane 1
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_mov_b32_e32 v1, 0xFFFFFFFF",
            "v_mov_b32_e32 v2, 0x7777",
            "ds_permute_b32 v10, v1, v2 offset:5",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_wrap_32bit", lines, threads=32)
        self.assertEqual(mem.get(4), 0x7777)

    def test_discard_of_upper_address_bits(self) -> None:
        self.assertEqual(calc_target_lane(0x1000), 0)
        self.assertEqual(calc_target_lane(0x80000000), 0)


class TestDsPermuteNegativeControls(unittest.TestCase):
    """Part 7: Negative controls, syntax rejection, and mutation contrast."""

    def test_syntax_rejection_missing_operands(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("cs_bad_perm_ops", ["ds_permute_b32 v0, v1"], fssm=FSSM32)

    def test_syntax_rejection_invalid_opcode(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("cs_bad_perm_opc", ["v_permute_b32 v0, v1, v2"], fssm=FSSM32)

    def test_mutation_contrast_offset(self) -> None:
        lines_no_off = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x5555",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_mut_no_off", lines_no_off, threads=32)
        self.assertEqual(mem.get(0), 0x5555)  # writes lane 0
        self.assertNotEqual(mem.get(4), 0x5555)  # does not write lane 1 without offset:4

    def test_mutation_contrast_address_shift(self) -> None:
        # byte address 4 -> dword 1 (lane 1)
        lines = [
            f"v_mov_b32_e32 v10, {OLD_VDST}",
            "v_mov_b32_e32 v1, 4",
            "v_mov_b32_e32 v2, 0x6666",
            "ds_permute_b32 v10, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v10, v21, s[0:3], 0 offen",
        ]
        mem = run_permute_cs("cs_mut_shift", lines, threads=32)
        self.assertEqual(mem.get(4), 0x6666)  # lane 1
        self.assertNotEqual(mem.get(16), 0x6666)  # if shifted >> 3 would be lane 0, if not shifted lane 4


if __name__ == "__main__":
    unittest.main()

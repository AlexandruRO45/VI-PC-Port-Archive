from __future__ import annotations

"""Hardware and translator regression tests for RDNA2 ds_bpermute_b32.

Covers:
- Cross-lane backward permute (gather) execution through tools/rdna2spirv.py
- Subgroup shuffle, ballot, and bit extract lowering in SPIR-V
- Identity, broadcast, reverse, stride, and rotate gather patterns
- Offset addressing (offset:0, offset:16, offset:124)
- Register aliasing (vdst == vaddr, vdst == vdata, vdst == vaddr == vdata)
- Chained back-to-back permutations
- IEEE 754 bit-exact transport (signed zeros, infinities, quiet/signaling NaNs, subnormals)
- Translator syntax and malformed instruction rejections
- Mutation guards (byte division scaling, offset variation)

Standard library only: python -I, unittest, struct. Strictly zero game data.
"""

import os
import struct
import sys
from typing import Sequence
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


class TestDsBPermuteB32Execution(unittest.TestCase):
    def test_identity_permute(self):
        # Each lane gathers its own value: addr = lane * 4
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32 v2, 1000, v0",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        expected = [1000 + i for i in range(32)]
        self.assertEqual(results, expected)

    def test_broadcast_permute(self):
        # Every lane gathers from lane 7 (addr = 28)
        lines = [
            "v_mov_b32_e32 v1, 28",
            "v_mad_u32_u24 v2, v0, 100, 5",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        # Lane 7 had 7 * 100 + 5 = 705
        self.assertEqual(len(results), 32)
        self.assertTrue(all(v == 705 for v in results))

    def test_reverse_permute(self):
        # Lane i gathers from lane (31 - i): addr = (31 - i) * 4
        lines = [
            "v_sub_nc_u32 v1, 31, v0",
            "v_lshlrev_b32_e32 v1, 2, v1",
            "v_mul_lo_u32 v2, v0, 10",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        expected = [(31 - i) * 10 for i in range(32)]
        self.assertEqual(results, expected)

    def test_offset_addressing_16(self):
        # Base addr = 0, offset:16 -> reads lane (0 + 16) // 4 = 4
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mad_u32_u24 v2, v0, 7, 1",
            "ds_bpermute_b32 v3, v1, v2 offset:16",
        ]
        results = run_cs_bpermute(lines, 32)
        # Lane 4 has 4 * 7 + 1 = 29
        self.assertTrue(all(v == 29 for v in results))

    def test_offset_addressing_124(self):
        # Base addr = 0, offset:124 -> reads lane (0 + 124) // 4 = 31
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mad_u32_u24 v2, v0, 7, 1",
            "ds_bpermute_b32 v3, v1, v2 offset:124",
        ]
        results = run_cs_bpermute(lines, 32)
        # Lane 31 has 31 * 7 + 1 = 218
        self.assertTrue(all(v == 218 for v in results))

    def test_chained_permutations(self):
        # Step 1: gather from lane 3 -> lane 3 has data 16 (lane 4 * 4)
        # Step 2: use step 1's result as vaddr to gather from lane 4 -> gets lane 4's data (500)
        lines = [
            "v_mov_b32_e32 v1, 12",         # points to lane 3
            "v_mov_b32_e32 v2, 16",         # data: 16 (lane 4 * 4)
            "ds_bpermute_b32 v4, v1, v2",   # v4 gets 16
            "v_mov_b32_e32 v5, 500",        # data: 500
            "ds_bpermute_b32 v3, v4, v5",   # gathers using addr in v4
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == 500 for v in results))

    def test_rotate_permute(self):
        # Rotate right by 1: lane i gathers from lane (i - 1) & 31
        lines = [
            "v_sub_nc_u32 v1, v0, 1",
            "v_and_b32_e32 v1, 31, v1",
            "v_lshlrev_b32_e32 v1, 2, v1",
            "v_add_nc_u32 v2, 50, v0",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        expected = [50 + ((i - 1) & 31) for i in range(32)]
        self.assertEqual(results, expected)


class TestDsBPermuteB32RegisterAliasing(unittest.TestCase):
    def test_alias_dst_equals_vaddr(self):
        lines = [
            "v_mov_b32_e32 v1, 28",         # addr = lane 7 * 4
            "v_mad_u32_u24 v2, v0, 10, 1",  # data = 10 * lane + 1 (lane 7 has 71)
            "ds_bpermute_b32 v1, v1, v2",
            "v_mov_b32_e32 v3, v1",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == 71 for v in results))

    def test_alias_dst_equals_vdata(self):
        lines = [
            "v_mov_b32_e32 v1, 28",         # addr = lane 7 * 4
            "v_mad_u32_u24 v2, v0, 10, 1",  # data = 10 * lane + 1 (lane 7 has 71)
            "ds_bpermute_b32 v2, v1, v2",
            "v_mov_b32_e32 v3, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == 71 for v in results))

    def test_alias_all_registers_equal(self):
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",   # addr = lane * 4
            "ds_bpermute_b32 v1, v1, v1",
            "v_mov_b32_e32 v3, v1",
        ]
        results = run_cs_bpermute(lines, 32)
        for lane in range(32):
            self.assertEqual(results[lane], lane * 4)


class TestDsBPermuteB32SpirvLowering(unittest.TestCase):
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
        self.assertIn("OpGroupNonUniformBallotBitExtract", spv)

    def test_translator_emits_subgroup_capabilities(self):
        lines = [
            "ds_bpermute_b32 v0, v1, v2",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_bp_caps", lines, fssm=make_fssm(32))
        self.assertIn("OpCapability GroupNonUniformBallot", spv)
        self.assertIn("OpCapability GroupNonUniformShuffle", spv)


class TestDsBPermuteB32IEEE754Transport(unittest.TestCase):
    def test_pos_zero_transport(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            f"v_mov_b32_e32 v2, {POS_ZERO:#x}",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == POS_ZERO for v in results))

    def test_neg_zero_transport(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            f"v_mov_b32_e32 v2, {NEG_ZERO:#x}",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == NEG_ZERO for v in results))

    def test_pos_inf_transport(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            f"v_mov_b32_e32 v2, {POS_INF:#x}",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == POS_INF for v in results))

    def test_neg_inf_transport(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            f"v_mov_b32_e32 v2, {NEG_INF:#x}",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == NEG_INF for v in results))

    def test_qnan_transport(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            f"v_mov_b32_e32 v2, {QNAN:#x}",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == QNAN for v in results))

    def test_snan_transport(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            f"v_mov_b32_e32 v2, {SNAN:#x}",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == SNAN for v in results))

    def test_subnorm_min_transport(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            f"v_mov_b32_e32 v2, {SUBNORM_MIN:#x}",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == SUBNORM_MIN for v in results))

    def test_subnorm_max_transport(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            f"v_mov_b32_e32 v2, {SUBNORM_MAX:#x}",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertTrue(all(v == SUBNORM_MAX for v in results))


class TestDsBPermuteB32Rejections(unittest.TestCase):
    def test_syntax_rejection_bad_operands(self):
        invalid_assemblies = [
            ["ds_bpermute_b32"],
            ["ds_bpermute_b32 v1"],
            ["ds_bpermute_b32 v1, v2"],
            ["ds_bpermute_b32 s0, v1, v2"],
            ["ds_bpermute_b32 v0, s1, v2"],
            ["ds_bpermute_b32 v0, v1, s2"],
            ["ds_bpermute_b32 v0, v1, v2 offset:-4"],
            ["ds_bpermute_b32 v0, v1, v2 bad_modifier"],
        ]
        for idx, lines in enumerate(invalid_assemblies):
            with self.subTest(idx=idx, asm=lines[0]):
                with self.assertRaises(Exception):
                    support.translate_lines(f"bad_bp_{idx}", lines + ["s_endpgm"], fssm=make_fssm(32))


class TestDsBPermuteB32MutationGuards(unittest.TestCase):
    def test_mutation_omitted_byte_division(self):
        # Correct RDNA2 ISA: src_lane = ((vaddr + offset) // 4) % 32
        # For vaddr = 16: correct reads lane 4, mutant without // 4 would read lane 16.
        # We set each lane's data to lane * 10.
        lines = [
            "v_mov_b32_e32 v1, 16",
            "v_mul_lo_u32 v2, v0, 10",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        correct_lane_val = 4 * 10
        mutant_lane_val = 16 * 10
        self.assertTrue(all(v == correct_lane_val for v in results))
        self.assertFalse(any(v == mutant_lane_val for v in results))

    def test_mutation_offset_variation(self):
        # Offset 16 reads lane (0 + 16)//4 = 4 -> 29
        # Offset 0 reads lane (0 + 0)//4 = 0 -> 1
        lines_off16 = [
            "v_mov_b32_e32 v1, 0",
            "v_mad_u32_u24 v2, v0, 7, 1",
            "ds_bpermute_b32 v3, v1, v2 offset:16",
        ]
        res16 = run_cs_bpermute(lines_off16, 32)
        lines_off0 = [
            "v_mov_b32_e32 v1, 0",
            "v_mad_u32_u24 v2, v0, 7, 1",
            "ds_bpermute_b32 v3, v1, v2",
        ]
        res0 = run_cs_bpermute(lines_off0, 32)
        self.assertTrue(all(v == 29 for v in res16))
        self.assertTrue(all(v == 1 for v in res0))
        self.assertNotEqual(res16, res0)


if __name__ == "__main__":
    unittest.main()

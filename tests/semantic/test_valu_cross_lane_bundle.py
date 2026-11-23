from __future__ import annotations

"""Semantic translator verification suite for Bundle 54: VALU Cross-Lane Permutation & Data Exchange.

Covers RDNA2 Cross-Lane Instructions:
    - v_readfirstlane_b32 sD, vS
        Reads vector register from first active lane in EXEC into scalar destination.
    - v_readlane_b32 sD, vS, sLane
        Reads vector register from indexed lane into scalar destination (sLane mod 64).
    - v_writelane_b32 vD, sVal, sLane
        Writes scalar value into single indexed lane of vector register (EXEC-independent).
    - v_permlane16_b32 vD, vSrc0, sSrc1, sSrc2
        Permutes lanes within the same 16-lane half-row using 4-bit lane selectors.
    - v_permlanex16_b32 vD, vSrc0, sSrc1, sSrc2
        Permutes lanes from the opposing 16-lane half-row using 4-bit lane selectors.

Hardware rules (GFX10 RDNA2):
    - Subgroup cross-lane shuffles preserve dword bit patterns.
    - v_readfirstlane selects the first lane set in EXEC; if EXEC is 0, defaults to lane 0.
    - v_readlane and v_writelane index lanes modulo 64 (Wave64) or modulo 32 (Wave32).
    - v_writelane writes to the indexed lane regardless of whether that lane is active in EXEC.
    - v_permlane16 and v_permlanex16 decode 4-bit selectors from sSrc1 (lanes 0-7) and sSrc2 (lanes 8-15).

100% of tests translate synthetic shader listings through tools/rdna2spirv.py and execute
via spirv_mini_interp.run_workgroup, asserting computed memory results across workgroup lanes.
Zero in-file mock models, zero game data, standard library Python only.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 64, 1, 1)
FSSM = bytes(FSSM)

# VSHARP buffer descriptor occupies s[0:3]. Any temporary scalar registers in shaders MUST use s4+.
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def run_cs(name: str, lines: list[str], threads: int = 64) -> dict[int, int]:
    """Translate and run a 64-thread compute shader, returning memory map."""
    prog = list(lines) + ["s_endpgm"]
    spv = support.translate_lines(name, prog, fssm=FSSM)
    memory = si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})
    return memory


# ==============================================================================
# 1. v_readfirstlane_b32 Tests
# ==============================================================================

class TestReadFirstLane(unittest.TestCase):
    def test_active_first_lane(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v2, 700, v1",
            "v_readfirstlane_b32 s4, v2",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s4",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rfl_oracle", prog)
        self.assertEqual(mem.get(0), 700)

    def test_divergence_exec_mask_low(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mul_u32_u24 v2, v1, 7",
            "s_mov_b32 exec_lo, 0x0000FF00",
            "s_mov_b32 exec_hi, 0x00000000",
            "v_readfirstlane_b32 s4, v2",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s4",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rfl_ast", prog)
        self.assertEqual(mem.get(0), 8 * 7)

    def test_divergence_exec_mask_high(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v2, 50, v1",
            "s_mov_b32 exec_lo, 0x00000000",
            "s_mov_b32 exec_hi, 0x00010000",
            "v_readfirstlane_b32 s8, v2",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s8",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rfl_div", prog)
        # Lane 48 is the lowest active bit in exec_hi (bit 16 + 32 = 48)
        self.assertEqual(mem.get(0), 50 + 48)

    def test_aliasing_and_constant_propagation(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, 0x12345678",
            "v_readfirstlane_b32 s4, v2",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s4",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rfl_alias", prog)
        self.assertEqual(mem.get(0), 0x12345678)

    def test_spirv_instructions(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v3, 0xABCDEF01",
            "v_readfirstlane_b32 s10, v3",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s10",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        spv = support.translate_lines("cs_rfl_trans", prog + ["s_endpgm"], fssm=FSSM)
        self.assertTrue("OpGroupNonUniformShuffle" in spv or "OpGroupNonUniformBallot" in spv)
        mem = run_cs("cs_rfl_trans", prog)
        self.assertEqual(mem.get(0), 0xABCDEF01)


# ==============================================================================
# 2. v_readlane_b32 Tests
# ==============================================================================

class TestReadLane(unittest.TestCase):
    def test_indexing(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v2, 1000, v1",
            "s_mov_b32 s4, 4",
            "v_readlane_b32 s5, v2, s4",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s5",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rl_oracle", prog)
        self.assertEqual(mem.get(0), 1004)

    def test_indexing_arbitrary_lane(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mul_u32_u24 v2, v1, 3",
            "s_mov_b32 s4, 15",
            "v_readlane_b32 s5, v2, s4",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s5",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rl_ast", prog)
        self.assertEqual(mem.get(0), 45)

    def test_modulo_64(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v2, 800, v1",
            "s_mov_b32 s4, 67",  # 67 mod 64 = 3
            "v_readlane_b32 s5, v2, s4",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s5",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rl_mod", prog)
        self.assertEqual(mem.get(0), 803)

    def test_dest_aliasing_with_index(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v2, 55, v1",
            "s_mov_b32 s4, 7",
            "v_readlane_b32 s4, v2, s4",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s4",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rl_alias", prog)
        self.assertEqual(mem.get(0), 55 + 7)

    def test_mutation_guard(self):
        # Ensures that reading lane 5 yields lane 5 and not lane 0
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v5, 100, v1",
            "s_mov_b32 s5, 5",
            "v_readlane_b32 s6, v5, s5",
            "s_mov_b64 exec, -1",
            "v_cmpx_eq_u32_e32 0, v1",
            "v_mov_b32_e32 v20, s6",
            "v_mov_b32_e32 v21, 0",
            "buffer_store_dword v20, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_rl_trans", prog)
        self.assertEqual(mem.get(0), 105)
        self.assertNotEqual(mem.get(0), 100)


# ==============================================================================
# 3. v_writelane_b32 Tests
# ==============================================================================

class TestWriteLane(unittest.TestCase):
    def test_write_single_lane(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, 0",
            "s_mov_b32 s4, 999",
            "s_mov_b32 s5, 5",
            "v_writelane_b32 v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_wl_oracle", prog)
        self.assertEqual(mem.get(4 * 5), 999)
        self.assertEqual(mem.get(4 * 4), 0)
        self.assertEqual(mem.get(4 * 6), 0)

    def test_write_boundary_lane_31(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, 1",
            "s_mov_b32 s4, 0xCAFE",
            "s_mov_b32 s5, 31",
            "v_writelane_b32 v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_wl_ast", prog)
        self.assertEqual(mem.get(4 * 31), 0xCAFE)
        self.assertEqual(mem.get(4 * 30), 1)

    def test_exec_independence(self):
        # Even with EXEC=0, v_writelane writes to the targeted lane (68 mod 64 = 4)
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, 0",
            "s_mov_b32 exec_lo, 0x00000000",
            "s_mov_b32 exec_hi, 0x00000000",
            "s_mov_b32 s4, 777",
            "s_mov_b32 s5, 68",
            "v_writelane_b32 v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_wl_exec_ind", prog)
        self.assertEqual(mem.get(4 * 4), 777)
        self.assertEqual(mem.get(4 * 0), 0)

    def test_multiple_writes(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, 0",
            "s_mov_b32 s4, 10",
            "s_mov_b32 s5, 0",
            "v_writelane_b32 v2, s4, s5",
            "s_mov_b32 s4, 20",
            "s_mov_b32 s5, 1",
            "v_writelane_b32 v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_wl_alias", prog)
        self.assertEqual(mem.get(4 * 0), 10)
        self.assertEqual(mem.get(4 * 1), 20)
        self.assertEqual(mem.get(4 * 2), 0)

    def test_mutation_guard(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, v1",
            "s_mov_b32 s5, 0xBEEF",
            "s_mov_b32 s6, 12",
            "v_writelane_b32 v2, s5, s6",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_wl_trans", prog)
        self.assertEqual(mem.get(4 * 12), 0xBEEF)
        self.assertEqual(mem.get(4 * 0), 0)
        self.assertEqual(mem.get(4 * 11), 11)


# ==============================================================================
# 4. v_permlane16_b32 Tests
# ==============================================================================

class TestPermLane16(unittest.TestCase):
    def test_identity(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, v1",
            "s_mov_b32 s4, 0x76543210",
            "s_mov_b32 s5, 0xFEDCBA98",
            "v_permlane16_b32 v0, v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v0, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_p16_oracle", prog)
        for i in range(64):
            self.assertEqual(mem.get(4 * i), i)

    def test_constant_selector(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mul_u32_u24 v3, v1, 2",
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0",
            "v_permlane16_b32 v2, v3, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_p16_ast", prog)
        for i in range(16):
            self.assertEqual(mem.get(4 * i), 0)
        for i in range(16, 32):
            self.assertEqual(mem.get(4 * i), 32)

    def test_reverse_routing(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, v1",
            "s_mov_b32 s4, 0x01234567",
            "s_mov_b32 s5, 0x89ABCDEF",
            "v_permlane16_b32 v0, v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v0, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_p16_rout", prog)
        self.assertEqual(mem.get(4 * 0), 7)
        self.assertEqual(mem.get(4 * 7), 0)
        self.assertEqual(mem.get(4 * 8), 15)
        self.assertEqual(mem.get(4 * 15), 8)

    def test_aliasing(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v2, 100, v1",
            "s_mov_b32 s4, 0x11111111",
            "s_mov_b32 s5, 0x22222222",
            "v_permlane16_b32 v2, v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_p16_alias", prog)
        self.assertEqual(mem.get(4 * 0), 101)
        self.assertEqual(mem.get(4 * 8), 102)

    def test_mutation_guard(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v2, 50, v1",
            "s_mov_b32 s4, 0x76543210",
            "s_mov_b32 s5, 0xFEDCBA98",
            "v_permlane16_b32 v4, v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v4, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_p16_trans", prog)
        self.assertEqual(mem.get(4 * 0), 50)
        self.assertEqual(mem.get(4 * 15), 65)


# ==============================================================================
# 5. v_permlanex16_b32 Tests
# ==============================================================================

class TestPermLaneX16(unittest.TestCase):
    def test_cross_half_row_swap(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mov_b32_e32 v2, v1",
            "s_mov_b32 s4, 0x76543210",
            "s_mov_b32 s5, 0xFEDCBA98",
            "v_permlanex16_b32 v0, v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v0, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_px16_oracle", prog)
        self.assertEqual(mem.get(4 * 0), 16)
        self.assertEqual(mem.get(4 * 15), 31)
        self.assertEqual(mem.get(4 * 16), 0)
        self.assertEqual(mem.get(4 * 31), 15)

    def test_constant_selector(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v3, 100, v1",
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0",
            "v_permlanex16_b32 v2, v3, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_px16_ast", prog)
        for i in range(16):
            self.assertEqual(mem.get(4 * i), 116)
        for i in range(16, 32):
            self.assertEqual(mem.get(4 * i), 100)

    def test_routing(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_mul_u32_u24 v2, v1, 10",
            "s_mov_b32 s4, 0x11111111",
            "s_mov_b32 s5, 0x22222222",
            "v_permlanex16_b32 v0, v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v0, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_px16_rout", prog)
        self.assertEqual(mem.get(4 * 0), 170)
        self.assertEqual(mem.get(4 * 8), 180)
        self.assertEqual(mem.get(4 * 16), 10)
        self.assertEqual(mem.get(4 * 24), 20)

    def test_mutation_guard(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_add_nc_u32_e32 v2, 200, v1",
            "s_mov_b32 s4, 0x76543210",
            "s_mov_b32 s5, 0xFEDCBA98",
            "v_permlanex16_b32 v4, v2, s4, s5",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "buffer_store_dword v4, v21, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_px16_trans", prog)
        self.assertEqual(mem.get(4 * 0), 216)
        self.assertEqual(mem.get(4 * 16), 200)


if __name__ == "__main__":
    unittest.main()

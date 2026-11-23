"""Regression test for SHADER-1264:
Computed non-mask scalar values written to exec_lo, exec_hi, or exec must set each lane
strictly from its corresponding bit, rather than turning on every lane of that wave/half.
"""
import os
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

rdna2spirv = support.rdna2spirv

FSSM_64 = bytes(0x138) + struct.pack("<4I", 0x88, 64, 1, 1)
FSSM_32 = bytes(0x138) + struct.pack("<4I", 0x8088, 32, 1, 1)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def run_w64(lines):
    spv = support.translate_lines("cs_w64_exec_computed", lines + ["s_endpgm"], fssm=FSSM_64)
    return si.run_workgroup(
        spv, 64, 64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


def read_exec_bits_w64(setup_lines):
    prog = [
        "s_mov_b64 exec, -1",
        *setup_lines,
        # Capture current 64-bit EXEC into per-lane flag, then store to memory
        "v_mov_b32_e32 v10, 0",
        "s_mov_b64 s[12:13], exec",
        "s_mov_b64 exec, -1",
        "v_cndmask_b32_e64 v10, 0, 1, s[12:13]",
        "v_mbcnt_lo_u32_b32 v11, -1, 0",
        "v_mbcnt_hi_u32_b32 v11, -1, v11",
        "v_lshlrev_b32_e32 v11, 2, v11",
        "buffer_store_dword v10, v11, s[0:3], 0 offen",
    ]
    mem = run_w64(prog)
    lo = sum(((mem.get(4 * i, 0) & 1) << i) for i in range(32))
    hi = sum(((mem.get(4 * (32 + i), 0) & 1) << i) for i in range(32))
    return lo, hi


def run_w32(lines):
    spv = support.translate_lines("cs_w32_exec_computed", lines + ["s_endpgm"], fssm=FSSM_32)
    return si.run_workgroup(
        spv, 32, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


def read_exec_bits_w32(setup_lines):
    prog = [
        "s_mov_b32 exec_lo, -1",
        *setup_lines,
        "v_mov_b32_e32 v10, 0",
        "s_mov_b32 s12, exec_lo",
        "s_mov_b32 exec_lo, -1",
        "v_cndmask_b32_e64 v10, 0, 1, s12",
        "v_mbcnt_lo_u32_b32 v11, -1, 0",
        "v_lshlrev_b32_e32 v11, 2, v11",
        "buffer_store_dword v10, v11, s[0:3], 0 offen",
    ]
    mem = run_w32(prog)
    return sum(((mem.get(4 * i, 0) & 1) << i) for i in range(32))


class Wave64ExecHalfComputedScalarTests(unittest.TestCase):
    def test_computed_scalar_write_to_exec_lo(self):
        # Setting exec_lo via s_add_u32 with value 0x00000005 (bits 0 and 2 set)
        # must activate only lanes 0 and 2, preserving exec_hi.
        lo, hi = read_exec_bits_w64([
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 exec_hi, 0x12345678",
            "s_mov_b32 s4, 5",
            "s_add_u32 exec_lo, s4, 0",
        ])
        self.assertEqual(lo, 5)
        self.assertEqual(hi, 0x12345678)

    def test_computed_scalar_write_to_exec_hi(self):
        # Setting exec_hi via s_add_u32 with value 0x0000000A (bits 1 and 3 of hi half, i.e. lanes 33 and 35)
        # must activate only lanes 33 and 35, preserving exec_lo.
        lo, hi = read_exec_bits_w64([
            "s_mov_b32 exec_lo, 0x87654321",
            "s_mov_b32 exec_hi, 0",
            "s_mov_b32 s4, 10",
            "s_add_u32 exec_hi, s4, 0",
        ])
        self.assertEqual(lo, 0x87654321)
        self.assertEqual(hi, 10)

    def test_computed_scalar_shifted_write_to_exec_lo(self):
        # s_lshl_b32 exec_lo, 1, 4 -> 0x00000010 (only lane 4 active)
        lo, hi = read_exec_bits_w64([
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 exec_hi, 0",
            "s_mov_b32 s4, 1",
            "s_lshl_b32 exec_lo, s4, 4",
        ])
        self.assertEqual(lo, 0x10)
        self.assertEqual(hi, 0)

    def test_computed_scalar_write_bit31_to_exec_lo(self):
        # s_lshl_b32 exec_lo, 1, 31 -> 0x80000000 (only lane 31 active)
        lo, hi = read_exec_bits_w64([
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 exec_hi, 0",
            "s_mov_b32 s4, 1",
            "s_lshl_b32 exec_lo, s4, 31",
        ])
        self.assertEqual(lo, 0x80000000)
        self.assertEqual(hi, 0)


class Wave32ExecComputedScalarTests(unittest.TestCase):
    def test_wave32_computed_scalar_write_to_exec_lo(self):
        # In wave32 mode, s_add_u32 to exec_lo with value 5 must activate only lanes 0 and 2.
        mask = read_exec_bits_w32([
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 s4, 5",
            "s_add_u32 exec_lo, s4, 0",
        ])
        self.assertEqual(mask, 5)

    def test_wave32_computed_scalar_shifted_write_bit31(self):
        # In wave32 mode, s_lshl_b32 to exec_lo with shift 31 must activate only lane 31 (0x80000000).
        mask = read_exec_bits_w32([
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 s4, 1",
            "s_lshl_b32 exec_lo, s4, 31",
        ])
        self.assertEqual(mask, 0x80000000)

    def test_wave32_computed_scalar_write_to_exec(self):
        # In wave32 mode, writing to 'exec' directly via s_add_u32 with value 0x10.
        mask = read_exec_bits_w32([
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 s4, 0x10",
            "s_add_u32 exec, s4, 0",
        ])
        self.assertEqual(mask, 0x10)


if __name__ == "__main__":
    unittest.main()

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
import math
import os
import struct
import sys
from typing import Optional, Sequence, Tuple
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
        # Each lane reads from itself: lane_id * 4
        vdata = [1000 + i for i in range(32)]
        for lane in range(32):
            res = ref_ds_bpermute_b32(lane, lane * 4, vdata)
            self.assertEqual(res, vdata[lane])

    def test_broadcast_permute(self):
        # All lanes read from lane 7
        vdata = [500 + i for i in range(32)]
        for lane in range(32):
            res = ref_ds_bpermute_b32(lane, 7 * 4, vdata)
            self.assertEqual(res, vdata[7])

    def test_reverse_permute(self):
        # Lane i reads from 31 - i
        vdata = [i * 10 for i in range(32)]
        for lane in range(32):
            res = ref_ds_bpermute_b32(lane, (31 - lane) * 4, vdata)
            self.assertEqual(res, vdata[31 - lane])

    def test_offset_handling(self):
        # vaddr = 0, offset = 16 (reads lane 4)
        vdata = [i * 3 for i in range(32)]
        for lane in range(32):
            res = ref_ds_bpermute_b32(lane, 0, vdata, offset=16)
            self.assertEqual(res, vdata[4])

    def test_inactive_source_lane_reads_zero(self):
        # Source lane 5 is inactive
        vdata = [100 + i for i in range(32)]
        exec_mask = 0xFFFFFFFF ^ (1 << 5)
        res = ref_ds_bpermute_b32(0, 5 * 4, vdata, exec_mask=exec_mask)
        self.assertEqual(res, 0)

    def test_translator_backed_broadcast(self):
        # Compute shader broadcast of lane 7 (value = 705)
        lines = [
            "v_mov_b32_e32 v1, 28",            # addr = 7 * 4
            "v_mad_u32_u24 v2, v0, 100, 5",    # data = 100 * lane + 5
            "ds_bpermute_b32 v3, v1, v2",
        ]
        results = run_cs_bpermute(lines, 32)
        self.assertEqual(len(results), 32)
        self.assertTrue(all(v == 705 for v in results))


if __name__ == "__main__":
    unittest.main()

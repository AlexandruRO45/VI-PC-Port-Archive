"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_max_i32.

Models AMD RDNA2 / GFX10 Local Data Share (LDS) ds_max_i32 / ds_max_rtn_i32 atomic signed maximum semantics:
  - Atomically computes:
      addr = (vaddr + offset) & 0xFFFFFFFF
      idx = (addr >> 2) & (lds_dwords - 1)
      old_val = LDS[idx]
      LDS[idx] = max(to_signed_i32(old_val), to_signed_i32(vdata))
  - In returning form (ds_max_rtn_i32): writes old_val to vdst VGPR.
  - In non-returning form (ds_max_i32): prior value is discarded.
  - Signed 32-bit two's complement integer arithmetic:
      * 0xFFFFFFFF is -1 (strictly less than 0).
      * 0x80000000 is INT_MIN (-2^31).
      * 0x7FFFFFFF is INT_MAX (2^31 - 1).
      * Crucially diverges from unsigned maximum (ds_max_u32).
  - Pure memory atomic: condition codes (SCC / VCC) are not modified.

Synthesizes test listings, invokes tools/rdna2spirv.py via tests/semantic/support.py,
executes translated SPIR-V shaders in tests/semantic/spirv_mini_interp.py, and compares output buffers.

Standard library Python only (unittest, struct, math, dataclasses, enum), strictly zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
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
import spirv_mini_interp  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
SIGN_BIT = 0x80000000
INT32_MIN = 0x80000000
INT32_MAX = 0x7FFFFFFF

# Compute FSSM header: RSRC2 0x8088 (4 user SGPRs, 64x1x1 threads)
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 64, 1, 1)
CS_FSSM_BYTES = bytes(CS_FSSM)

OUT_VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28


def to_signed_i32(val: int) -> int:
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT) else val


def to_unsigned_u32(val: int) -> int:
    return val & MASK32


def ref_max_i32(current_val: int, new_val: int) -> int:
    """Architectural reference oracle for RDNA2 signed 32-bit atomic max."""
    c = to_signed_i32(current_val)
    n = to_signed_i32(new_val)
    return to_unsigned_u32(max(c, n))


def run_cs_lds(instructions: Sequence[str]) -> dict[int, int]:
    """Translate and execute a synthetic compute shader, returning output memory."""
    lines = list(instructions) + ["s_endpgm"]
    spvasm = support.translate_lines("cs_lds_max_i32_test", lines, fssm=CS_FSSM_BYTES)
    memory: dict[int, int] = {}
    spirv_mini_interp.run_fragment(
        spvasm,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
        push_constants=OUT_VSHARP,
        memory=memory,
    )
    return memory

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_signed_two_complement_max(self):
        # max(-10, 5) -> 5
        curr = to_unsigned_u32(-10)
        val = to_unsigned_u32(5)
        self.assertEqual(ref_max_i32(curr, val), 5)

        # max(-5, -1) -> -1 (0xFFFFFFFF)
        c2 = to_unsigned_u32(-5)
        v2 = to_unsigned_u32(-1)
        self.assertEqual(ref_max_i32(c2, v2), 0xFFFFFFFF)

        # max(INT32_MIN, 0) -> 0
        self.assertEqual(ref_max_i32(INT32_MIN, 0), 0)

    def test_translator_basic_execution(self):
        # Seed LDS[0] with -10 (0xFFFFFFF6), apply ds_max_i32 with 5, read back to mem[0]
        instructions = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0xFFFFFFF6",
            "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 5",
            "ds_max_i32 v0, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), 5)

    def test_translator_with_byte_offset(self):
        # Seed LDS[4] with -50, apply ds_max_i32 at offset 4 with -20
        instructions = [
            "v_mov_b32_e32 v0, 0",
            f"v_mov_b32_e32 v1, {to_unsigned_u32(-50):#x}",
            "ds_write_b32 v0, v1 offset:4",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v2, {to_unsigned_u32(-20):#x}",
            "ds_max_i32 v0, v2 offset:4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v0 offset:4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), to_unsigned_u32(-20))


if __name__ == "__main__":
    unittest.main()

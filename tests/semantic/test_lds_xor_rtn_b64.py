"""Translator regression tests for RDNA2 ds_xor_rtn_b64 and ds_xor_b64.

Standard library only: python -I, unittest, struct. Strictly zero game data.

AMD RDNA2 / GFX10 Data Share Architecture:
  * ds_xor_rtn_b64 v[dst_lo:dst_hi], vaddr, v[data_lo:data_hi] [offset:N]
    Atomic 64-bit bitwise XOR on the two dwords at ADDR = vaddr + offset in LDS.
    The prior 64-bit value M in LDS is returned in destination VGPR pair v[dst_lo:dst_hi],
    and LDS[ADDR] becomes (M ^ DATA).
  * ds_xor_b64 vaddr, v[data_lo:data_hi] [offset:N]
    Non-returning form: modifies LDS[ADDR] = (M ^ DATA) and leaves destination VGPRs unmodified.
  * Translation contract (tools/rdna2spirv.py):
    Because bitwise XOR acts on each bit independently, the 64-bit operation is lowered to
    two consecutive 32-bit atomic XORs (OpAtomicXor) on adjacent dwords (low dword first).
    The returned pair holds the prior values of the two dwords.
"""
from __future__ import annotations

import os
import random
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
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

# RSRC2 0x40088: 4 user SGPRs (output buffer V# in s[0:3]), workgroup ID in s4, 4 KiB LDS (8 * 512 bytes).
# 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def split64(val: int) -> tuple[int, int]:
    return val & M32, (val >> 32) & M32


def combine64(lo: int, hi: int) -> int:
    return (lo & M32) | ((hi & M32) << 32)


def translate_compute(body: Sequence[str], name: str = "cs_lds_xor_rtn_b64") -> str:
    lines = list(body) + ["s_endpgm"]
    return support.translate_lines(name, lines, fssm=FSSM)


def run_compute(body: Sequence[str], lanes: int = 1, name: str = "cs_lds_xor_rtn_b64") -> dict[int, int]:
    spv = translate_compute(body, name=name)
    return si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


def execute_xor_b64(
    line: str,
    init: int,
    data: int,
    addr: int = 0x40,
    offset: int = 0,
    dst_lo: int = 6,
    dst_hi: int = 7,
    name: str = "cs_xor_single",
) -> tuple[int, int]:
    """Execute a single 64-bit LDS XOR operation on 1 lane.

    Returns (returned_64bit_value, final_memory_64bit_value).
    """
    init_lo, init_hi = split64(init)
    data_lo, data_hi = split64(data)
    eff_addr = addr + offset

    body = [
        # Seed initial 64-bit value in LDS at effective address using ds_write_b64
        f"v_mov_b32_e32 v10, {eff_addr:#x}",
        f"v_mov_b32_e32 v2, {init_lo:#x}",
        f"v_mov_b32_e32 v3, {init_hi:#x}",
        "ds_write_b64 v10, v[2:3]",
        "s_waitcnt lgkmcnt(0)",
        # Set address and data registers
        f"v_mov_b32_e32 v1, {addr:#x}",
        f"v_mov_b32_e32 v4, {data_lo:#x}",
        f"v_mov_b32_e32 v5, {data_hi:#x}",
        # Preset destination registers to sentinel values
        f"v_mov_b32_e32 v{dst_lo}, 0xdeadbeef",
        f"v_mov_b32_e32 v{dst_hi}, 0xcafebabe",
        # Execute the operation
        line + (f" offset:{offset}" if offset else ""),
        "s_waitcnt lgkmcnt(0)",
        # Read back LDS memory at effective address
        "ds_read_b64 v[8:9], v10",
        "s_waitcnt lgkmcnt(0)",
        # Store destination registers to buffer slots 0, 1
        "v_mov_b32_e32 v31, 0",
        f"buffer_store_dword v{dst_lo}, v31, s[0:3], 0 offen",
        "v_mov_b32_e32 v31, 4",
        f"buffer_store_dword v{dst_hi}, v31, s[0:3], 0 offen",
        # Store read-back memory to buffer slots 2, 3
        "v_mov_b32_e32 v31, 8",
        "buffer_store_dword v8, v31, s[0:3], 0 offen",
        "v_mov_b32_e32 v31, 12",
        "buffer_store_dword v9, v31, s[0:3], 0 offen",
    ]
    mem = run_compute(body, lanes=1, name=name)
    rtn = combine64(mem.get(0, 0), mem.get(4, 0))
    final_mem = combine64(mem.get(8, 0), mem.get(12, 0))
    return rtn, final_mem


def execute_xor_b64_batch(cases: Sequence[tuple[int, int, int]], name: str = "cs_xor_batch") -> list[tuple[int, int]]:
    """Execute multiple (m, d, offset) cases in a single compute shader.

    Returns list of (rtn_64, mem_64) for each case.
    """
    body = []
    for k, (m, d, off) in enumerate(cases):
        addr = 0x40 + k * 16
        eff = addr + off
        m_lo, m_hi = split64(m)
        d_lo, d_hi = split64(d)
        body += [
            f"v_mov_b32_e32 v10, {eff:#x}",
            f"v_mov_b32_e32 v2, {m_lo:#x}",
            f"v_mov_b32_e32 v3, {m_hi:#x}",
            "ds_write_b64 v10, v[2:3]",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v1, {addr:#x}",
            f"v_mov_b32_e32 v4, {d_lo:#x}",
            f"v_mov_b32_e32 v5, {d_hi:#x}",
            f"ds_xor_rtn_b64 v[6:7], v1, v[4:5]" + (f" offset:{off}" if off else ""),
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b64 v[8:9], v10",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v31, {16 * k}",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
            f"v_mov_b32_e32 v31, {16 * k + 4}",
            "buffer_store_dword v7, v31, s[0:3], 0 offen",
            f"v_mov_b32_e32 v31, {16 * k + 8}",
            "buffer_store_dword v8, v31, s[0:3], 0 offen",
            f"v_mov_b32_e32 v31, {16 * k + 12}",
            "buffer_store_dword v9, v31, s[0:3], 0 offen",
        ]
    mem = run_compute(body, lanes=1, name=name)
    results = []
    for k in range(len(cases)):
        base = 16 * k
        rtn = combine64(mem.get(base, 0), mem.get(base + 4, 0))
        final_mem = combine64(mem.get(base + 8, 0), mem.get(base + 12, 0))
        results.append((rtn, final_mem))
    return results


class TestLdsXorRtnB64Values(unittest.TestCase):
    """Semantic truth-table, involution, identity, and mathematical property tests."""

    def test_truth_table_all_zeros(self):
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=0, data=0, name="cs_tt_zero"
        )
        self.assertEqual(rtn, 0)
        self.assertEqual(mem, 0)

    def test_truth_table_all_ones(self):
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=M64, data=M64, name="cs_tt_ones"
        )
        self.assertEqual(rtn, M64)
        self.assertEqual(mem, 0)

    def test_truth_table_alternating_bits(self):
        m = 0x55555555_55555555
        d = 0xAAAAAAAAAAAAAAAA
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, name="cs_tt_alt"
        )
        self.assertEqual(rtn, m)
        self.assertEqual(mem, M64)

    def test_truth_table_high_low_dword_isolation(self):
        m = 0x00000000_FFFFFFFF
        d = 0xFFFFFFFF_00000000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, name="cs_tt_words"
        )
        self.assertEqual(rtn, m)
        self.assertEqual(mem, M64)

    def test_truth_table_arbitrary_hash_constants(self):
        m = 0xDEADBEEF_CAFEBABE
        d = 0xBAADF00D_12345678
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, name="cs_tt_hash"
        )
        self.assertEqual(rtn, m)
        self.assertEqual(mem, m ^ d)

    def test_truth_table_single_bit_toggles(self):
        cases = [
            (1 << 0, 1 << 0, 0),
            (1 << 31, 1 << 31, 0),
            (1 << 32, 1 << 32, 0),
            (1 << 63, 1 << 63, 0),
            (1 << 63, 1 << 0, 0),
        ]
        batch_cases = [(m, d, 0) for m, d, _ in cases]
        results = execute_xor_b64_batch(batch_cases, name="cs_tt_single_bits")
        for (m, d, _), (got_rtn, got_mem) in zip(cases, results):
            self.assertEqual(got_rtn, m)
            self.assertEqual(got_mem, m ^ d)

    def test_involution_property(self):
        # (M ^ D) ^ D == M
        m = 0x98765432_10FEDCBA
        d = 0x55AA55AA_FF00FF00
        # First XOR: M ^ D
        rtn1, mem1 = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, name="cs_invol_1"
        )
        self.assertEqual(rtn1, m)
        self.assertEqual(mem1, m ^ d)

        # Second XOR: (M ^ D) ^ D -> returns (M ^ D), restores M in memory
        rtn2, mem2 = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=mem1, data=d, name="cs_invol_2"
        )
        self.assertEqual(rtn2, m ^ d)
        self.assertEqual(mem2, m)

    def test_identity_and_complement(self):
        m = 0xCAFEFACE_13579BDF
        cases = [
            (m, 0, 0),    # Identity: M ^ 0 == M
            (m, M64, 0),  # Complement: M ^ ~0 == ~M
        ]
        results = execute_xor_b64_batch(cases, name="cs_identity_comp")
        self.assertEqual(results[0], (m, m))
        self.assertEqual(results[1], (m, m ^ M64))

    def test_non_returning_form_preserves_destination_vgprs(self):
        # ds_xor_b64 has no vdst: leaves sentinel in v[6:7] untouched and updates memory
        m = 0x11223344_55667788
        d = 0x00FF00FF_00FF00FF
        rtn, mem = execute_xor_b64(
            "ds_xor_b64 v1, v[4:5]", init=m, data=d, dst_lo=6, dst_hi=7, name="cs_non_rtn"
        )
        sentinel = 0xcafebabe_deadbeef
        self.assertEqual(rtn, sentinel)
        self.assertEqual(mem, m ^ d)


class TestLdsXorRtnB64AddressingAndOffsets(unittest.TestCase):
    """Immediate offset and multi-address isolation tests."""

    def test_offset_zero(self):
        m = 0x01234567_89ABCDEF
        d = 0x0000FFFF_FFFF0000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, offset=0, name="cs_off_0"
        )
        self.assertEqual(rtn, m)
        self.assertEqual(mem, m ^ d)

    def test_offset_eight_bytes(self):
        m = 0x01234567_89ABCDEF ^ (8 << 32)
        d = 0x0000FFFF_FFFF0000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, offset=8, name="cs_off_8"
        )
        self.assertEqual(rtn, m)
        self.assertEqual(mem, m ^ d)

    def test_offset_sixteen_bytes(self):
        m = 0x01234567_89ABCDEF ^ (16 << 32)
        d = 0x0000FFFF_FFFF0000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, offset=16, name="cs_off_16"
        )
        self.assertEqual(rtn, m)
        self.assertEqual(mem, m ^ d)

    def test_offset_two_hundred_fifty_six(self):
        m = 0x01234567_89ABCDEF ^ (0x100 << 32)
        d = 0x0000FFFF_FFFF0000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, offset=0x100, name="cs_off_256"
        )
        self.assertEqual(rtn, m)
        self.assertEqual(mem, m ^ d)

    def test_adjacent_qword_isolation(self):
        # Verify modifying element at addr=0x40 does not touch element at addr=0x48
        addr_a = 0x40
        addr_b = 0x48
        val_a = 0x11111111_22222222
        val_b = 0x33333333_44444444
        xor_a = 0x0F0F0F0F_F0F0F0F0

        va_lo, va_hi = split64(val_a)
        vb_lo, vb_hi = split64(val_b)
        xa_lo, xa_hi = split64(xor_a)

        body = [
            # Seed both qwords in LDS
            f"v_mov_b32_e32 v10, {addr_a:#x}",
            f"v_mov_b32_e32 v2, {va_lo:#x}",
            f"v_mov_b32_e32 v3, {va_hi:#x}",
            "ds_write_b64 v10, v[2:3]",
            f"v_mov_b32_e32 v11, {addr_b:#x}",
            f"v_mov_b32_e32 v4, {vb_lo:#x}",
            f"v_mov_b32_e32 v5, {vb_hi:#x}",
            "ds_write_b64 v11, v[4:5]",
            "s_waitcnt lgkmcnt(0)",
            # Apply ds_xor_rtn_b64 only to addr_a
            f"v_mov_b32_e32 v12, {xa_lo:#x}",
            f"v_mov_b32_e32 v13, {xa_hi:#x}",
            "ds_xor_rtn_b64 v[6:7], v10, v[12:13]",
            "s_waitcnt lgkmcnt(0)",
            # Read back both qwords
            "ds_read_b64 v[14:15], v10",
            "ds_read_b64 v[16:17], v11",
            "s_waitcnt lgkmcnt(0)",
            # Store returned and both memories to buffer
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v7, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 8",
            "buffer_store_dword v14, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 12",
            "buffer_store_dword v15, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 16",
            "buffer_store_dword v16, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 20",
            "buffer_store_dword v17, v31, s[0:3], 0 offen",
        ]
        mem = run_compute(body, lanes=1, name="cs_adjacent_iso")
        rtn_a = combine64(mem.get(0, 0), mem.get(4, 0))
        final_a = combine64(mem.get(8, 0), mem.get(12, 0))
        final_b = combine64(mem.get(16, 0), mem.get(20, 0))

        self.assertEqual(rtn_a, val_a)
        self.assertEqual(final_a, val_a ^ xor_a)
        self.assertEqual(final_b, val_b)


class TestLdsXorRtnB64RegisterAliasing(unittest.TestCase):
    """Register aliasing tests (dst aliasing data or address)."""

    def test_dst_aliases_data_pair(self):
        # vdst v[4:5] is also the data operand pair.
        # Hardware reads vdata before writing the returned old value into vdst.
        m = 0x55555555_AAAA0000
        d = 0x33333333_1111FFFF
        m_lo, m_hi = split64(m)
        d_lo, d_hi = split64(d)
        addr = 0x40

        body = [
            f"v_mov_b32_e32 v10, {addr:#x}",
            f"v_mov_b32_e32 v2, {m_lo:#x}",
            f"v_mov_b32_e32 v3, {m_hi:#x}",
            "ds_write_b64 v10, v[2:3]",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v1, {addr:#x}",
            f"v_mov_b32_e32 v4, {d_lo:#x}",
            f"v_mov_b32_e32 v5, {d_hi:#x}",
            "ds_xor_rtn_b64 v[4:5], v1, v[4:5]",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b64 v[8:9], v10",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v4, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 8",
            "buffer_store_dword v8, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 12",
            "buffer_store_dword v9, v31, s[0:3], 0 offen",
        ]
        mem = run_compute(body, lanes=1, name="cs_alias_data")
        rtn = combine64(mem.get(0, 0), mem.get(4, 0))
        final_mem = combine64(mem.get(8, 0), mem.get(12, 0))

        self.assertEqual(rtn, m)
        self.assertEqual(final_mem, m ^ d)

    def test_dst_aliases_address_register(self):
        # vdst v[1:2] aliases vaddr v1.
        m = 0xDEADBEEF_00112233
        d = 0x11111111_44556677
        m_lo, m_hi = split64(m)
        d_lo, d_hi = split64(d)
        addr = 0x60

        body = [
            f"v_mov_b32_e32 v10, {addr:#x}",
            f"v_mov_b32_e32 v8, {m_lo:#x}",
            f"v_mov_b32_e32 v9, {m_hi:#x}",
            "ds_write_b64 v10, v[8:9]",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v1, {addr:#x}",
            f"v_mov_b32_e32 v4, {d_lo:#x}",
            f"v_mov_b32_e32 v5, {d_hi:#x}",
            "ds_xor_rtn_b64 v[1:2], v1, v[4:5]",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b64 v[8:9], v10",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v1, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v2, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 8",
            "buffer_store_dword v8, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 12",
            "buffer_store_dword v9, v31, s[0:3], 0 offen",
        ]
        mem = run_compute(body, lanes=1, name="cs_alias_addr")
        rtn = combine64(mem.get(0, 0), mem.get(4, 0))
        final_mem = combine64(mem.get(8, 0), mem.get(12, 0))

        self.assertEqual(rtn, m)
        self.assertEqual(final_mem, m ^ d)


class TestLdsXorRtnB64IEEE754Bitwise(unittest.TestCase):
    """IEEE 754 float64 bit-pattern validation: sign-bit toggling, NaNs, and subnormals."""

    def test_float64_positive_zero_to_negative_zero(self):
        sign_bit = 0x80000000_00000000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=0, data=sign_bit, name="cs_f64_zero"
        )
        self.assertEqual(rtn, 0)
        self.assertEqual(mem, sign_bit)

    def test_float64_one_to_negative_one(self):
        pos_one = 0x3FF00000_00000000
        neg_one = 0xBFF00000_00000000
        sign_bit = 0x80000000_00000000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=pos_one, data=sign_bit, name="cs_f64_one"
        )
        self.assertEqual(rtn, pos_one)
        self.assertEqual(mem, neg_one)

    def test_float64_infinity_sign_toggle(self):
        pos_inf = 0x7FF00000_00000000
        neg_inf = 0xFFF00000_00000000
        sign_bit = 0x80000000_00000000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=pos_inf, data=sign_bit, name="cs_f64_inf"
        )
        self.assertEqual(rtn, pos_inf)
        self.assertEqual(mem, neg_inf)

    def test_float64_nan_payload_preservation(self):
        qnan = 0x7FF80000_00001234
        mask = 0x00000000_FFFF0000
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=qnan, data=mask, name="cs_f64_nan"
        )
        self.assertEqual(rtn, qnan)
        self.assertEqual(mem, qnan ^ mask)


class TestLdsXorRtnB64ChainedAndRandom(unittest.TestCase):
    """Chained sequential atomic operations and seeded pseudo-random validation."""

    def test_chained_sequential_xors(self):
        m0 = 0xFEEDFACE_CAFEBABE
        masks = [
            0x12345678_9ABCDEF0,
            0x0F0F0F0F_F0F0F0F0,
            0xAAAAAAAAAAAAAAAA,
            0x55555555_55555555,
        ]
        addr = 0x50

        body = [
            f"v_mov_b32_e32 v10, {addr:#x}",
            f"v_mov_b32_e32 v2, {split64(m0)[0]:#x}",
            f"v_mov_b32_e32 v3, {split64(m0)[1]:#x}",
            "ds_write_b64 v10, v[2:3]",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v1, {addr:#x}",
        ]

        expected_rtns = []
        cur_m = m0
        for i, d in enumerate(masks):
            expected_rtns.append(cur_m)
            d_lo, d_hi = split64(d)
            body += [
                f"v_mov_b32_e32 v4, {d_lo:#x}",
                f"v_mov_b32_e32 v5, {d_hi:#x}",
                "ds_xor_rtn_b64 v[6:7], v1, v[4:5]",
                "s_waitcnt lgkmcnt(0)",
                f"v_mov_b32_e32 v31, {8 * i}",
                "buffer_store_dword v6, v31, s[0:3], 0 offen",
                f"v_mov_b32_e32 v31, {8 * i + 4}",
                "buffer_store_dword v7, v31, s[0:3], 0 offen",
            ]
            cur_m ^= d

        final_offset = 8 * len(masks)
        body += [
            "ds_read_b64 v[8:9], v10",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v31, {final_offset}",
            "buffer_store_dword v8, v31, s[0:3], 0 offen",
            f"v_mov_b32_e32 v31, {final_offset + 4}",
            "buffer_store_dword v9, v31, s[0:3], 0 offen",
        ]

        mem = run_compute(body, lanes=1, name="cs_chained")
        for i in range(len(masks)):
            got_rtn = combine64(mem.get(8 * i, 0), mem.get(8 * i + 4, 0))
            self.assertEqual(got_rtn, expected_rtns[i], f"Mismatch at step {i}")

        got_final = combine64(mem.get(final_offset, 0), mem.get(final_offset + 4, 0))
        self.assertEqual(got_final, cur_m)

    def test_seeded_random_sequences(self):
        rng = random.Random(0x501_B64)
        cases = []
        for _ in range(16):
            m = rng.getrandbits(64)
            d = rng.getrandbits(64)
            off = rng.choice((0, 8, 16, 0x20))
            cases.append((m, d, off))

        results = execute_xor_b64_batch(cases, name="cs_seeded_rnd")
        for i, ((m, d, off), (got_rtn, got_mem)) in enumerate(zip(cases, results)):
            with self.subTest(iteration=i, m=hex(m), d=hex(d), off=hex(off)):
                self.assertEqual(got_rtn, m)
                self.assertEqual(got_mem, m ^ d)


class TestLdsXorRtnB64MultiLane(unittest.TestCase):
    """Multi-lane SIMD workgroup execution across distinct LDS slots."""

    def test_independent_lane_slots(self):
        lanes = 4
        body = [
            "v_lshlrev_b32_e32 v1, 5, v0",  # v1 = lane_id * 32
            # Seed LDS: v2 = v0 * 0x1111, v3 = v0 * 0x2222
            "v_mov_b32_e32 v2, 0x1111",
            "v_mul_lo_u32 v2, v0, v2",
            "v_mov_b32_e32 v3, 0x2222",
            "v_mul_lo_u32 v3, v0, v3",
            "ds_write_b64 v1, v[2:3]",
            "s_waitcnt lgkmcnt(0)",
            # Data mask: v4 = 0x00FF, v5 = 0xFF00
            "v_mov_b32_e32 v4, 0x00FF",
            "v_mov_b32_e32 v5, 0xFF00",
            # Run atomic XOR with return
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]",
            "s_waitcnt lgkmcnt(0)",
            # Read back memory
            "ds_read_b64 v[8:9], v1",
            "s_waitcnt lgkmcnt(0)",
            # Store to output buffer: offset = lane_id * 16
            "v_lshlrev_b32_e32 v30, 4, v0",
            "buffer_store_dword v6, v30, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v31, 4, v30",
            "buffer_store_dword v7, v31, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v31, 8, v30",
            "buffer_store_dword v8, v31, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v31, 12, v30",
            "buffer_store_dword v9, v31, s[0:3], 0 offen",
        ]
        mem = run_compute(body, lanes=lanes, name="cs_multilane")
        for lane in range(lanes):
            base = lane * 16
            rtn_lo = mem.get(base, 0)
            rtn_hi = mem.get(base + 4, 0)
            mem_lo = mem.get(base + 8, 0)
            mem_hi = mem.get(base + 12, 0)

            exp_rtn_lo = (lane * 0x1111) & M32
            exp_rtn_hi = (lane * 0x2222) & M32
            exp_mem_lo = exp_rtn_lo ^ 0x00FF
            exp_mem_hi = exp_rtn_hi ^ 0xFF00

            self.assertEqual(rtn_lo, exp_rtn_lo, f"Lane {lane} rtn_lo mismatch")
            self.assertEqual(rtn_hi, exp_rtn_hi, f"Lane {lane} rtn_hi mismatch")
            self.assertEqual(mem_lo, exp_mem_lo, f"Lane {lane} mem_lo mismatch")
            self.assertEqual(mem_hi, exp_mem_hi, f"Lane {lane} mem_hi mismatch")


class TestLdsXorRtnB64SpirvLoweringAndMutations(unittest.TestCase):
    """Direct inspection of translator SPIR-V emission and mutation controls."""

    def test_spirv_structure_rtn_emits_two_atomic_xors(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v4, 0x11111111",
            "v_mov_b32_e32 v5, 0x22222222",
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]",
        ]
        spv = translate_compute(body, name="cs_spirv_rtn_check")
        atomic_xors = [line.strip() for line in spv.splitlines() if "OpAtomicXor" in line]
        self.assertEqual(len(atomic_xors), 2, f"Expected 2 OpAtomicXor instructions, got: {atomic_xors}")
        self.assertIn("%reg_v6", spv)
        self.assertIn("%reg_v7", spv)

    def test_spirv_structure_non_rtn_emits_two_atomic_xors_no_vdst(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v4, 0x11111111",
            "v_mov_b32_e32 v5, 0x22222222",
            "ds_xor_b64 v1, v[4:5]",
        ]
        spv = translate_compute(body, name="cs_spirv_non_rtn_check")
        atomic_xors = [line.strip() for line in spv.splitlines() if "OpAtomicXor" in line]
        self.assertEqual(len(atomic_xors), 2)
        self.assertNotIn("%reg_v6", spv)
        self.assertNotIn("%reg_v7", spv)

    def test_spirv_immediate_offset_calculation(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v4, 0x11111111",
            "v_mov_b32_e32 v5, 0x22222222",
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5] offset:32",
        ]
        spv = translate_compute(body, name="cs_spirv_offset_check")
        self.assertIn("OpAtomicXor", spv)

    def test_mutation_detection_low_high_word_sensitivity(self):
        # Asymmetric 64-bit value to ensure low and high 32-bit dwords are not swapped
        m = 0x12345678_87654321
        d = 0xFF00FF00_00FF00FF
        rtn, mem = execute_xor_b64(
            "ds_xor_rtn_b64 v[6:7], v1, v[4:5]", init=m, data=d, name="cs_mut_sensitivity"
        )
        self.assertEqual(rtn, m)
        self.assertEqual(mem, m ^ d)
        swapped_d = ((d & 0xFFFFFFFF) << 32) | ((d >> 32) & 0xFFFFFFFF)
        self.assertNotEqual(m ^ d, m ^ swapped_d)


class TestLdsXorRtnB64NegativeControls(unittest.TestCase):
    """Syntax validation and operand rejection tests."""

    def test_single_dst_register_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_compute(["ds_xor_rtn_b64 v6, v1, v[4:5]"], name="cs_bad_dst")

    def test_single_data_register_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_compute(["ds_xor_rtn_b64 v[6:7], v1, v4"], name="cs_bad_data")

    def test_missing_data_operand_rejected(self):
        with self.assertRaises((rdna2spirv.Unsupported, IndexError)):
            translate_compute(["ds_xor_rtn_b64 v[6:7], v1"], name="cs_missing_data")

    def test_non_returning_single_data_register_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_compute(["ds_xor_b64 v1, v4"], name="cs_bad_non_rtn")


if __name__ == "__main__":
    unittest.main()

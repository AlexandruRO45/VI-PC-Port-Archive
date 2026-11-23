"""Semantic coverage for tbuffer_load_format_d16_xyz under partial EXEC (wave64).

The load writes two destination VGPRs (v[4:5]): the xyz components pack into
v4 = x | (y << 16) and v5 = z | (old v5 high half). With only some lanes active,
active lanes must get the packed values (including the preserved high half of
v5) while inactive lanes keep both destination VGPRs untouched. Synthetic
inputs only: no GPU, no game files.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402
import test_value_semantics as tv  # noqa: E402

BASE = 0x1000
OUT_V = 0x31016FAC  # RAW output V# at base 0
SEED0, SEED1 = 0x12345678, 0xA9BCDEF0
FMT_16_16_16_16_FLOAT = 71

# 12 user SGPRs (s[0:11] from push constants), 64x1x1 threads.
CS_12_USER_FSSM = tv._fssm(12 << 1, (64, 1, 1))


def word3(fmt_code: int, dst_sel: int = 0xFAC, oob_select: int = 3) -> int:
    return (
        (dst_sel & 0xFFF)
        | ((fmt_code & 0x7F) << 12)
        | (1 << 24)
        | ((oob_select & 0x3) << 28)
    )


def f16_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def pack_u16(lo: int, hi: int) -> int:
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


class TbufferLoadD16XyzPartialExecTests(unittest.TestCase):
    def test_partial_exec_keeps_both_dest_vgprs_on_inactive_lanes(self):
        """Lanes 0 (low half) and 35 (high half) active; lanes 1 and 34 inactive.

        Active lanes load 16_16_16_16_FLOAT xyz: v4 = pack(x, y), v5 = pack(z, old
        high half of v5). Inactive lanes keep both seeds in v4 and v5.
        """
        hx, hy, hz = f16_bits(1.0), f16_bits(-1.0), f16_bits(2.0)
        lines = [
            # Enable only lane 0 (low half) and lane 35 (high half).
            "v_cmp_eq_u32_e64 s[12:13], 0, v0",
            "v_cmp_eq_u32_e64 s[14:15], 35, v0",
            "s_or_b64 s[12:13], s[12:13], s[14:15]",
            f"s_mov_b32 s4, {BASE:#x}",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 0x100",
            f"s_mov_b32 s7, {word3(FMT_16_16_16_16_FLOAT):#x}",
            "v_mov_b32 v1, 0",
            f"v_mov_b32 v4, {SEED0:#x}",
            f"v_mov_b32 v5, {SEED1:#x}",
            "s_and_saveexec_b64 s[16:17], s[12:13]",
            "tbuffer_load_format_d16_xyz v[4:5], v1, s[4:7], 0 format:[BUF_FMT_16_16_16_16_FLOAT] offen",
            "s_waitcnt vmcnt(0)",
            "s_mov_b64 exec, s[16:17]",
            # Store each lane's v[4:5] at lane * 8 in the output buffer.
            "v_lshlrev_b32 v2, 3, v0",
            "buffer_store_dwordx2 v[4:5], v2, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_tbuf_d16_xyz_partial_exec", lines, fssm=CS_12_USER_FSSM)
        mem = tv.ReadLog({
            BASE: pack_u16(hx, hy),
            BASE + 4: pack_u16(hz, f16_bits(3.0)),
        })
        pcs = [0, 0, 0x1000, OUT_V] + [0] * 28
        si.run_workgroup(
            spv,
            64,
            64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=pcs,
            memory=mem,
        )
        want_active = [pack_u16(hx, hy), pack_u16(hz, SEED1 >> 16)]
        for active_lane in (0, 35):
            base_off = active_lane * 8
            self.assertEqual(
                [dict.get(mem, base_off + 4 * i, 0) for i in range(2)],
                want_active,
                active_lane,
            )
        for inactive_lane in (1, 34):
            base_off = inactive_lane * 8
            self.assertEqual(
                [dict.get(mem, base_off + 4 * i, 0) for i in range(2)],
                [SEED0, SEED1],
                inactive_lane,
            )


if __name__ == "__main__":
    unittest.main()

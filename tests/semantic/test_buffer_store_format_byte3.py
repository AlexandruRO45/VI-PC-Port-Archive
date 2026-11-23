"""2-byte typed buffer stores (`buffer_store_format_*` and `tbuffer_store_format_*`) crossing a dword boundary.

When a 2-byte format element (such as BUF_FMT_16_UINT, BUF_FMT_16_FLOAT, or BUF_FMT_8_8_UINT) is stored at
byte offset 3 within a dword (`addr & 3 == 3`), byte 0 lands in bits 24..31 of the aligned dword and byte 1
lands in bits 0..7 of the next dword (`+4` bytes in linear mode, or `+step` bytes in scratch-swizzle mode).
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import spirv_asm          # noqa: E402
import spirv_mini_interp  # noqa: E402
import support            # noqa: E402


def make_fssm(rsrc2=0x8088, group=(64, 1, 1)):
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


CS_FSSM = make_fssm(0x8088, (64, 1, 1))
BASE = 0x1000
# V# word 3: RESOURCE_LEVEL=1 (bit 24), DST_SEL=XYZW (0xFAC), OOB_SELECT=3 (bits 29:28).
W3_RAW = 0x31000FAC
FMT_8_UINT = 5
FMT_16_UINT = 11
FMT_16_FLOAT = 13
FMT_8_8_UINT = 18


def vsharp_lines(base, num_records, fmt_code, *, stride=0, swizzle=False, tid=False, istride=0, oob=3,
                 dst_sel=0xFAC, runtime_fmt=False):
    w1 = (stride << 16) | ((1 if swizzle else 0) << 31)
    w3 = 0x01000000 | dst_sel | (fmt_code << 12) | (istride << 21) | ((1 if tid else 0) << 23) | (oob << 28)
    lines = [
        f"s_mov_b32 s4, {base:#x}",
        f"s_mov_b32 s5, {w1:#x}",
        f"s_mov_b32 s6, {num_records:#x}",
    ]
    if runtime_fmt:
        lines += [
            "s_mov_b32 s7, 0",
            f"s_mov_b32 s8, {w3:#x}",
            "s_or_b32 s7, s7, s8",
        ]
    else:
        lines.append(f"s_mov_b32 s7, {w3:#x}")
    return lines


def run_cs(name, lines, memory, *, lane=0):
    spv = support.translate_lines(name, lines, fssm=CS_FSSM)
    spirv_asm.assemble(spv)
    spirv_mini_interp.run_fragment(
        spv,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
        push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28,
        memory=memory,
    )
    return spv


class BufferStoreFormatByte3Tests(unittest.TestCase):
    def test_mubuf_16_uint_at_byte3_and_other_offsets(self):
        """buffer_store_format_x (16_UINT) writes both bytes at offset 3 and leaves dword 1 alone at offsets 0..2."""
        for off, exp0, exp1 in (
            (0, 0xAABB1234, 0xEEFF9988),
            (1, 0xAA1234DD, 0xEEFF9988),
            (2, 0x1234CCDD, 0xEEFF9988),
            (3, 0x34BBCCDD, 0xEEFF9912),
        ):
            with self.subTest(offset=off):
                mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
                lines = vsharp_lines(BASE, 0x40, FMT_16_UINT) + [
                    "v_mov_b32_e32 v4, 0x1234",
                    f"v_mov_b32_e32 v1, {off}",
                    "buffer_store_format_x v4, v1, s[4:7], 0 offen",
                    "s_endpgm",
                ]
                run_cs(f"cs_mubuf_16u_off{off}", lines, mem)
                self.assertEqual(f"{mem[BASE]:#010x}", f"{exp0:#010x}")
                self.assertEqual(f"{mem[BASE + 4]:#010x}", f"{exp1:#010x}")

    def test_mubuf_16_uint_at_offset7_and_runtime_vsharp(self):
        """Dynamic V# format (non-literal word 3) at offset 7 crosses from dword 1 to dword 2."""
        mem = {BASE + 4: 0xAABBCCDD, BASE + 8: 0xEEFF9988}
        lines = vsharp_lines(BASE, 0x40, FMT_16_UINT, runtime_fmt=True) + [
            "v_mov_b32_e32 v4, 0x5678",
            "v_mov_b32_e32 v1, 7",
            "buffer_store_format_x v4, v1, s[4:7], 0 offen",
            "s_endpgm",
        ]
        run_cs("cs_mubuf_16u_off7_rt", lines, mem)
        self.assertEqual(f"{mem[BASE + 4]:#010x}", "0x78bbccdd")
        self.assertEqual(f"{mem[BASE + 8]:#010x}", "0xeeff9956")

    def test_mubuf_16_float_and_8_8_uint_and_d16_at_byte3(self):
        """2-byte float, 2-component 8_8_UINT, and _d16_x/_d16_xy stores all cross dword boundary at byte 3."""
        # 16_FLOAT: 1.5 -> 0x3E00 (lo=0x00, hi=0x3E)
        mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
        lines = vsharp_lines(BASE, 0x40, FMT_16_FLOAT) + [
            "v_mov_b32_e32 v4, 0x3fc00000",
            "v_mov_b32_e32 v1, 3",
            "buffer_store_format_x v4, v1, s[4:7], 0 offen",
            "s_endpgm",
        ]
        run_cs("cs_mubuf_16f_off3", lines, mem)
        self.assertEqual(f"{mem[BASE]:#010x}", "0x00bbccdd")
        self.assertEqual(f"{mem[BASE + 4]:#010x}", "0xeeff993e")

        # 8_8_UINT: x=0x34, y=0x12 -> bytes [0x34, 0x12]
        # DST_SEL = XY01 (0x0000022C: X=4, Y=5, Z=0, W=1) so (x, y) stores (v4, v5)
        mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
        lines = vsharp_lines(BASE, 0x40, FMT_8_8_UINT, dst_sel=0x22C) + [
            "v_mov_b32_e32 v4, 0x34",
            "v_mov_b32_e32 v5, 0x12",
            "v_mov_b32_e32 v1, 3",
            "buffer_store_format_xy v[4:5], v1, s[4:7], 0 offen",
            "s_endpgm",
        ]
        run_cs("cs_mubuf_88u_off3", lines, mem)
        self.assertEqual(f"{mem[BASE]:#010x}", "0x34bbccdd")
        self.assertEqual(f"{mem[BASE + 4]:#010x}", "0xeeff9912")

        # buffer_store_format_d16_x with 16_UINT: low 16 bits of v4 = 0x1234
        mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
        lines = vsharp_lines(BASE, 0x40, FMT_16_UINT) + [
            "v_mov_b32_e32 v4, 0xdead1234",
            "v_mov_b32_e32 v1, 3",
            "buffer_store_format_d16_x v4, v1, s[4:7], 0 offen",
            "s_endpgm",
        ]
        run_cs("cs_mubuf_d16_16u_off3", lines, mem)
        self.assertEqual(f"{mem[BASE]:#010x}", "0x34bbccdd")
        self.assertEqual(f"{mem[BASE + 4]:#010x}", "0xeeff9912")

    def test_tbuffer_static_2byte_formats_at_byte3(self):
        """tbuffer_store_format_x/xy (static MTBUF path) writes both bytes across the dword boundary."""
        # BUF_FMT_16_UINT at byte 3
        mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
        lines = vsharp_lines(BASE, 0x40, 22) + [
            "v_mov_b32_e32 v4, 0x1234",
            "v_mov_b32_e32 v1, 3",
            "tbuffer_store_format_x v4, v1, s[4:7], 0 format:[BUF_FMT_16_UINT] offen",
            "s_endpgm",
        ]
        run_cs("cs_tbuf_16u_off3", lines, mem)
        self.assertEqual(f"{mem[BASE]:#010x}", "0x34bbccdd")
        self.assertEqual(f"{mem[BASE + 4]:#010x}", "0xeeff9912")

        # BUF_FMT_16_FLOAT at byte 3: -2.0f -> 0xC000 (lo=0x00, hi=0xC0)
        mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
        lines = vsharp_lines(BASE, 0x40, 22) + [
            "v_mov_b32_e32 v4, 0xc0000000",
            "v_mov_b32_e32 v1, 3",
            "tbuffer_store_format_x v4, v1, s[4:7], 0 format:[BUF_FMT_16_FLOAT] offen",
            "s_endpgm",
        ]
        run_cs("cs_tbuf_16f_off3", lines, mem)
        self.assertEqual(f"{mem[BASE]:#010x}", "0x00bbccdd")
        self.assertEqual(f"{mem[BASE + 4]:#010x}", "0xeeff99c0")

        # BUF_FMT_8_8_UINT at byte 3
        mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
        lines = vsharp_lines(BASE, 0x40, 22) + [
            "v_mov_b32_e32 v4, 0x77",
            "v_mov_b32_e32 v5, 0x55",
            "v_mov_b32_e32 v1, 3",
            "tbuffer_store_format_xy v[4:5], v1, s[4:7], 0 format:[BUF_FMT_8_8_UINT] offen",
            "s_endpgm",
        ]
        run_cs("cs_tbuf_88u_off3", lines, mem)
        self.assertEqual(f"{mem[BASE]:#010x}", "0x77bbccdd")
        self.assertEqual(f"{mem[BASE + 4]:#010x}", "0xeeff9955")

    def test_1byte_format_at_byte3_does_not_touch_next_dword(self):
        """1-byte formats at byte 3 must never touch dword 1 (both MUBUF and MTBUF)."""
        for name, fmt_code, store_ins in (
            ("cs_mubuf_8u_off3", FMT_8_UINT, "buffer_store_format_x v4, v1, s[4:7], 0 offen"),
            ("cs_tbuf_8u_off3", 22, "tbuffer_store_format_x v4, v1, s[4:7], 0 format:[BUF_FMT_8_UINT] offen"),
        ):
            with self.subTest(name=name):
                mem = {BASE: 0xAABBCCDD}
                lines = vsharp_lines(BASE, 4, fmt_code) + [
                    "v_mov_b32_e32 v4, 0x34",
                    "v_mov_b32_e32 v1, 3",
                    store_ins,
                    "s_endpgm",
                ]
                run_cs(name, lines, mem)
                self.assertEqual(f"{mem[BASE]:#010x}", "0x34bbccdd")
                self.assertNotIn(BASE + 4, mem)

    def test_store_then_load_roundtrip_at_byte3(self):
        """buffer_store_format_x and tbuffer_store_format_x round-trip identically through load_format_x at byte 3."""
        for name, fmt_code, store_ins, load_ins in (
            (
                "cs_rt_mubuf_16u_off3",
                FMT_16_UINT,
                "buffer_store_format_x v4, v1, s[4:7], 0 offen",
                "buffer_load_format_x v5, v1, s[4:7], 0 offen",
            ),
            (
                "cs_rt_tbuf_16u_off3",
                22,
                "tbuffer_store_format_x v4, v1, s[4:7], 0 format:[BUF_FMT_16_UINT] offen",
                "tbuffer_load_format_x v5, v1, s[4:7], 0 format:[BUF_FMT_16_UINT] offen",
            ),
        ):
            with self.subTest(name=name):
                mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
                lines = vsharp_lines(BASE, 0x40, fmt_code) + [
                    "v_mov_b32_e32 v4, 0x1234",
                    "v_mov_b32_e32 v1, 3",
                    store_ins,
                    load_ins,
                    "v_mov_b32_e32 v2, 0",
                    "buffer_store_dword v5, v2, s[0:3], 0 offen",
                    "s_endpgm",
                ]
                run_cs(name, lines, mem)
                self.assertEqual(f"{mem[0]:#010x}", "0x00001234")

    def test_bounds_at_byte3_preserve_oob_guard(self):
        """With num_records=4, a 2-byte store at offset 3 is OOB (neither dword touched); num_records=5 is in bounds."""
        for nr, exp0, exp1 in (
            (4, 0xAABBCCDD, 0xEEFF9988),
            (5, 0x34BBCCDD, 0xEEFF9912),
        ):
            for kind, fmt_code, ins in (
                ("mubuf", FMT_16_UINT, "buffer_store_format_x v4, v1, s[4:7], 0 offen"),
                ("tbuf", 22, "tbuffer_store_format_x v4, v1, s[4:7], 0 format:[BUF_FMT_16_UINT] offen"),
            ):
                with self.subTest(kind=kind, num_records=nr):
                    mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
                    lines = vsharp_lines(BASE, nr, fmt_code) + [
                        "v_mov_b32_e32 v4, 0x1234",
                        "v_mov_b32_e32 v1, 3",
                        ins,
                        "s_endpgm",
                    ]
                    run_cs(f"cs_oob_{kind}_nr{nr}", lines, mem)
                    self.assertEqual(f"{mem[BASE]:#010x}", f"{exp0:#010x}")
                    self.assertEqual(f"{mem[BASE + 4]:#010x}", f"{exp1:#010x}")

    def test_indexed_swizzle_and_inactive_lane_at_byte3(self):
        """Indexed swizzle (SWIZZLE_ENABLE=1, idxen=1) steps linearly by +4, and inactive EXEC lanes write nothing."""
        # INDEX_STRIDE=0 -> IS=8, stride=8, idx=5, off=3:
        #   buffer_offset = (5 // 8 * 8 + (3 & ~3)) * 8 + (5 % 8) * 4 + 3 = 20 + 3 = 23
        #   dword 0 is at BASE + 20, next dword is linearly +4 at BASE + 24 (matches buffer_load_format).
        dw0 = BASE + 20
        dw1 = dw0 + 4
        for kind, fmt_code, store_ins, load_ins in (
            (
                "mubuf",
                FMT_16_UINT,
                "buffer_store_format_x v4, v[1:2], s[4:7], 0 idxen offen",
                "buffer_load_format_x v5, v[1:2], s[4:7], 0 idxen offen",
            ),
            (
                "tbuf",
                22,
                "tbuffer_store_format_x v4, v[1:2], s[4:7], 0 format:[BUF_FMT_16_UINT] idxen offen",
                "tbuffer_load_format_x v5, v[1:2], s[4:7], 0 format:[BUF_FMT_16_UINT] idxen offen",
            ),
        ):
            with self.subTest(kind=kind):
                mem = {dw0: 0xAABBCCDD, dw1: 0xEEFF9988}
                lines = vsharp_lines(BASE, 0x40, fmt_code, stride=8, swizzle=True, tid=False, istride=0) + [
                    "v_mov_b32_e32 v4, 0x1234",
                    "v_mov_b32_e32 v1, 5",
                    "v_mov_b32_e32 v2, 3",
                    store_ins,
                    load_ins,
                    "v_mov_b32_e32 v3, 0",
                    "buffer_store_dword v5, v3, s[0:3], 0 offen",
                    "s_endpgm",
                ]
                run_cs(f"cs_idxswz_{kind}_16u_off3", lines, mem)
                self.assertEqual(f"{mem[dw0]:#010x}", "0x34bbccdd")
                self.assertEqual(f"{mem[dw1]:#010x}", "0xeeff9912")
                self.assertEqual(f"{mem[0]:#010x}", "0x00001234")

        # Inactive lane (EXEC = 0): neither dword 0 nor dword 1 is modified at byte 3.
        for kind, fmt_code, store_ins in (
            ("mubuf_inactive", FMT_16_UINT, "buffer_store_format_x v4, v1, s[4:7], 0 offen"),
            ("tbuf_inactive", 22, "tbuffer_store_format_x v4, v1, s[4:7], 0 format:[BUF_FMT_16_UINT] offen"),
        ):
            with self.subTest(kind=kind):
                mem = {BASE: 0xAABBCCDD, BASE + 4: 0xEEFF9988}
                lines = vsharp_lines(BASE, 0x40, fmt_code) + [
                    "v_mov_b32_e32 v4, 0x1234",
                    "v_mov_b32_e32 v1, 3",
                    "s_mov_b64 exec, 0",
                    store_ins,
                    "s_endpgm",
                ]
                run_cs(f"cs_{kind}_off3", lines, mem)
                self.assertEqual(f"{mem[BASE]:#010x}", "0xaabbccdd")
                self.assertEqual(f"{mem[BASE + 4]:#010x}", "0xeeff9988")


if __name__ == "__main__":
    unittest.main()

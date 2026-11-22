"""Semantic interpreter + spirv-val tests for RDNA2 v_sat_pk_u8_i16.

VOP1/VOP3 (opcode 0x062, VOP_I16_I32): S0 holds two signed 16-bit integers (S0[15:0].i16, S0[31:16].i16). Each is
clamped to [0, 255] and packed into a 16-bit result (res[7:0], res[15:8]), written to D[15:0] (or D[31:16] with VOP3
op_sel:[0,1]) while preserving the untouched half of D.
"""
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def s16(u: int) -> int:
    u &= 0xFFFF
    return u - 0x10000 if u & 0x8000 else u


def sat_u8(v: int) -> int:
    return 0 if v < 0 else (255 if v > 255 else v)


def ref_sat_pk_u8_i16(src: int, dst_old: int, dhi: bool = False) -> int:
    lo = sat_u8(s16(src & 0xFFFF))
    hi = sat_u8(s16((src >> 16) & 0xFFFF))
    pk = lo | (hi << 8)
    if dhi:
        return (dst_old & 0xFFFF) | (pk << 16)
    return (dst_old & 0xFFFF0000) | pk


EDGE_I16 = [-32768, -32767, -256, -255, -1, 0, 1, 127, 128, 254, 255, 256, 32767]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_sat_pk_u8.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sat_pk_u8.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    out_lines = list(lines) + [
        "v_and_b32 v5, 0xffff, v0",
        "v_lshrrev_b32 v6, 16, v0",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
    ]
    asm = "; test_sat_pk_u8\n" + "".join(f"\t{ln}\n" for ln in out_lines)
    asm += "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_lines(lines) -> int:
    spv = translate_lines(lines)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


class SatPkU8I16Tests(unittest.TestCase):
    def test_edge_values_and_high_half_preserved(self):
        dst_old = 0xCAFEBABE
        for i, a in enumerate(EDGE_I16):
            b = EDGE_I16[(i * 5 + 2) % len(EDGE_I16)]
            src = (a & 0xFFFF) | ((b & 0xFFFF) << 16)
            want = ref_sat_pk_u8_i16(src, dst_old, dhi=False)
            with self.subTest(a=a, b=b):
                got = run_lines([
                    f"v_mov_b32 v0, {dst_old:#x}",
                    f"v_mov_b32 v1, {src:#x}",
                    "v_sat_pk_u8_i16 v0, v1",
                ])
                self.assertEqual(got, want)

    def test_opsel_dst_hi_preserves_low_half(self):
        dst_old = 0x12345678
        src = ((-10) & 0xFFFF) | ((300 & 0xFFFF) << 16)  # lo -> 0x00, hi -> 0xFF => 0xFF00
        got = run_lines([
            f"v_mov_b32 v0, {dst_old:#x}",
            f"v_mov_b32 v1, {src:#x}",
            "v_sat_pk_u8_i16_e64 v0, v1 op_sel:[0,1]",
        ])
        self.assertEqual(got, ref_sat_pk_u8_i16(src, dst_old, dhi=True))
        self.assertEqual(got, 0xFF005678)

    def test_aliasing_sgpr_and_immediate(self):
        # Aliasing: v0 as both src and dst (lo write preserves original high half of v0)
        src = (42 & 0xFFFF) | ((200 & 0xFFFF) << 16)
        got = run_lines([
            f"v_mov_b32 v0, {src:#x}",
            "v_sat_pk_u8_i16 v0, v0",
        ])
        self.assertEqual(got, ref_sat_pk_u8_i16(src, src, dhi=False))
        # SGPR source
        got_s = run_lines([
            "v_mov_b32 v0, 0xabcd1234",
            "s_mov_b32 s0, 0x0105fffe",  # lo = -2 -> 0, hi = 261 -> 255 => 0xFF00
            "v_sat_pk_u8_i16 v0, s0",
        ])
        self.assertEqual(got_s, 0xABCDFF00)
        # Immediate source (-1 = 0xFFFFFFFF -> both halves -1 -> 0x0000)
        got_i = run_lines([
            "v_mov_b32 v0, 0xbeefcafe",
            "v_sat_pk_u8_i16 v0, -1",
        ])
        self.assertEqual(got_i, 0xBEEF0000)

    def test_seeded_cases(self):
        rng = random.Random(0x8160)
        for _ in range(40):
            src = rng.randrange(1 << 32)
            dst_old = rng.randrange(1 << 32)
            dhi = bool(rng.randrange(2))
            extra = " op_sel:[0,1]" if dhi else ""
            op = "v_sat_pk_u8_i16_e64" if dhi else "v_sat_pk_u8_i16"
            want = ref_sat_pk_u8_i16(src, dst_old, dhi=dhi)
            with self.subTest(src=hex(src), dst_old=hex(dst_old), dhi=dhi):
                got = run_lines([
                    f"v_mov_b32 v0, {dst_old:#x}",
                    f"v_mov_b32 v1, {src:#x}",
                    f"{op} v0, v1{extra}",
                ])
                self.assertEqual(got, want)

    def test_rejected_modifiers_and_spirv_val(self):
        for bad in (
            "v_sat_pk_u8_i16_e64 v0, v1 clamp",
            "v_sat_pk_u8_i16_e64 v0, v1 mul:2",
            "v_sat_pk_u8_i16_e64 v0, v1 op_sel:[1,0]",
            "v_sat_pk_u8_i16_e64 v0, -v1",
        ):
            with self.subTest(bad=bad), self.assertRaises(rdna2spirv.Unsupported):
                translate_lines([bad])
        if SPIRV_AS and SPIRV_VAL:
            spv = translate_lines([
                "v_mov_b32 v0, 0x11223344",
                "v_mov_b32 v1, 0x0100fff0",
                "v_sat_pk_u8_i16 v0, v1",
                "v_sat_pk_u8_i16_e64 v0, v0 op_sel:[0,1]",
            ])
            spv_bin = os.path.join(_TMP_DIR.name, "sat_pk_u8.spv")
            p1 = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", "-", "-o", spv_bin],
                                input=spv.encode("utf-8"), capture_output=True)
            self.assertEqual(p1.returncode, 0, p1.stderr.decode("utf-8", "replace"))
            p2 = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", spv_bin], capture_output=True)
            self.assertEqual(p2.returncode, 0, p2.stderr.decode("utf-8", "replace"))


if __name__ == "__main__":
    unittest.main()

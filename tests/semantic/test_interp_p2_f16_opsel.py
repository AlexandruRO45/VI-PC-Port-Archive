"""v_interp_p2_f16 VOP3 op_sel[3] (word 0 bit 14, 0x4000) writes vdst[31:16] and preserves vdst[15:0].

On RDNA2, VOP3 word 0 bit 14 on v_interp_p2_f16 (encoding 0xd75a....) selects the destination high 16-bit half
(hardware-measured on native gfx1036 in wave32 and wave64 across plain, high and clamp forms). llvm-mc drops bit 14
when disassembling v_interp_p2_f16 and re-encodes it as 0, causing check_printed_encodings to refuse the listing;
rdna2_decode prints op_sel:[0,0,0,1], which the translator used to ignore. Hand-built synthetic encodings only,
no game data.

    python -I tests/semantic/test_interp_p2_f16_opsel.py -v
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines, translate_listing, varyings  # noqa: E402

NAME = "ps_interp_p2_f16_opsel"
CODE_START = 0x100

# Barycentric weights of vertices 0, 1, 2 (AMD i = 0.25, j = 0.5).
BARY = (0.25, 0.25, 0.5)
LO, HI = (0.5, 1.5, 2.5), (-4.0, 8.0, 16.0)  # interpolated: 1.75 (0x3f00; clamped 1.0 = 0x3c00) and 9.0 (0x4880)


def pack(lo, hi):
    return struct.unpack("<I", struct.pack("<ee", lo, hi))[0]


ATTR0 = [(pack(lo, hi), 0, 0, 0) for lo, hi in zip(LO, HI)]

# Encodings assembled by llvm-mc -triple=amdgcn -mcpu=gfx1013 --show-encoding for a synthetic pixel shader.
P1_LO = ("v_interp_p1ll_f16 v2, v0, attr0.x", "02 00 42 d7 00 00 02 02")
P1_HI = ("v_interp_p1ll_f16 v3, v0, attr0.x high", "03 00 42 d7 00 01 02 02")
MOV_V4 = ("v_mov_b32_e32 v4, 0x5a5abebe", "ff 02 08 7e be be 5a 5a")
P2_V4_LO = ("v_interp_p2_f16 v4, v1, attr0.x, v2", "04 00 5a d7 00 02 0a 04")
P2_V4_HI_PRINTED = ("v_interp_p2_f16 v4, v1, attr0.x, v3 high", "04 00 5a d7 00 03 0e 04")
P2_V4_HI_STORED = ("v_interp_p2_f16 v4, v1, attr0.x, v3 high", "04 40 5a d7 00 03 0e 04")
MOV_V5 = ("v_mov_b32_e32 v5, 0x12345678", "ff 02 0a 7e 78 56 34 12")
P2_V5_CLAMP_PRINTED = ("v_interp_p2_f16 v5, v1, attr0.x, v2 clamp", "05 80 5a d7 00 02 0a 04")
P2_V5_CLAMP_STORED = ("v_interp_p2_f16 v5, v1, attr0.x, v2 clamp", "05 c0 5a d7 00 02 0a 04")
MOV_V6 = ("v_mov_b32_e32 v6, 0x9abcde01", "ff 02 0c 7e 01 de bc 9a")
P2_V6_PLAIN_PRINTED = ("v_interp_p2_f16 v6, v1, attr0.x, v2", "06 00 5a d7 00 02 0a 04")
P2_V6_PLAIN_STORED = ("v_interp_p2_f16 v6, v1, attr0.x, v2", "06 40 5a d7 00 02 0a 04")
EXP = ("exp mrt0, v4, v5, v6, v4 done vm", "0f 18 00 f8 04 05 06 04")
END = ("s_endpgm", "00 00 81 bf")

PRINTED = [
    P1_LO, P1_HI,
    MOV_V4, P2_V4_LO, P2_V4_HI_PRINTED,
    MOV_V5, P2_V5_CLAMP_PRINTED,
    MOV_V6, P2_V6_PLAIN_PRINTED,
    EXP, END,
]

STORED = [
    P1_LO, P1_HI,
    MOV_V4, P2_V4_LO, P2_V4_HI_STORED,
    MOV_V5, P2_V5_CLAMP_STORED,
    MOV_V6, P2_V6_PLAIN_STORED,
    EXP, END,
]


def listing(prog, invalid=False):
    size = sum(len(bytes.fromhex(enc)) for _, enc in prog)
    lines = [
        f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + size:x} (scan), gfx1013, "
        f"{1 if invalid else 0} invalid words"
    ]
    for text, enc in prog:
        printed = ",".join(f"0x{b:02x}" for b in bytes.fromhex(enc))
        lines.append(f"\t{text:<48}; encoding: [{printed}]")
    if invalid:
        lines += ["", "; --- llvm-mc messages ---", "; <stdin>:1:1: warning: invalid instruction encoding"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def blob(prog):
    return bytes(CODE_START) + b"".join(bytes.fromhex(enc) for _, enc in prog)


def run_spv(spv):
    return mrt0_words(spv, inputs=varyings(spv, {0: ATTR0}, BARY))


class InterpP2F16OpselTest(unittest.TestCase):
    def test_blob_restores_opsel_dst_for_plain_high_and_clamp(self):
        spv = translate_listing(NAME, listing(PRINTED), fssm=blob(STORED))
        words = run_spv(spv)
        # v4: low half 1.75 (0x3f00) from P2_V4_LO, high half 9.0 (0x4880) from P2_V4_HI_STORED
        # v5: low half kept (0x5678), high half clamped 1.0 (0x3c00) from P2_V5_CLAMP_STORED
        # v6: low half kept (0xde01), high half 1.75 (0x3f00) from P2_V6_PLAIN_STORED
        self.assertEqual([f"{w:#010x}" for w in words], ["0x48803f00", "0x3c005678", "0x3f00de01", "0x48803f00"])

        spirv_as = shutil.which("spirv-as")
        spirv_val = shutil.which("spirv-val")
        if spirv_as and spirv_val:
            with tempfile.TemporaryDirectory() as tmp:
                spvasm_path = os.path.join(tmp, "m.spvasm")
                spv_path = os.path.join(tmp, "m.spv")
                with open(spvasm_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(spv)
                subprocess.run([spirv_as, "--target-env", "vulkan1.2", spvasm_path, "-o", spv_path], check=True)
                subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv_path], check=True)

    def test_reanchored_listing_restores_opsel_dst_too(self):
        spv = translate_listing(NAME, listing(PRINTED, invalid=True), fssm=blob(STORED))
        words = run_spv(spv)
        self.assertEqual([f"{w:#010x}" for w in words], ["0x48803f00", "0x3c005678", "0x3f00de01", "0x48803f00"])

    def test_printed_zero_opsel_is_updated_to_high_half(self):
        printed_with_zero_opsel = list(PRINTED)
        printed_with_zero_opsel[8] = ("v_interp_p2_f16 v6, v1, attr0.x, v2 op_sel:[0,0,0,0]", "06 00 5a d7 00 02 0a 04")
        spv = translate_listing(NAME, listing(printed_with_zero_opsel), fssm=blob(STORED))
        words = run_spv(spv)
        self.assertEqual([f"{w:#010x}" for w in words], ["0x48803f00", "0x3c005678", "0x3f00de01", "0x48803f00"])

        # Also check on the re-anchored path.
        spv_re = translate_listing(NAME, listing(printed_with_zero_opsel, invalid=True), fssm=blob(STORED))
        words_re = run_spv(spv_re)
        self.assertEqual([f"{w:#010x}" for w in words_re], ["0x48803f00", "0x3c005678", "0x3f00de01", "0x48803f00"])

    def test_without_blob_llvm_text_writes_low_half(self):
        spv = translate_listing(NAME, listing(PRINTED), fssm=None)
        words = run_spv(spv)
        self.assertEqual([f"{w:#010x}" for w in words], ["0x5a5a4880", "0x12343c00", "0x9abc3f00", "0x5a5a4880"])

    def test_explicit_opsel_text_matches_blob_restoration(self):
        lines = [
            "v_interp_p1ll_f16 v2, v0, attr0.x",
            "v_interp_p1ll_f16 v3, v0, attr0.x high",
            "v_mov_b32_e32 v4, 0x5a5abebe",
            "v_interp_p2_f16 v4, v1, attr0.x, v2 op_sel:[0,0,0,0]",
            "v_interp_p2_f16 v4, v1, attr0.x, v3 high op_sel:[0,0,0,1]",
            "v_mov_b32_e32 v5, 0x12345678",
            "v_interp_p2_f16 v5, v1, attr0.x, v2 op_sel:[0,0,0,1] clamp",
            "v_mov_b32_e32 v6, 0x9abcde01",
            "v_interp_p2_f16 v6, v1, attr0.x, v2 op_sel:[0,0,0,1]",
            "exp mrt0, v4, v5, v6, v4 done vm",
            "s_endpgm",
        ]
        spv = translate_lines(NAME, lines)
        words = run_spv(spv)
        self.assertEqual([f"{w:#010x}" for w in words], ["0x48803f00", "0x3c005678", "0x3f00de01", "0x48803f00"])

    def test_source_opsel_on_interp_f16_is_refused(self):
        for bad_line in (
            "v_interp_p2_f16 v4, v1, attr0.x, v2 op_sel:[1,0,0,1]",
            "v_interp_p2_f16 v4, v1, attr0.x, v2 op_sel:[0,1,0,0]",
            "v_interp_p2_f16 v4, v1, attr0.x, v2 op_sel:[0,0,1,0]",
        ):
            lines = [
                "v_interp_p1ll_f16 v2, v0, attr0.x",
                bad_line,
                "exp mrt0, v4, v4, v4, v4 done vm",
                "s_endpgm",
            ]
            with self.assertRaises(rdna2spirv.Unsupported):
                translate_lines(NAME, lines)

        for p1_line in (
            "v_interp_p1ll_f16 v2, v0, attr0.x op_sel:[0,0,0,1]",
            "v_interp_p1lv_f16 v2, v0, attr0.x, v3 op_sel:[0,0,0,1]",
        ):
            lines = [
                p1_line,
                "v_interp_p2_f16 v4, v1, attr0.x, v2",
                "exp mrt0, v4, v4, v4, v4 done vm",
                "s_endpgm",
            ]
            with self.assertRaises(rdna2spirv.Unsupported):
                translate_lines(NAME, lines)

    def test_other_dropped_bits_on_interp_p2_f16_are_still_refused(self):
        # Bit 11 (op_sel[0] = 0x0800) set alongside bit 14: byte 1 = 0x48 vs printed 0x00.
        bad_opsel0 = list(STORED)
        bad_opsel0[8] = ("v_interp_p2_f16 v6, v1, attr0.x, v2", "06 48 5a d7 00 02 0a 04")
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate_listing(NAME, listing(PRINTED), fssm=blob(bad_opsel0))
        self.assertIn("dropped", str(cm.exception))

        # Different VGPR operand in the blob (v7 instead of v6) alongside bit 14.
        bad_vdst = list(STORED)
        bad_vdst[8] = ("v_interp_p2_f16 v6, v1, attr0.x, v2", "07 40 5a d7 00 02 0a 04")
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate_listing(NAME, listing(PRINTED), fssm=blob(bad_vdst))
        self.assertIn("dropped", str(cm.exception))

        # Bit 14 clear in blob (0x00) while set in printed encoding (0x40): never accepted in reverse.
        rev_printed = list(PRINTED)
        rev_printed[8] = P2_V6_PLAIN_STORED
        rev_stored = list(STORED)
        rev_stored[8] = P2_V6_PLAIN_PRINTED
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate_listing(NAME, listing(rev_printed), fssm=blob(rev_stored))
        self.assertIn("dropped", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

"""v_fma_mix_f32 with a 32-bit literal operand whose high half op_sel selects (RDNA2 VOP3P).

A VOP3P literal is one dword after the instruction. llvm-mc prints it as an f16 (its low half) and re-encodes the
high half as 0, so a listing alone loses the half that op_sel[i] = 1 reads. The translator takes the dword from the
code blob next to the listing (fssm/<name>.fssm) when only that half differs.

Synthetic shader: v_fma_mix_f32 v0, 0x3e00, 1.0, 0x3e00 op_sel:[0,0,1] op_sel_hi:[1,0,1] whose literal dword is
0xc1003e00: src0 = f16 low half 1.5, src1 = f32 1.0, src2 = f16 high half -2.5, so fma = 1.5 * 1.0 - 2.5 = -1.0.
Run with: python -I -m unittest discover -s tests/semantic -p test_valu_fma_mix_literal.py
"""
import os
import struct
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
F32 = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]  # noqa: E731


def head(src0=0xFF, src1=0xF2, src2=0xFF):
    """VOP3P v_fma_mix_f32 v0 with op_sel 0b100, op_sel_hi 0b101 and the three SRC fields given (0xff = literal,
    0xf2 = 1.0, 0x81 = inline integer 1, 0x80 = inline 0, 0 = s0)."""
    dword = 0x0BFDE4FF & ~((0x1FF) | (0x1FF << 9) | (0x1FF << 18)) | src0 | (src1 << 9) | (src2 << 18)
    return bytes.fromhex("006020cc") + struct.pack("<I", dword)


def run(literal, blob_literal, fields=(0xFF, 0xF2, 0xFF), args=None, blob_fields=None, invalid=False,
        op="v_fma_mix_f32"):
    """Translate the listing (which prints `literal`) against a blob holding `blob_literal`; return the export."""
    printed = head(*fields) + struct.pack("<I", literal)
    blob = head(*(blob_fields or fields)) + struct.pack("<I", blob_literal)
    exp_enc, end_enc = "0x0f,0x18,0x00,0xf8,0x00,0x00,0x00,0x00", "0x00,0x00,0x81,0xbf"
    blob += bytes.fromhex("0f1800f800000000000081bf")
    if invalid:   # a data dword after the code: the listing carries llvm-mc's warning, so every pc is re-anchored
        blob += b"\xff" * 4
    enc = ",".join(f"0x{b:02x}" for b in printed)
    low = f"0x{literal & 0xFFFF:x}"
    args = args or f"{low}, 1.0, {low}"
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        with open(os.path.join(tmp, "ps_mix_lit.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "ps_mix_lit.fssm"), "wb") as f:
            f.write(blob)
        src = os.path.join(tmp, "disasm", "ps_mix_lit.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(f"; ps_mix_lit  code 0x0..0x{len(blob):x} (header), gfx1013, {int(invalid)} invalid words\n")
            f.write(f"\t{op} v0, {args} op_sel:[0,0,1] op_sel_hi:[1,0,1] ; encoding: [{enc}]\n"
                    f"\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [{exp_enc}]\n"
                    f"\ts_endpgm ; encoding: [{end_enc}]\n")
            if invalid:
                f.write("; warning: invalid instruction encoding\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
    return si.run_fragment(spv)["%out_mrt0"][0]


class FmaMixLiteralHighHalf(unittest.TestCase):
    def test_high_half_comes_from_the_blob_literal(self):
        self.assertEqual(run(0x00003E00, 0xC1003E00), F32(-1.0))

    def test_high_half_zero_reads_plus_zero(self):
        self.assertEqual(run(0x00003E00, 0x00003E00), F32(1.5))

    def test_other_literal_difference_is_still_refused(self):
        # The low half is what llvm-mc printed; a different low half is a real mismatch, not a dropped field.
        with self.assertRaises(rdna2spirv.Unsupported):
            run(0x00003E00, 0xC1003C00)

    def test_inline_constant_equal_to_the_literal_low_half_is_left_alone(self):
        # SRC0 and SRC2 are the literal 0x40000001, SRC1 is the inline integer 1 (an f32 denormal): only the
        # SRC fields encoded 0xff take the dword. fma = f16(1) * 2^-149 + f16(0x4000) = 2.0 exactly.
        self.assertEqual(run(0x00000001, 0x40000001, fields=(0xFF, 0x81, 0xFF), args="0x1, 1, 0x1"), F32(2.0))

    def test_stored_s0_is_not_printed_inline_zero(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run(0x00003E00, 0xC1003E00, fields=(0xFF, 0x80, 0xFF), blob_fields=(0xFF, 0x00, 0xFF),
                args="0x3e00, 0, 0x3e00")

    def test_printed_high_half_must_be_zero(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run(0x12343E00, 0x40003E00)

    def test_only_the_mix_opcodes_take_the_repair(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run(0x00003E00, 0xC1003E00, op="v_fma_f32")

    def test_listing_with_undecodable_words_takes_the_blob_literal_too(self):
        self.assertEqual(run(0x00003E00, 0xC1003E00, invalid=True), F32(-1.0))


if __name__ == "__main__":
    unittest.main()

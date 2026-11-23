"""SDWA sext() sources: the sign-extended field is built with shifts, and no OpConvertUToF reads an
OpBitFieldSExtract.

NVIDIA's Vulkan driver (620.70.06 on an RTX 2060 SUPER) compiles OpConvertUToF of an OpBitFieldSExtract result as a
signed conversion. `v_cvt_f32_u32_sdwa v8, sext(v2) src0_sel:WORD_1` with v2 = 0xbb62b012 gave -17566.0 (0xc6893c00)
there instead of 4294949632.0 (0x4f7fffbb), the unsigned value of 0xffffbb62, and every BYTE_n/WORD_n select with a
negative field did the same. The CPU interpreter converts both forms right, so the value tests only show that the
shifts keep every select's bits; the shape test is the one that catches the extract feeding the conversion again
(docs/design.md, the SDWA sext() sources row).
"""
import re
import struct
import unittest

import support

SELECTS = {"BYTE_0": (0, 8), "BYTE_1": (8, 8), "BYTE_2": (16, 8), "BYTE_3": (24, 8),
           "WORD_0": (0, 16), "WORD_1": (16, 16)}
WORDS = (0xBB62B012, 0x00C5FF80, 0x8001FFFE, 0x7F7F7F7F, 0x80808080, 0x00000000, 0xFFFFFFFF)


def sext_field(word, sel):
    """The selected field of word, sign-extended to 32 bits."""
    off, width = SELECTS[sel]
    field = (word >> off) & ((1 << width) - 1)
    if field >> (width - 1):
        field -= 1 << width
    return field & 0xFFFFFFFF


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def words(op, word, sel):
    spv = support.translate_lines("ps_sdwa_sext", program(op, word, sel))
    return spv, support.mrt0_words(spv)


def program(op, word, sel):
    return [f"v_mov_b32 v2, 0x{word:08x}",
            f"{op}_sdwa v8, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:{sel}",
            "v_mov_b32 v9, v2", "v_mov_b32 v10, 0", "v_mov_b32 v11, 0",
            "exp mrt0 v8, v9, v10, v11 done vm", "s_endpgm"]


def opcodes(spvasm):
    """{result id: opcode} for every instruction of the module that has a result."""
    out = {}
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\S+) = (Op\w+)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


class SdwaSextSource(unittest.TestCase):
    def test_no_unsigned_conversion_of_a_field_extract(self):
        for sel in SELECTS:
            with self.subTest(sel=sel):
                spv, _ = words("v_cvt_f32_u32", 0xBB62B012, sel)
                defs = opcodes(spv)
                sources = re.findall(r"= OpConvertUToF %float (%\S+)", spv)
                self.assertTrue(sources, "the module has no OpConvertUToF")
                for src in sources:
                    self.assertNotEqual(defs.get(src), "OpBitFieldSExtract",
                                        f"{src} = OpBitFieldSExtract feeds OpConvertUToF (NVIDIA converts it as signed)")

    def test_sign_extended_bits(self):
        for sel in SELECTS:
            for word in WORDS:
                with self.subTest(sel=sel, word=f"0x{word:08x}"):
                    _, got = words("v_mov_b32", word, sel)
                    self.assertEqual(got[0], sext_field(word, sel))
                    self.assertEqual(got[1], word, "the source register is unchanged")

    def test_unsigned_conversion_of_the_sign_extended_field(self):
        for sel in SELECTS:
            for word in WORDS:
                with self.subTest(sel=sel, word=f"0x{word:08x}"):
                    _, got = words("v_cvt_f32_u32", word, sel)
                    self.assertEqual(got[0], f32_bits(float(sext_field(word, sel))))

    def test_reference_values(self):
        # the numbers in the docstring, worked by hand: sext(0xbb62) = 0xffffbb62 = 4294949730, which rounds to
        # 4294949632 = 0xffffbb * 2^8 in binary32
        self.assertEqual(sext_field(0xBB62B012, "WORD_1"), 0xFFFFBB62)
        self.assertEqual(f32_bits(float(0xFFFFBB62)), 0x4F7FFFBB)
        self.assertEqual(f32_bits(-17566.0), 0xC6893C00)


if __name__ == "__main__":
    unittest.main()

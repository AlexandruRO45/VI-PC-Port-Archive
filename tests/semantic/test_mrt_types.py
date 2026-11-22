"""EGEN-PS-MRT-TYPES: pixel-shader render-target outputs typed from the output signature, and compressed exports
decoded by what produced them.

On RDNA2 the export format follows the render target (measured with tools/ps_oracle.py --format on gfx1036): a
compressed (`compr`) export's 16-bit halves are f16 for an FP16_ABGR target, and reach a 16-bit UNORM, SNORM, UINT or
SINT target bit for bit. The translator now declares an SV_Target the OSG1 calls uint / sint as a uvec4 / ivec4
output (the halves zero- / sign-extended, an uncompressed export's dwords as they are), and decodes a float target's
compressed export as unorm16 / snorm16 when its registers come from v_cvt_pknorm_u16_f32 / v_cvt_pknorm_i16_f32,
in a form the UNORM16 / SNORM16 target converts back to the same bits. Hand constants are from the fixtures' headers
(runtime/test/shaders/ps_mrt_*.asm, ps_compr_*.asm)."""
import struct
import unittest

import support

EXPECTED_MRT_UINT = [0x00007FFF, 0x00008001, 0x00003C01, 0x0000C000]
EXPECTED_MRT_SINT = [0x00007FFF, 0xFFFF8001, 0x00003C01, 0xFFFFC000]
EXPECTED_UNORM16_TEXEL = [0xFFFF, 0x8000, 0x0000, 0x4000]       # RGBA16_UNORM channels the target stores
EXPECTED_SNORM16_TEXEL = [0x8001, 0x4000, 0x2000, 0xC000]       # RGBA16_SNORM channels


def osg1_dxbc(ctype):
    """A DXBC container whose OSG1 has SV_Target0 (system value 64) of component type ctype (1 uint, 2 sint, 3 float)."""
    payload = bytearray(struct.pack("<II", 1, 8))
    e = bytearray(32)
    struct.pack_into("<6I", e, 0, 0, 0, 0, 64, ctype, 0)
    e[24] = 0xF
    payload += e
    part = b"OSG1" + struct.pack("<I", len(payload)) + bytes(payload)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def to_unorm16(x):
    """What an R16_UNORM target stores for x (Vulkan: clamp to [0, 1], scale, round to nearest)."""
    return int(round(min(max(x, 0.0), 1.0) * 65535))


def to_snorm16(x):
    return int(round(min(max(x, -1.0), 1.0) * 32767)) & 0xFFFF


def reference_halves(words):
    """The fixtures' two exported registers as the four 16-bit channels (x, y from the first, z, w from the second)."""
    return [words[0] & 0xFFFF, words[0] >> 16, words[1] & 0xFFFF, words[1] >> 16]


def run(lines, ctype=None):
    spv = support.translate_lines("ps_mrt", lines + ["s_endpgm"], dxbc=osg1_dxbc(ctype) if ctype else support.STUB_DXBC)
    return support.mrt0_words(spv)


class TypedRenderTargets(unittest.TestCase):
    def test_reference(self):
        halves = reference_halves([0x80017FFF, 0xC0003C01])
        self.assertEqual(halves, EXPECTED_MRT_UINT)
        self.assertEqual([h - 0x10000 * (h >> 15) & 0xFFFFFFFF for h in halves], EXPECTED_MRT_SINT)
        self.assertEqual(reference_halves([0x8000FFFF, 0x40000000]), EXPECTED_UNORM16_TEXEL)
        self.assertEqual(reference_halves([0x40008001, 0xC0002000]), EXPECTED_SNORM16_TEXEL)
        # every unorm16 / snorm16 value survives the decode the translator emits and the target's conversion
        self.assertTrue(all(to_unorm16(f32(struct.unpack("<I", struct.pack("<f", u / 65535))[0])) == u
                            for u in range(65536)))
        # -32768 (0x8000) would come back as -32767, but v_cvt_pknorm_i16_f32 never writes it: it clamps to -1.0
        self.assertTrue(all(to_snorm16(max(s / 32767, -1.0)) == s & 0xFFFF for s in range(-32767, 32768)))

    def test_uint_target_gets_the_halves_zero_extended(self):
        spv = support.translate_fixture("ps_mrt_uint", dxbc=osg1_dxbc(1))
        self.assertEqual(support.mrt0_words(spv), EXPECTED_MRT_UINT)
        self.assertIn("OpTypePointer Output %v4uint", spv.replace("_out", ""))

    def test_sint_target_gets_the_halves_sign_extended(self):
        spv = support.translate_fixture("ps_mrt_sint", dxbc=osg1_dxbc(2))
        self.assertEqual(support.mrt0_words(spv), EXPECTED_MRT_SINT)

    def test_uncompressed_export_to_an_integer_target_is_its_bits(self):
        words = run(["v_mov_b32 v4, 0x12345678", "v_mov_b32 v5, -1", "exp mrt0 v4, v5, off, v4 done vm"], 1)
        self.assertEqual(words, [0x12345678, 0xFFFFFFFF, 0, 0x12345678])

    def test_pknorm_u16_halves_decode_as_unorm16(self):
        words = support.mrt0_words(support.translate_fixture("ps_compr_unorm16", dxbc=osg1_dxbc(3)))
        self.assertEqual([to_unorm16(f32(w)) for w in words], EXPECTED_UNORM16_TEXEL)

    def test_pknorm_i16_halves_decode_as_snorm16(self):
        words = support.mrt0_words(support.translate_fixture("ps_compr_snorm16", dxbc=osg1_dxbc(3)))
        self.assertEqual([to_snorm16(f32(w)) for w in words], EXPECTED_SNORM16_TEXEL)

    def test_the_producer_is_followed_through_a_copy_and_needs_no_signature(self):
        words = run(["v_cvt_pknorm_u16_f32 v6, 1.0, 0", "v_mov_b32 v2, v6", "exp mrt0 v2, v2, v2, v2 done compr vm"])
        self.assertEqual([to_unorm16(f32(w)) for w in words], [0xFFFF, 0, 0xFFFF, 0])

    def test_other_producers_keep_f16(self):
        words = run(["v_cvt_pkrtz_f16_f32 v2, 1.0, -2.0", "exp mrt0 v2, v2, v2, v2 done compr vm"], 3)
        self.assertEqual(words, [0x3F800000, 0xC0000000, 0x3F800000, 0xC0000000])

    def test_mixed_or_integer_halves_on_a_float_target_are_refused(self):
        for lines in (["v_cvt_pknorm_u16_f32 v2, 1.0, 0", "v_cvt_pkrtz_f16_f32 v3, 1.0, 0",
                       "exp mrt0 v2, v2, v3, v3 done compr vm"],
                      ["v_cvt_pknorm_u16_f32 v2, 1.0, 0", "v_cvt_pknorm_i16_f32 v2, 1.0, 0",
                       "exp mrt0 v2, v2, v2, v2 done compr vm"],
                      ["v_cvt_pk_u16_u32 v2, 1, 2", "exp mrt0 v2, v2, v2, v2 done compr vm"]):
            with self.subTest(lines=lines), self.assertRaisesRegex(Exception, "compressed export of"):
                run(lines, 3)


if __name__ == "__main__":
    unittest.main()

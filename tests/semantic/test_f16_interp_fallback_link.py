"""rdna2spirv.py --f16-interp-fallback: a PS linked to a producer that left out the halves it reads (TEST-398).

With the flag a VS exports each param's two f16 halves decoded to f32 at locations 28 - 2N and 29 - 2N, but drops
them when they would land on one of its own params (test_f16_interp_fallback.py). A PS that reads attr0's halves
(locations 28 and 29) then gets whatever that producer put there. Here the VS is linked to the PS by location, as
Vulkan links them, in the interpreter:
* with a producer that exports param0's halves, the PS reads them: 0.5 and 1.0, the interpolated f16 halves;
* with a producer that also exports param28, the halves are left out and the PS reads param28 at location 28
  (5.0) instead of attr0's low halves;
* the index's f16_halves tell the two pairs apart: the PS's entries are among the first producer's, not the
  second's.
"""
import re
import struct
import unittest

import support
import rdna2spirv
from spirv_mini_interp import run_fragment
from test_interp_f16_high import SHADER

BARY = (0.25, 0.25, 0.5)

# u, v from the vertex index (vertices 0, 1, 2 give (0, 0), (2, 0), (0, 2)); param0 = (f16(u) | f16(v) << 16, ...)
# and param1 = f32 (u, v, 0, 1). Interpolated with BARY: u = 0.5 and v = 1.0, both exact in f16.
VS = """v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v1, v2, v5, v6 done
v_cvt_pkrtz_f16_f32_e32 v7, v1, v2
v_cvt_pkrtz_f16_f32_e32 v8, v2, v1
exp param0 v7, v8, v5, v6
exp param1 v1, v2, v5, v6
"""
PARAM28 = "v_mov_b32_e32 v9, 0x40a00000\nexp param28 v9, v9, v9, v9\n"     # 5.0 in every channel
U, V, PARAM28_VALUE = 0.5, 1.0, 5.0


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def f16_bits(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def translate(name, text):
    old = rdna2spirv.F16_HALVES
    rdna2spirv.F16_HALVES = True
    try:
        spv = support.translate_lines(name, text.splitlines())
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.F16_HALVES = old
    return spv, meta


def locations(spv, prefix):
    return {name: int(loc) for name, loc in re.findall(rf"OpDecorate ({prefix}\S+) Location (\d+)", spv)}


def link(vs_spv, ps_spv):
    """The PS inputs Vulkan would hand over: every PS input takes the VS output at its location, from vertex 0 when
    Flat, else interpolated with BARY. A location the VS doesn't write reads as 0 here (undefined in Vulkan)."""
    by_location = {}
    outs = locations(vs_spv, "%out_")
    for vid in range(3):
        result = run_fragment(vs_spv, inputs={"%vertex_index": vid, "%instance_index": 0})
        for name, loc in outs.items():
            by_location.setdefault(loc, []).append([f32(w) for w in result[name]])
    inputs = {}
    for name, loc in locations(ps_spv, "%in_").items():
        verts = by_location.get(loc, [[0.0] * 4] * 3)
        if f"OpDecorate {name} Flat" in ps_spv:
            inputs[name] = tuple(verts[0])
        else:
            inputs[name] = tuple(sum(b * v[c] for b, v in zip(BARY, verts)) for c in range(4))
    return inputs


def halves_match(ps_meta, vs_meta):
    """The pairing the index allows: every decoded-half pair the PS reads is one the producer writes."""
    produced = vs_meta.get("f16_halves", {})
    return all(produced.get(n) == locs for n, locs in ps_meta.get("f16_halves", {}).items())


class FallbackLink(unittest.TestCase):
    def setUp(self):
        self.ps, self.ps_meta = translate("ps_f16_fallback", SHADER)

    def test_the_ps_reads_attr0s_halves_at_28_and_29(self):
        ps_in = locations(self.ps, "%in_")
        self.assertEqual((ps_in["%in_f16h0_0"], ps_in["%in_f16h0_1"]), (28, 29))
        self.assertEqual(self.ps_meta["f16_halves"], {"0": [28, 29]})

    def test_a_producer_with_the_halves(self):
        vs, vs_meta = translate("vs_f16_link", VS + "s_endpgm\n")
        self.assertEqual(vs_meta["f16_halves"], {"0": [28, 29], "1": [26, 27]})
        self.assertTrue(halves_match(self.ps_meta, vs_meta))
        words = support.mrt0_words(self.ps, inputs=link(vs, self.ps))
        self.assertEqual(words[0] & 0xFFFF, f16_bits(U))             # attr0.x low halves: 0.5
        self.assertEqual(words[1] & 0xFFFF, f16_bits(V))             # attr0.x high halves: 1.0
        self.assertEqual(f32(words[2]), U)                            # attr1.x as f32
        self.assertEqual(words[3], 0)                                 # attr0.y flat: vertex 0's pkrtz(0, 0)

    def test_a_producer_whose_param28_takes_the_halves_place(self):
        vs, vs_meta = translate("vs_f16_link_far", VS + PARAM28 + "s_endpgm\n")
        self.assertNotIn("%out_f16h0", vs)                            # param0's halves are left out
        self.assertEqual(locations(vs, "%out_")["%out_param28"], 28)
        self.assertNotIn("f16_halves", vs_meta)
        self.assertFalse(halves_match(self.ps_meta, vs_meta))         # the index tells the pairs apart
        words = support.mrt0_words(self.ps, inputs=link(vs, self.ps))
        self.assertEqual(words[0] & 0xFFFF, f16_bits(PARAM28_VALUE))  # the producer's param28, not attr0's 0.5
        self.assertNotEqual(words[0] & 0xFFFF, f16_bits(U))
        self.assertEqual(f32(words[2]), U)                            # the other reads are unchanged
        self.assertEqual(words[3], 0)

    def test_halves_match_needs_the_same_locations(self):
        self.assertTrue(halves_match({}, {}))
        self.assertTrue(halves_match({"f16_halves": {"0": [28, 29]}}, {"f16_halves": {"0": [28, 29], "1": [26, 27]}}))
        self.assertFalse(halves_match({"f16_halves": {"0": [28, 29]}}, {"f16_halves": {"1": [26, 27]}}))
        self.assertFalse(halves_match({"f16_halves": {"0": [28, 29]}}, {"f16_halves": {"0": [26, 27]}}))


if __name__ == "__main__":
    unittest.main()

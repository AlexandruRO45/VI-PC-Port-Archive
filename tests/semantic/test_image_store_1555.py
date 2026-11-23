"""image_store to a 1_5_5_5_UNORM T#: the texel also goes, packed, through the uint storage view.

Where VK_FORMAT_A1R5G5B5_UNORM_PACK16 has no storage support the D3D12 layer backs a B5G5R5A1_UNORM UAV texture
with a mutable R16_UINT image whose storage view is in the uint class, while its T# keeps FORMAT 134. The float
view of that index is then the 1x1 fallback, so the translator stores such a T# twice: the float texel through the
float view (the image itself where the GPU stores A1R5G5B5 natively) and the 16 bits through the uint view.

The fixture is executed in spirv_mini_interp with recording storage images, so the test sees which view of which
descriptor index each OpImageWrite reaches and the exact texel.
"""
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SEMANTIC = os.path.dirname(os.path.abspath(__file__))
if SEMANTIC not in sys.path:
    sys.path.insert(0, SEMANTIC)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

FIXTURE = Path(SEMANTIC) / "fixtures" / "image_store_1555.asm"
STORE = "image_store v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm"
FORMAT_1555, FORMAT_5551, FORMAT_565, FORMAT_RGBA8, FORMAT_R16_UINT = 134, 135, 133, 56, 11
BASE, INDEX = 0x1000, 2
SPIRV_AS, SPIRV_VAL = shutil.which("spirv-as"), shutil.which("spirv-val")


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def pack_1555(r, g, b, a):
    """The A1R5G5B5 texel of a float store: each channel clamped to [0, 1] (NaN is 0) and rounded to nearest."""
    def level(x, top):
        x = 0.0 if math.isnan(x) else min(max(x, 0.0), 1.0)
        return int(math.floor(struct.unpack("<f", struct.pack("<f", x))[0] * top + 0.5))
    return level(a, 1) << 15 | level(r, 31) << 10 | level(g, 31) << 5 | level(b, 31)


class Recorder:
    """A storage image that keeps what OpImageWrite stores: [(coordinate, texel), ...]."""

    def __init__(self):
        self.stores = []

    def store(self, coordinate, texel):
        self.stores.append((tuple(coordinate), tuple(texel)))


def lines(store=STORE):
    program = [ln for ln in FIXTURE.read_text(encoding="utf-8").splitlines() if not ln.startswith(";")]
    assert program.count(STORE) == 1
    return [store if ln == STORE else ln for ln in program]


def run(spv, fmt, words, bound=True):
    """Run the fixture once with a T# of FORMAT `fmt` and the four component words. Returns the recorders of the
    float, uint and sint views: [class][descriptor index]. `bound` False leaves the T# out of the hash table."""
    tsharp = [0x5551, fmt << 20, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66]
    memory = {BASE + 4 * i: w for i, w in enumerate(tsharp + list(words))}
    # The runtime's table: word 0 the slot mask, then 10-word slots (the key in words 1-8, the index in word 9).
    desc_map = [0, *(tsharp if bound else [0] * 8), INDEX, 0]
    views = [[Recorder() for _ in range(INDEX + 1)] for _ in range(3)]
    resources = {"%desc_map": desc_map, **{f"%stv{cls}_0": views[cls] for cls in range(3)}}
    si.run_workgroup(spv, 1, 1, inputs_for=lambda i: {"%sg_lane": 0}, push_constants=[BASE, 0], memory=memory,
                     resources=resources)
    return views


def validate(spv):
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as tmp:
        text, binary = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
        with open(text, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        for cmd in ([SPIRV_AS, text, "-o", binary, "--target-env", "vulkan1.2"],
                    [SPIRV_VAL, "--target-env", "vulkan1.2", binary]):
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode:
                raise AssertionError(f"{cmd[0]} failed:\n{done.stderr}")


class ImageStore1555Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spv = support.translate_lines("ps_image_store_1555", lines())
        validate(cls.spv)

    def stores(self, fmt, components, bound=True):
        return run(self.spv, fmt, [fbits(c) for c in components], bound)

    def check_packed(self, components, want=None):
        views = self.stores(FORMAT_1555, components)
        want = pack_1555(*components) if want is None else want
        self.assertEqual(views[1][INDEX].stores, [((5, 3), (want, 0, 0, 0))], components)
        return views

    def test_packs_the_texel_through_the_uint_view(self):
        # The values the D3D12 layer's render-target clears read back as (d3d12_b5g5r5a1_uav_test): A is bit 15,
        # R bits 14:10, G 9:5, B 4:0.
        for components, want in (((1.0, 0.0, 0.0, 1.0), 0xFC00), ((0.0, 1.0, 0.0, 0.0), 0x03E0),
                                 ((0.0, 0.0, 1.0, 0.0), 0x001F), ((0.0, 0.0, 0.0, 1.0), 0x8000),
                                 ((1.0, 1.0, 1.0, 1.0), 0xFFFF), ((0.0, 0.0, 0.0, 0.0), 0x0000)):
            with self.subTest(components=components):
                views = self.check_packed(components, want)
                self.assertEqual(want, pack_1555(*components))
                # The float view of the same index gets the texel as written: the image itself where the GPU stores
                # A1R5G5B5, the fallback where the runtime bound the R16_UINT view.
                self.assertEqual(views[0][INDEX].stores, [((5, 3), components)])
                self.assertEqual(views[2][INDEX].stores, [])
                for cls in range(3):
                    self.assertEqual(views[cls][0].stores + views[cls][1].stores, [])

    def test_every_level_of_every_channel(self):
        for channel, (top, shift) in enumerate(((31, 10), (31, 5), (31, 0), (1, 15))):
            for level in range(top + 1):
                components = [0.0] * 4
                components[channel] = level / top
                with self.subTest(channel=channel, level=level):
                    self.check_packed(tuple(components), level << shift)

    def test_rounds_to_nearest(self):
        for level in (0, 1, 15, 29, 30):
            for part, want in ((0.25, level), (0.49, level), (0.51, level + 1), (0.75, level + 1)):
                x = (level + part) / 31
                with self.subTest(level=level, part=part):
                    self.check_packed((x, 0.0, x, 0.0), want << 10 | want)
                    self.check_packed((0.0, x, 0.0, 0.0), want << 5)
        for alpha, want in ((0.25, 0), (0.49, 0), (0.5, 1), (0.51, 1), (0.75, 1)):
            with self.subTest(alpha=alpha):
                self.check_packed((0.0, 0.0, 0.0, alpha), want << 15)

    def test_clamps_to_zero_and_one(self):
        for x, want in ((-0.0, 0), (-1e-3, 0), (-1.0, 0), (-math.inf, 0), (math.nan, 0),
                        (1.0 + 1e-3, 1), (2.0, 1), (1e30, 1), (math.inf, 1)):
            with self.subTest(x=x):
                views = run(self.spv, FORMAT_1555, [fbits(x)] * 4)
                self.assertEqual(views[1][INDEX].stores, [((5, 3), (0xFFFF * want, 0, 0, 0))])

    def test_other_formats_keep_their_single_store(self):
        components = (0.25, 0.5, 0.75, 1.0)
        for fmt in (FORMAT_RGBA8, FORMAT_565, FORMAT_5551):
            with self.subTest(fmt=fmt):
                views = self.stores(fmt, components)
                self.assertEqual(views[0][INDEX].stores, [((5, 3), components)])
                self.assertEqual(views[1][INDEX].stores + views[2][INDEX].stores, [])
        # A UINT format is not packed either: the four words, through the uint view.
        words = [0xFC00, 7, 8, 9]
        views = run(self.spv, FORMAT_R16_UINT, words)
        self.assertEqual(views[1][INDEX].stores, [((5, 3), tuple(words))])
        self.assertEqual(views[0][INDEX].stores + views[2][INDEX].stores, [])

    def test_a_lookup_miss_stores_to_the_fallbacks(self):
        views = self.stores(FORMAT_1555, (1.0, 0.0, 0.0, 1.0), bound=False)
        self.assertEqual(views[0][0].stores, [((5, 3), (1.0, 0.0, 0.0, 1.0))])
        self.assertEqual(views[1][0].stores, [((5, 3), (0xFC00, 0, 0, 0))])
        self.assertEqual([v.stores for cls in views for v in cls[1:]], [[]] * 6)

    def test_dmask_leaves_missing_components_zero(self):
        # dmask 0x9: component 0 from v0 and component 3 from v1; 1 and 2 are 0.
        spv = support.translate_lines("ps_image_store_1555_dmask", lines(STORE.replace("dmask:0xf", "dmask:0x9")))
        validate(spv)
        views = run(spv, FORMAT_1555, [fbits(c) for c in (16 / 31, 1.0, 1.0, 1.0)])
        self.assertEqual(views[1][INDEX].stores, [((5, 3), (0x8000 | 16 << 10, 0, 0, 0))])
        self.assertEqual(views[0][INDEX].stores, [((5, 3), (si.f32(16 / 31), 0.0, 0.0, 1.0))])

    def test_d16_halves_are_packed_from_their_f16_values(self):
        # d16: two binary16 components a VGPR, low half first (v0 = x | y << 16, v1 = z | w << 16).
        spv = support.translate_lines("ps_image_store_1555_d16", lines(STORE.replace("v[0:3]", "v[0:1]") + " d16"))
        validate(spv)
        half = lambda x: struct.unpack("<H", struct.pack("<e", x))[0]    # noqa: E731
        components = (1.0, 0.5, 0.25, 1.0)
        words = [half(components[0]) | half(components[1]) << 16, half(components[2]) | half(components[3]) << 16, 0, 0]
        views = run(spv, FORMAT_1555, words)
        self.assertEqual(views[1][INDEX].stores, [((5, 3), (pack_1555(*components), 0, 0, 0))])
        self.assertEqual(pack_1555(*components), 0x8000 | 31 << 10 | 16 << 5 | 8)
        self.assertEqual(views[0][INDEX].stores, [((5, 3), components)])

    def test_the_packed_store_is_guarded_by_the_format(self):
        # One OpImageWrite a view class, plus the packed one: four in all, the last through the uint view, in a
        # selection on FORMAT == 134.
        writes = [ln.split()[1] for ln in self.spv.splitlines() if ln.strip().startswith("OpImageWrite ")]
        self.assertEqual(len(writes), 4)
        loads = {ln.split()[0]: ln.split()[3] for ln in self.spv.splitlines() if " = OpLoad %stimg" in ln}
        self.assertEqual([loads[w] for w in writes], ["%stimg2_0", "%stimg1_0", "%stimg0_0", "%stimg1_0"])
        self.assertRegex(self.spv, r"= OpIEqual %bool %\w+ %c86\n")      # 134 = 0x86


if __name__ == "__main__":
    unittest.main()

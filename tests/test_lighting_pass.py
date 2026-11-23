"""tools/lighting_pass.py: shader matching, table layout, plan building and the PASS checks, on hand-made inputs."""
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import lighting_pass as lp  # noqa: E402

PS_TEXT = "\n".join([
    "\ts_load_dwordx8 s[0:7], s[8:9], 0x1240",
    "\timage_load v[8:11], [v51, v52], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
    "\tv_and_b32_e32 v9, 0xfff, v9",
    "\tv_bfe_u32 v10, v10, 12, 12",
    "\ts_load_dwordx8 s[32:39], s[8:9], 0xbc0",
    "\timage_load v[14:16], [v51, v52], s[32:39] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm",
    "\ts_load_dwordx8 s[0:7], s[8:9], 0x1400",
    "\timage_load v7, [v51, v52], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
    "\ts_load_dwordx8 s[16:23], s[8:9], 0x900",
    "\timage_sample_lz v[56:59], [v44, v45, v43], s[16:23], s[88:91] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY",
    "\ts_load_dwordx4 s[68:71], s[8:9], 0x2a0",
    "\ts_load_dwordx4 s[4:7], s[8:9], 0x2c0",
    "\ts_load_dwordx4 s[60:63], s[10:11], null",
    "\ts_buffer_load_dword s55, s[68:71], 0x8",
    "\ts_buffer_load_dwordx4 s[16:19], s[60:63], 0x148",
    "\texp mrt0, v2, v2, v3, v3 compr",
    "\texp mrt1, v4, v4, off, off done compr vm",
]) + "\n"
NODIR_TEXT = "\n".join(l for l in PS_TEXT.splitlines() if "0x148" not in l and "s55" not in l) + "\n"
VS_TEXT = "\n".join(["\tv_cvt_f32_u32_e32 v2, s10", "\texp pos0, v0, v1, v2, v3 done", "\texp param0, v0, v0, v0, v3"]) + "\n"

PS_ENTRY = {"stage": "ps", "outputs": [0, 1], "user_sgpr_reads": [{"kind": k, "sgpr": s} for k, s in lp.PS_USER],
            "input_signature": [{"semantic": "SV_Position", "semantic_index": 0}]}
VS_ENTRY = {"stage": "vs", "bindings": [], "user_sgpr_reads": [{"kind": k, "sgpr": s} for k, s in lp.VS_USER],
            "input_signature": [{"semantic": "SV_VertexID", "semantic_index": 0}]}


def decode24(q):
    """The lighting PS's decode of the 24-bit normal (12 bits per axis, octahedral)."""
    x, y = (q & 0xFFF) * 2 / 4095 - 1, ((q >> 12) & 0xFFF) * 2 / 4095 - 1
    z = 1 - abs(x) - abs(y)
    t = max(-z, 0.0)
    x += -t if x >= 0 else t
    y += -t if y >= 0 else t
    l = (x * x + y * y + z * z) ** 0.5
    return x / l, y / l, z / l


class Matching(unittest.TestCase):
    def test_ps_and_directional(self):
        self.assertTrue(lp.ps_matches(PS_ENTRY, PS_TEXT))
        self.assertTrue(lp.is_directional(PS_TEXT))
        self.assertTrue(lp.ps_matches(PS_ENTRY, NODIR_TEXT))
        self.assertFalse(lp.is_directional(NODIR_TEXT))

    def test_ps_rejects_other_layouts(self):
        self.assertFalse(lp.ps_matches(PS_ENTRY, PS_TEXT.replace("0x1400", "0x1420")))     # no mask T#
        self.assertFalse(lp.ps_matches(PS_ENTRY, PS_TEXT.replace("0xfff", "0x3ff")))       # another normal encoding
        self.assertFalse(lp.ps_matches(PS_ENTRY, PS_TEXT + "\texp mrt2, v0, v0, v0, v0\n"))
        entry = dict(PS_ENTRY, user_sgpr_reads=[{"kind": "ptr32_lo", "sgpr": 8}])
        self.assertFalse(lp.ps_matches(entry, PS_TEXT))
        entry = dict(PS_ENTRY, input_signature=[{"semantic": "SV_Position", "semantic_index": 0},
                                                {"semantic": "TEXCOORD", "semantic_index": 0}])
        self.assertFalse(lp.ps_matches(entry, PS_TEXT))

    def test_vs(self):
        self.assertTrue(lp.vs_matches(VS_ENTRY, VS_TEXT))
        self.assertFalse(lp.vs_matches(VS_ENTRY, VS_TEXT + "\ts_load_dwordx4 s[0:3], s[8:9], 0x0\n"))
        self.assertFalse(lp.vs_matches(dict(VS_ENTRY, bindings=[0]), VS_TEXT))
        self.assertFalse(lp.vs_matches(VS_ENTRY, VS_TEXT + "\texp param1, v0, v0, v0, v0\n"))


class Layout(unittest.TestCase):
    def test_table_slots(self):
        s = lp.table_slots(PS_TEXT)
        self.assertEqual(s[("A", 0x1240)], ("T", "2d"))
        self.assertEqual(s[("A", 0x900)], ("T", "2darray"))
        self.assertEqual(s[("A", 0x2a0)], ("V", None))
        self.assertEqual(s[("B", 0)], ("V", None))   # `null` offset

    def test_table_b_alias(self):
        text = "\ts_mov_b32 s12, s9\n\ts_mov_b32 s9, 0\n" + PS_TEXT.replace("s[10:11]", "s[12:13]")
        s = lp.table_slots(text)
        self.assertEqual(s[("B", 0)], ("V", None))
        self.assertNotIn(("B", 0), lp.table_slots(PS_TEXT.replace("s[10:11]", "s[12:13]")))

    def test_normal_encoding_round_trip(self):
        for n in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0.48, -0.6, 0.64), (-0.36, 0.48, -0.8)):
            d = decode24(lp.oct24(n))
            self.assertLess(max(abs(a - b) for a, b in zip(d, n)), 2e-3, (n, d))

    def test_build_plan_and_heap(self):
        with tempfile.TemporaryDirectory() as t:
            plan = lp.build(t, "vs.spv", "ps.spv", PS_TEXT, [[0, 12582912, 407896064, 0]], normal=(0, 0, 1),
                            sun=(1.0, 0.0, 0.0), tint=2.0, mask=0, diffuse=(255, 0, 0))
            with open(plan, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r", raw)                       # the plan is written with \n line ends on every OS
            lines = raw.decode("utf-8").splitlines()
            self.assertIn("push 8 0", lines)
            self.assertIn("push 14 8000", lines)
            self.assertIn("sampler 0 c00000 18500000 0", lines)
            self.assertEqual(sum(l.startswith("target ") for l in lines), 2)
            images = {int(l.split()[-1], 16): l.split() for l in lines if l.startswith("image ")}
            self.assertEqual(images[0x900][1], "2darray")
            with open(images[lp.MASK_T][6], "rb") as f:
                self.assertEqual(set(f.read()), {0})
            with open(images[lp.NORMAL_T][6], "rb") as f:
                px = f.read(4)
            self.assertLess(max(abs(a - b) for a, b in zip(decode24(px[1] | px[2] << 8 | px[3] << 16), (0, 0, 1))), 2e-3)
            with open(os.path.join(t, "heap.bin"), "rb") as f:
                heap = f.read()
            cb = lambda va: struct.unpack_from("<I", heap, va)[0]
            self.assertEqual(struct.unpack_from("<3f", heap, cb(lp.TABLE_A + lp.SCALE_V)), (1.0, 1.0, 1.0))
            self.assertEqual(struct.unpack_from("<3f", heap, cb(lp.TABLE_B + lp.LIGHT_V) + lp.SUN_DIR), (-1.0, -0.0, -0.0))
            self.assertEqual(struct.unpack_from("<3f", heap, cb(lp.TABLE_B + lp.LIGHT_V) + lp.TINT), (2.0, 2.0, 2.0))
            self.assertEqual(struct.unpack_from("<f", heap, cb(lp.TABLE_A + lp.OCCL_V) + lp.OCCL)[0], 0.0)

    def test_diffuse_slot(self):
        text = PS_TEXT + "\ts_load_dwordx8 s[40:47], s[8:9], 0x920\n\timage_load v[2:5], [v51, v52], s[40:47] dmask:0xf dim:SQ_RSRC_IMG_2D unorm\n"
        with tempfile.TemporaryDirectory() as t:
            plan = lp.build(t, "vs.spv", "ps.spv", text, [], normal=(0, 0, 1), sun=(1.0, 0.0, 0.0), tint=1.0,
                            f0=(10, 20, 30), diffuse=(255, 0, 0))
            with open(plan, encoding="utf-8") as f:
                images = {int(l.split()[-1], 16): l.split() for l in f.read().splitlines() if l.startswith("image ")}
            for slot, px in ((lp.DIFFUSE_T, bytes([255, 0, 0, 255])), (lp.F0_T, bytes([10, 20, 30, 255]))):
                with open(images[slot][6], "rb") as f:
                    self.assertEqual(f.read(4), px)

    def test_read_rgb_checks_both_targets(self):
        with tempfile.TemporaryDirectory() as t:
            px = struct.pack("<4e", 1.0, 2.0, 3.0, 1.0) * 4
            for name, data in (("o_t0.bin", px), ("o_t1.bin", px)):
                with open(os.path.join(t, name), "wb") as f:
                    f.write(data)
            self.assertEqual(lp.read_rgb(t), ([1.0, 2.0, 3.0], True))
            with open(os.path.join(t, "o_t1.bin"), "wb") as f:      # a NaN in the last pixel of target 1
                f.write(px[:-8] + struct.pack("<4e", 0.0, 0.0, 0.0, float("nan")))
            self.assertEqual(lp.read_rgb(t), ([1.0, 2.0, 3.0], False))

    def test_pick(self):
        with tempfile.TemporaryDirectory() as t:
            spirv, disasm = os.path.join(t, "spirv"), os.path.join(t, "dxbc", "disasm")
            os.makedirs(spirv)
            os.makedirs(disasm)
            items = {"vs_a": (VS_ENTRY, VS_TEXT), "ps_dir_long": (PS_ENTRY, PS_TEXT + "\ts_nop 0\n"), "ps_dir": (PS_ENTRY, PS_TEXT),
                     "ps_nodir": (PS_ENTRY, NODIR_TEXT), "ps_other": (dict(PS_ENTRY, outputs=[0]), PS_TEXT)}
            for n, (e, text) in items.items():
                open(os.path.join(spirv, n + ".spv"), "wb").close()
                with open(os.path.join(disasm, n + ".s"), "w", encoding="utf-8") as f:
                    f.write(text)
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8") as f:
                json.dump({n: e for n, (e, _) in items.items()}, f)
            vs, dps, nps, counts = lp.pick(t)
            self.assertEqual(counts, {"vs": 1, "ps_dir": 2, "ps_nodir": 1})
            self.assertEqual([os.path.basename(p) for p in (vs, dps, nps)], ["vs_a.spv", "ps_dir.spv", "ps_nodir.spv"])
            self.assertEqual(lp.pick(os.path.join(t, "missing"))[3], {})


class Checks(unittest.TestCase):
    # Six normals in lp.NORMALS order: +sun, -sun, then four across.
    GOOD = dict(dir_terms=[3.0, 0.3, 1, 1, 1, 1], flip_terms=[0.5, 7.0, 1, 1, 1, 1],
                mask_terms=[0.0] * 6, nodir_terms=[0.0] * 6, finite=True)

    def test_pass(self):
        self.assertTrue(lp.check(**self.GOOD)[0])

    def test_failures(self):
        for change in (dict(dir_terms=[3.0, 3.0, 1, 1, 1, 1]),        # the normal does not matter
                       dict(flip_terms=[7.0, 0.5, 1, 1, 1, 1]),       # a flipped sun lights the same side
                       dict(mask_terms=[1.0] + [0.0] * 5),            # the mask does not gate the lighting term
                       dict(nodir_terms=[0.1] * 6),                   # the other variant reacts to the light
                       dict(dir_terms=[0.0] * 6),
                       dict(finite=False)):
            self.assertFalse(lp.check(**dict(self.GOOD, **change))[0], change)

    def test_albedo(self):
        good = [[3.9, 3.1, 3.1], [3.1, 3.9, 3.1], [3.2, 3.2, 3.9]]
        self.assertTrue(lp.albedo_ok(good))
        self.assertFalse(lp.albedo_ok([[3.2, 3.2, 3.2]] * 3))                   # albedo in the wrong slot
        self.assertFalse(lp.albedo_ok([[3.9, 3.1, 3.3], good[1], good[2]]))     # the other two channels differ
        self.assertFalse(lp.albedo_ok([good[1], good[0], good[2]]))             # channels swapped


class Cli(unittest.TestCase):
    def test_missing_tool(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(lp.main(["--work", t, "--build", t]), 77)

    def test_runner_exit_77_is_a_skip(self):
        with tempfile.TemporaryDirectory() as t:
            spirv, disasm = os.path.join(t, "shaders", "spirv"), os.path.join(t, "shaders", "dxbc", "disasm")
            os.makedirs(spirv)
            os.makedirs(disasm)
            items = {"vs_a": (VS_ENTRY, VS_TEXT), "ps_dir": (PS_ENTRY, PS_TEXT), "ps_nodir": (PS_ENTRY, NODIR_TEXT)}
            for n, (e, text) in items.items():
                open(os.path.join(spirv, n + ".spv"), "wb").close()
                with open(os.path.join(disasm, n + ".s"), "w", encoding="utf-8") as f:
                    f.write(text)
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8") as f:
                json.dump({n: e for n, (e, _) in items.items()}, f)
            open(os.path.join(t, "vk_lighting"), "wb").close()
            done = lp.subprocess.CompletedProcess([], 77, "", "no bufferDeviceAddress")
            with mock.patch.object(lp.subprocess, "run", return_value=done), mock.patch("sys.stdout"):
                self.assertEqual(lp.main(["--work", t, "--build", t]), 77)
            done = lp.subprocess.CompletedProcess([], 3, "", "device lost")
            with mock.patch.object(lp.subprocess, "run", return_value=done), mock.patch("sys.stdout"):
                self.assertEqual(lp.main(["--work", t, "--build", t]), 1)


if __name__ == "__main__":
    unittest.main()

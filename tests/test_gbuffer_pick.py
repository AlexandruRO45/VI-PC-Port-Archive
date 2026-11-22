"""tools/gbuffer.py: pair picking by layout, signature linking and the PASS thresholds, on hand-made inputs."""
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import gbuffer  # noqa: E402

VS_TEXT = "\n".join([
    "\ts_mov_b32 s4, s9",
    "\ts_load_dwordx4 s[20:23], s[4:5], 0x60",
    "\ts_load_dwordx4 s[24:27], s[4:5], 0x80",
    "\ts_load_dwordx4 s[28:31], s[4:5], 0xa0",
    "\ts_load_dwordx4 s[32:35], s[8:9], 0x6e0",
    "\ts_or_b32 s37, s12, 0x10000",
    "\ts_mov_b32 s36, s11",
    "\ts_mov_b32 s39, 0x11014fac",
    "\ts_lshl_b32 s40, s14, 5",
    "\tbuffer_load_format_x v6, v0, s[44:47], 0 idxen",
    "\ts_buffer_load_dwordx8 s[48:55], s[20:23], 0x60",
    "\ts_buffer_load_dwordx16 s[56:71], s[28:31], 0x160",
    "\ts_buffer_load_dwordx16 s[56:71], s[28:31], 0x1a0",
] + [f"\texp param{i} v{i}, v{i}, v{i}, v{i}" for i in range(6)]) + "\n"
PS_TEXT = "\n".join([
    "\ts_load_dwordx4 s[0:3], s[8:9], 0x6e0",
    "\ts_load_dwordx4 s[4:7], s[8:9], 0xac0",
    "\ts_mov_b32 s20, s9",
    "\ts_load_dwordx4 s[24:27], s[20:21], 0x20",
    "\ts_load_dwordx4 s[28:31], s[20:21], 0x60",
    "\ts_buffer_load_dword s32, s[4:7], s14",
    "\ts_mov_b32 s18, s15",
    "\ts_load_dwordx8 s[36:43], s[18:19], s32",
    "\ts_load_dwordx4 s[44:47], s[16:17], 0x520",
    "\timage_sample v[0:3], v[0:1], s[36:43], s[44:47] dmask:0x7 dim:SQ_RSRC_IMG_2D",
] + [f"\texp mrt{i} v0, v0, v1, v1 compr" for i in range(5)]) + "\n"


def user(pairs):
    return [{"kind": k, "sgpr": s} for k, s in pairs]


def signature_chunk(tag, elems):
    """A DXBC container holding one ISG1/OSG1 chunk; elems = (name, index, register, rw, min precision)."""
    n = len(elems)
    names = b""
    offs = []
    base = 8 + 32 * n
    for e in elems:
        offs.append(base + len(names))
        names += e[0].encode() + b"\0"
    body = struct.pack("<II", n, 8)
    for e, no in zip(elems, offs):
        body += struct.pack("<6I", 0, no, e[1], 0, 3, e[2]) + bytes([0xf, e[3], 0, 0]) + struct.pack("<I", e[4])
    body += names
    chunk = tag + struct.pack("<I", len(body)) + body
    head = b"DXBC" + bytes(16) + struct.pack("<III", 1, 32 + 4 + len(chunk), 1)
    return head + struct.pack("<I", 36) + chunk


VS_OUT = [("SV_POSITION", 0, 0, 0, 0), ("TEXCOORD", 0, 1, 0, 0), ("NORMAL", 0, 2, 0, 1),
          ("TEXCOORD", 1, 3, 0, 0), ("TEXCOORD", 10, 4, 0, 0), ("BARY", 0, 5, 0, 0), ("TRI", 0, 6, 0, 0)]
PS_IN = [("SV_POSITION", 0, 0, 0, 0), ("TEXCOORD", 0, 1, 3, 0), ("NORMAL", 0, 2, 7, 1),
         ("TEXCOORD", 10, 3, 15, 0)]


class Link(unittest.TestCase):
    def test_by_semantic(self):
        self.assertEqual(gbuffer.link(VS_OUT, 6, PS_IN, 3), [0, 1, 3])

    def test_precision_must_match(self):
        ps = [e if e[0] != "NORMAL" else e[:4] + (0,) for e in PS_IN]
        self.assertIsNone(gbuffer.link(VS_OUT, 6, ps, 3))

    def test_unread_inputs_are_skipped(self):
        ps = PS_IN + [("COLOR", 0, 4, 0, 0)]
        self.assertEqual(gbuffer.link(VS_OUT, 6, ps, 3), [0, 1, 3])

    def test_packed_register(self):
        vs = VS_OUT[:2] + [("COLOR", 0, 1, 0, 0)] + VS_OUT[2:]
        self.assertIsNone(gbuffer.link(vs, 7, PS_IN, 3))

    def test_export_count_mismatch(self):
        self.assertIsNone(gbuffer.link(VS_OUT, 7, PS_IN, 3))
        self.assertIsNone(gbuffer.link(VS_OUT, 6, PS_IN, 4))


class Signature(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.dxbc")
            with open(p, "wb") as f:
                f.write(signature_chunk(b"ISG1", PS_IN))
            self.assertEqual(gbuffer.dxbc_signature(p, (b"ISG1",)), PS_IN)
            self.assertIsNone(gbuffer.dxbc_signature(p, (b"OSG1",)))
            self.assertIsNone(gbuffer.dxbc_signature(os.path.join(d, "missing.dxbc"), (b"ISG1",)))


class PickPair(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        for sub in ("spirv", os.path.join("dxbc", "disasm")):
            os.makedirs(os.path.join(self.root, sub))
        self.index = {}

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, name, entry, text, chunk):
        self.index[name] = entry
        for path, data in ((("spirv", name + ".spv"), b"\x03\x02\x23\x07"), (("dxbc", name + ".dxbc"), chunk)):
            with open(os.path.join(self.root, *path), "wb") as f:
                f.write(data)
        with open(os.path.join(self.root, "dxbc", "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(self.root, "spirv", "index.json"), "w", encoding="utf-8") as f:
            json.dump(self.index, f)

    def vs(self, name, text=VS_TEXT, **kw):
        entry = {"stage": "vs", "bindings": [], "user_sgpr_reads": user(gbuffer.VS_USER),
                 "input_signature": [{"semantic": "SV_VertexID", "semantic_index": 0}]}
        entry.update(kw)
        self.add(name, entry, text, signature_chunk(b"OSG1", VS_OUT))

    def ps(self, name, text=PS_TEXT, **kw):
        entry = {"stage": "ps", "bindings": [0, 1, 2], "user_sgpr_reads": user(gbuffer.PS_USER),
                 "ps_interpolants": 3}
        entry.update(kw)
        self.add(name, entry, text, signature_chunk(b"ISG1", PS_IN))

    def test_picks_and_links(self):
        self.vs("b_vs")
        self.ps("a_ps")
        vs, ps, lk = gbuffer.pick_pair(self.root)
        self.assertEqual((os.path.basename(vs), os.path.basename(ps), lk), ("b_vs.spv", "a_ps.spv", [0, 1, 3]))

    def test_rejects_layout_mismatch(self):
        self.vs("vs_bound", bindings=[0])
        self.vs("vs_image", text=VS_TEXT + "\timage_load v0, v0, s[0:7]\n")
        self.vs("vs_input", input_signature=[{"semantic": "POSITION", "semantic_index": 0}])
        self.ps("ps_two", text=PS_TEXT + PS_TEXT.splitlines()[-6] + "\n")
        self.ps("ps_four", text=PS_TEXT.replace("\texp mrt4 v0, v0, v1, v1 compr\n", ""))
        self.ps("ps_user", user_sgpr_reads=user(gbuffer.PS_USER[:-1]))
        self.assertIsNone(gbuffer.pick_pair(self.root))
        self.vs("vs_ok")
        self.assertIsNone(gbuffer.pick_pair(self.root))
        self.ps("ps_ok")
        self.assertEqual(os.path.basename(gbuffer.pick_pair(self.root)[1]), "ps_ok.spv")

    def test_no_index(self):
        self.assertIsNone(gbuffer.pick_pair(self.root))


class Counts(unittest.TestCase):
    GOOD = "vk_gbuffer: draws=4 pixels=1000 covered=500 shaded=500 albedo_colours=300 normals_facing=400 " \
           "motion_still=500 motion_max=0.003"

    def test_pass(self):
        c = gbuffer.parse_counts(self.GOOD)
        self.assertEqual(c["motion_max"], 0.003)
        self.assertTrue(gbuffer.check_counts(c, True)[0])

    def test_fail_each(self):
        for key, value in (("covered", 5), ("shaded", 480), ("albedo_colours", 10), ("normals_facing", 200),
                           ("motion_still", 480)):
            c = gbuffer.parse_counts(self.GOOD)
            c[key] = value
            self.assertFalse(gbuffer.check_counts(c, True)[0], key)
        c = gbuffer.parse_counts(self.GOOD)
        c["albedo_colours"] = 1
        self.assertTrue(gbuffer.check_counts(c, False)[0])
        self.assertFalse(gbuffer.check_counts({}, False)[0])


if __name__ == "__main__":
    unittest.main()

"""tools/sandbox.py: the shader pair is chosen by signature and descriptor layout, never by name."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import sandbox  # noqa: E402
import sandbox_scene  # noqa: E402

VS_TEXT = """\ts_load_dwordx4 s[0:3], s[8:9], null
\ts_load_dwordx4 s[20:23], s[8:9], 0x40
\ts_load_dwordx4 s[16:19], s[8:9], 0x60
\ts_buffer_load_dwordx4 s[24:27], s[0:3], 0x30
\ts_buffer_load_dwordx16 s[0:15], s[20:23], 0x160
\texp pos0 v0, v1, v2, v3 done
"""
PS_TEXT = """\ts_load_dwordx8 s[4:11], s[8:9], 0x20
\timage_sample v[0:3], v[2:3], s[4:11], s[0:3] dmask:0xf dim:SQ_RSRC_IMG_2D
\texp mrt0 v0, v0, v1, v1 done compr vm
"""


def sig(*elems):
    return [{"semantic": s, "semantic_index": 0, "mask": m} for s, m in elems]


VS_SIG = sig(("POSITION", 7), ("NORMAL", 7), ("TEXCOORD", 3))
PS_SIG = sig(("SV_Position", 15), ("TEXCOORD", 3), ("COLOR", 15))
PTR8 = [{"kind": "ptr32_lo", "sgpr": 8}]


class PickPair(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "spirv"))
        os.makedirs(os.path.join(self.root, "dxbc", "disasm"))
        self.index = {}

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, name, entry, text, spv=True):
        self.index[name] = entry
        with open(os.path.join(self.root, "dxbc", "disasm", name + ".s"), "w") as f:
            f.write(text)
        if spv:
            open(os.path.join(self.root, "spirv", name + ".spv"), "wb").close()

    def pick(self):
        with open(os.path.join(self.root, "spirv", "index.json"), "w") as f:
            json.dump(self.index, f)
        return sandbox.pick_pair(self.root)

    def names(self):
        pair = self.pick()
        return pair and tuple(os.path.splitext(os.path.basename(p))[0] for p in pair)

    def test_picks_matching_pair(self):
        self.add("a_vs", {"stage": "vs", "input_signature": VS_SIG, "user_sgpr_reads": PTR8, "bindings": []}, VS_TEXT)
        self.add("b_ps", {"stage": "ps", "input_signature": PS_SIG, "user_sgpr_reads": PTR8 + [{"kind": "const32", "sgpr": 9}],
                          "bindings": [0, 1, 2]}, PS_TEXT)
        self.assertEqual(self.names(), ("a_vs", "b_ps"))

    def test_rejects_wrong_layout_and_signature(self):
        self.add("a_vs", {"stage": "vs", "input_signature": VS_SIG, "user_sgpr_reads": PTR8, "bindings": []},
                 VS_TEXT.replace("0x160", "0x100"))
        self.add("c_vs", {"stage": "vs", "input_signature": VS_SIG[:2], "user_sgpr_reads": PTR8, "bindings": []}, VS_TEXT)
        self.add("b_ps", {"stage": "ps", "input_signature": PS_SIG, "user_sgpr_reads": PTR8, "bindings": [0, 1, 2]}, PS_TEXT)
        self.assertIsNone(self.pick())

    def test_rejects_two_samples_and_missing_spirv(self):
        self.add("a_vs", {"stage": "vs", "input_signature": VS_SIG, "user_sgpr_reads": PTR8, "bindings": []}, VS_TEXT)
        self.add("b_ps", {"stage": "ps", "input_signature": PS_SIG, "user_sgpr_reads": PTR8, "bindings": [0, 1, 2]},
                 PS_TEXT + PS_TEXT)
        self.add("c_ps", {"stage": "ps", "input_signature": PS_SIG, "user_sgpr_reads": PTR8, "bindings": [0, 1, 2]},
                 PS_TEXT, spv=False)
        self.assertIsNone(self.pick())
        self.add("d_ps", {"stage": "ps", "input_signature": PS_SIG, "user_sgpr_reads": PTR8, "bindings": [0, 1, 2]}, PS_TEXT)
        self.assertEqual(self.names(), ("a_vs", "d_ps"))

    def test_missing_index(self):
        self.assertIsNone(sandbox.pick_pair(os.path.join(self.root, "nowhere")))


class CheckFrame(unittest.TestCase):
    def frame(self, pixels, w, h):
        fd, path = tempfile.mkstemp(suffix=".ppm")
        with os.fdopen(fd, "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (w, h) + bytes(c for p in pixels for c in p))
        self.addCleanup(os.remove, path)
        return path

    def test_empty_frame_fails(self):
        ok, msg = sandbox.check_frame(self.frame([(140, 173, 217)] * 100, 10, 10), False)
        self.assertFalse(ok)
        self.assertIn("0 of 100", msg)

    def test_covered_frame(self):
        px = [(140, 173, 217)] * 50 + [(64, 64, 64)] * 50
        self.assertTrue(sandbox.check_frame(self.frame(px, 10, 10), False)[0])
        self.assertFalse(sandbox.check_frame(self.frame(px, 10, 10), True)[0])   # textured needs colour variety
        varied = [(140, 173, 217)] * 300 + [(i % 256, i // 256, 7) for i in range(300)]
        self.assertTrue(sandbox.check_frame(self.frame(varied, 30, 20), True)[0])


class ScenePlacementOptions(unittest.TestCase):
    """--radius and --max-props are refused by sandbox.py as by sandbox_scene.py, before vk_sandbox is looked up."""
    BAD = (["--radius", "nan"], ["--radius", "inf"], ["--radius", "-5"], ["--max-props", "-1"])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.entities = os.path.join(self.tmp.name, "entities.jsonl")
        open(self.entities, "w").close()

    def refused(self, main, argv):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as stop:
            main(argv)
        self.assertEqual(stop.exception.code, 2)
        return err.getvalue()

    def test_sandbox_refuses_bad_values_before_looking_for_vk_sandbox(self):
        # No vk_sandbox and no shaders here: before the check this was exit 77, "unsupported", which reads as a skip.
        base = ["--work", self.tmp.name, "--entities", self.entities, "--models", self.tmp.name,
                "--build", os.path.join(self.tmp.name, "no-build"), "--check"]
        for extra in self.BAD:
            with self.subTest(extra=extra):
                self.assertIn(f"{extra[0]} must be", self.refused(sandbox.main, base + extra))

    def test_sandbox_scene_refuses_the_same_values(self):
        base = ["--entities", self.entities, "--models", self.tmp.name, "--out", os.path.join(self.tmp.name, "s.bin")]
        for extra in self.BAD:
            with self.subTest(extra=extra):
                self.assertIn(f"{extra[0]} must be", self.refused(sandbox_scene.main, base + extra))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "s.bin")))

    def test_zero_is_still_accepted(self):
        # 0 is a valid radius (only props at the centre) and a valid cap (an empty scene), so it reaches the tool lookup.
        base = ["--work", self.tmp.name, "--entities", self.entities, "--models", self.tmp.name,
                "--build", os.path.join(self.tmp.name, "no-build"), "--check", "--radius", "0", "--max-props", "0"]
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(sandbox.main(base), 77)
        self.assertIn("vk_sandbox is not built", out.getvalue())


if __name__ == "__main__":
    unittest.main()

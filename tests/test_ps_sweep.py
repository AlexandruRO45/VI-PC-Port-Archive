"""tools/ps_sweep.py on a synthetic index and synthetic outputs: no GPU, game data or Vulkan tools needed.

    python -I tests/test_ps_sweep.py
"""

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import ps_sweep  # noqa: E402


def meta(bary=False, bindings=(), flow="structured", stage="ps", **extra):
    m = {"stage": stage, "features": ["fragmentShaderBarycentric"] if bary else [], "bindings": list(bindings),
         "control_flow": flow}
    m.update(extra)
    return m


def synthetic_index():
    index = {}
    for i in range(60):
        index[f"a{i}"] = meta()
    for i in range(40):
        index[f"b{i}"] = meta(bary=True, bindings=(0, 1, 2))
    for i in range(30):
        index[f"c{i}"] = meta(bindings=(0, 1, 2, 5), flow="dispatch loop")
    for i in range(3):
        index[f"d{i}"] = meta(bindings=(0, 1, 2, 3), flow="straight line")
    for i in range(20):
        index[f"v{i}"] = meta(stage="vs")
    return index


TARGET_FLOATS = 32 * 32 * 4


def write_raw(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))
        f.write(struct.pack("<f", -1.0) * (TARGET_FLOATS - len(values)))


class Sampling(unittest.TestCase):
    def test_strata_and_pixel_shaders_only(self):
        ps = {k: v for k, v in synthetic_index().items() if v["stage"] == "ps"}
        self.assertEqual(len(ps), 133)
        picked, per = ps_sweep.pick_sample(ps, 40, 7)
        self.assertEqual(set(per), {(False, "none", "structured"), (True, "low", "structured"),
                                    (False, "high", "dispatch loop"), (False, "low", "straight line")})
        self.assertEqual(len(picked), len(set(picked)))
        self.assertTrue(all(name in ps for name in picked))
        # every stratum is represented, a stratum smaller than the floor is taken whole
        self.assertEqual(per[(False, "low", "straight line")], (3, 3))
        self.assertTrue(all(sampled >= min(10, size) for size, sampled in per.values()))
        self.assertEqual(sum(sampled for _, sampled in per.values()), len(picked))

    def test_fixed_seed_is_stable_and_seed_matters(self):
        ps = {k: v for k, v in synthetic_index().items() if v["stage"] == "ps"}
        first = ps_sweep.pick_sample(ps, 40, 7)[0]
        self.assertEqual(first, ps_sweep.pick_sample(ps, 40, 7)[0])
        self.assertNotEqual(first, ps_sweep.pick_sample(ps, 40, 8)[0])

    def test_sample_command_writes_numbers_only_to_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            spirv, out = os.path.join(tmp, "spirv"), os.path.join(tmp, "out")
            os.makedirs(spirv)
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(synthetic_index(), f)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ps_sweep.main(["sample", "--spirv", spirv, "--out", out, "--n", "40"])
            self.assertNotIn("a1", buf.getvalue().replace("sampled", ""))
            self.assertIn("population 133", buf.getvalue())
            with open(os.path.join(out, "sample.json"), encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), int(buf.getvalue().split("sampled ")[1].split()[0]))


class VertexShader(unittest.TestCase):
    def test_outputs_match_inputs_and_flat_locations(self):
        glsl = ps_sweep.vs_glsl([0, 2, 5], {2})
        self.assertIn("layout(location=0) out vec4 o0;", glsl)
        self.assertIn("layout(location=2) flat out vec4 o2;", glsl)
        self.assertIn("layout(location=5) out vec4 o5;", glsl)
        self.assertNotIn("o1", glsl)
        self.assertIn("gl_VertexIndex", glsl)

    def test_no_inputs(self):
        glsl = ps_sweep.vs_glsl([], set())
        self.assertNotIn("out vec4", glsl)
        self.assertIn("gl_Position", glsl)


class Classification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def raw(self, name, values):
        path = os.path.join(self.tmp.name, name)
        write_raw(path, values)
        return path

    @staticmethod
    def rec(code):
        return {"i": 0, "code": code, "reason": "", "targets": 1}

    def classify(self, a, b):
        return ps_sweep.classify_pair(self.rec(0), self.rec(0), self.raw("a", a), self.raw("b", b))

    def test_ok_within_tolerance(self):
        self.assertEqual(self.classify([0.5, 1.0, 2.0, -1.0], [0.5004, 1.0004, 2.0, -1.0]), ("OK on both", ""))

    def test_differs_beyond_tolerance_with_tier(self):
        self.assertEqual(self.classify([0.5, 1.0], [0.5, 1.5]), ("differs beyond tolerance", "gross (>0.1)"))
        self.assertEqual(self.classify([0.5, 1.0], [0.5, 1.05]), ("differs beyond tolerance", "moderate (>1e-2)"))
        self.assertEqual(self.classify([0.5, 1.0], [0.5, 1.005]), ("differs beyond tolerance", "small (>1e-3)"))

    def test_nan_and_zero_on_one_side_only(self):
        nan = float("nan")
        self.assertEqual(self.classify([0.5, 1.0], [0.5, nan]), ("NaN or zero on one GPU only", "nan"))
        self.assertEqual(self.classify([0.5, 1.0], [0.0, 0.0]), ("NaN or zero on one GPU only", "zero"))
        self.assertEqual(self.classify([0.5, 1.0], [-1.0, -1.0]), ("NaN or zero on one GPU only", "unwritten"))
        self.assertEqual(self.classify([nan, 1.0], [nan, 1.0]), ("OK on both", ""))   # NaN on both is not a difference

    def test_non_finite_values_compare_per_component(self):
        nan, inf = float("nan"), float("inf")
        gross = ("differs beyond tolerance", "gross (>0.1)")
        # NaN in different components of the two outputs
        self.assertEqual(self.classify([nan, 1.0], [1.0, nan]), gross)
        # an infinity against a finite value, and against the opposite infinity
        self.assertEqual(self.classify([inf, 1.0], [1.0, 1.0]), gross)
        self.assertEqual(self.classify([inf, 1.0], [-inf, 1.0]), gross)
        # the same non-finite value at the same component matches
        self.assertEqual(self.classify([inf, nan, 1.0], [inf, nan, 1.0]), ("OK on both", ""))

    def test_failures_and_timeouts(self):
        # A non-success code takes precedence; neither path should even be opened.
        for a, b, expected in ((0, 77, ("fails on one GPU", "b: missing device feature")),
                               (4, 0, ("fails on one GPU", "a: exit 4")),
                               (77, 5, ("fails on both", "77/5")),
                               (3, 0, ("timeout", "")), (0, 3, ("timeout", ""))):
            with self.subTest(a=a, b=b):
                result = ps_sweep.classify_pair(self.rec(a), self.rec(b), self.tmp.name, self.tmp.name)
                self.assertEqual(result, expected)


class CompareCommand(unittest.TestCase):
    def test_readback_shapes(self):
        one = struct.pack("<f", 0.25) * TARGET_FLOATS
        two = one * 2
        cases = [
            # recorded counts, raw outputs (None = absent), expected classification and detail
            (2, 2, two, two, "OK on both", ""),
            (8, 8, one * 8, one * 8, "OK on both", ""),
            (2, 2, two, two[:-4] + struct.pack("<f", 1.0), "differs beyond tolerance", "gross (>0.1)"),
            (1, 1, b"", b"", "invalid readback", "a: wrong output size"),
            (1, 1, one[:-4], one[:-4], "invalid readback", "a: wrong output size"),
            (1, 1, one, one[:-4], "invalid readback", "b: wrong output size"),
            (1, 1, two, two, "invalid readback", "a: wrong output size"),
            (1, 1, one, one + b"\x00", "invalid readback", "b: wrong output size"),
            (2, 2, one, one, "invalid readback", "a: wrong output size"),
            (1, 2, one, one, "invalid readback", "target counts differ"),
            (None, 1, one, one, "invalid readback", "invalid target count"),
            (0, 0, b"", b"", "invalid readback", "invalid target count"),
            (9, 9, one * 9, one * 9, "invalid readback", "invalid target count"),
            (1, True, one, one, "invalid readback", "invalid target count"),
            (1, 1, one, None, "invalid readback", "b: missing output"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sample = [{"name": str(i), "stratum": [False, "none", "structured"]} for i in range(len(cases))]
            (out / "sample.json").write_text(json.dumps(sample), encoding="utf-8", newline="\n")
            for side, tag in enumerate(("a", "b")):
                folder = out / f"{tag}_z"
                folder.mkdir()
                with (folder / "results_0.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                    for i, case in enumerate(cases):
                        rec = {"i": i, "code": 0, "reason": ""}
                        if case[side] is not None:
                            rec["targets"] = case[side]
                        f.write(json.dumps(rec) + "\n")
                        data = case[2 + side]
                        if data is not None:
                            (folder / f"{i}.raw").write_bytes(data)
            with contextlib.redirect_stdout(io.StringIO()):
                ps_sweep.main(["compare", "--out", tmp])
            rows = json.loads((out / "classes_z.json").read_text(encoding="utf-8"))
            self.assertEqual([(r["class"], r["sub"]) for r in rows], [case[-2:] for case in cases])

    def test_rerun_overrides_and_counts(self):
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "sample.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([{"name": "x", "stratum": [False, "none", "structured"]},
                           {"name": "y", "stratum": [False, "none", "structured"]}], f)
            for tag, codes in (("a", (0, 3)), ("b", (0, 0))):
                d = os.path.join(out, f"{tag}_z")
                os.makedirs(d)
                with open(os.path.join(d, "results_0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                    for i, c in enumerate(codes):
                        f.write(json.dumps({"i": i, "code": c, "reason": "", "targets": 1}) + "\n")
                write_raw(os.path.join(d, "0.raw"), [0.25] * 8)
                write_raw(os.path.join(d, "1.raw"), [0.25] * 8)
            # shader 1 timed out on a while another process was starting: its re-run is clean
            re = os.path.join(out, "a_z_re")
            os.makedirs(re)
            with open(os.path.join(re, "results_0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"i": 1, "code": 0, "reason": "", "targets": 1}) + "\n")
            write_raw(os.path.join(re, "1.raw"), [0.25] * 8)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ps_sweep.main(["compare", "--out", out, "--tags", "a", "b"])
            self.assertIn("OK on both: 2", buf.getvalue())
            self.assertNotIn("timeout", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Synthetic tests for tools/render_buckets.py (no game data)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import render_buckets as rb  # noqa: E402


def model(name, buckets, verts=10, tris=4):
    return {"Name": name, "Verts": verts, "Tris": tris,
            "Materials": [{"Shader": f"{i:08x}", "RenderBucket": b} for i, b in enumerate(buckets)]}


def write_manifest(folder, items, count=None, extra_before=True):
    os.makedirs(folder, exist_ok=True)
    doc = {"src": "synthetic"}
    if extra_before:
        doc["Count"] = len(items) if count is None else count
    doc["Items"] = items
    doc["ModelDecodeFailures"] = []
    path = os.path.join(folder, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return path


class StreamTests(unittest.TestCase):
    def test_items_are_read_across_small_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = [model(f"m{i}", [i % 7, 0x80 | 2]) for i in range(25)]
            path = write_manifest(tmp, items)
            for chunk in (5, 7, 64, rb.CHUNK):
                gen = rb.iter_items(path, chunk)
                got = []
                while True:
                    try:
                        got.append(next(gen))
                    except StopIteration as stop:
                        count = stop.value
                        break
                self.assertEqual(got, items, chunk)
                self.assertEqual(count, 25)

    def test_count_before_or_after_items(self):
        # review on #4946: a top-level Count after Items must be read too, with the items still streamed
        with tempfile.TemporaryDirectory() as tmp:
            for i, text in enumerate(('{"Count": 2, "Items": [{}]}', '{"Items": [{}], "Count": 2}',
                                      '{"Items": [{}], "Failures": [{"Count": 7}], "Count": 2}',
                                      '{"Items": [{}], "Failures": [{"Count": 7}]}')):
                path = os.path.join(tmp, f"m{i}.json")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                want = None if i == 3 else 2       # a nested Count is not the top-level one
                for chunk in (3, rb.CHUNK):
                    gen = rb.iter_items(path, chunk)
                    got = []
                    while True:
                        try:
                            got.append(next(gen))
                        except StopIteration as stop:
                            self.assertEqual(stop.value, want, (text, chunk))
                            break
                    self.assertEqual(got, [{}])

    def test_multi_digit_count_split_between_reads(self):
        # review on #4946: a read that ends inside the digits of a Count after Items must not give a shorter number
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            text = '{"Items": [{}], "Count":   1234}'
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            for chunk in range(1, len(text) + 2):
                gen = rb.iter_items(path, chunk)
                while True:
                    try:
                        next(gen)
                    except StopIteration as stop:
                        self.assertEqual(stop.value, 1234, chunk)
                        break

    def test_unrelated_values_are_skipped_not_built(self):
        # review on #4946: a large member before a trailing Count is skipped, not decoded into Python objects
        import tracemalloc
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            failures = ",".join(['{"Offset":1234,"Reason":"synthetic ] } [ \\" x"}'] * 40000)
            peaks = {}
            for last in (False, True):
                fields = ['"Items":[{}]', '"ModelDecodeFailures":[' + failures + ']']
                fields.insert(2 if last else 0, '"Count":1')
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{" + ",".join(fields) + "}")
                tracemalloc.start()
                gen = rb.iter_items(path)
                while True:
                    try:
                        next(gen)
                    except StopIteration as stop:
                        self.assertEqual(stop.value, 1)
                        break
                peaks[last] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            self.assertLess(peaks[True], 2 * peaks[False] + (1 << 20), peaks)

    def test_skipped_values_must_be_well_formed(self):
        bad = ['{"Items":[],"X":[1,2}', '{"Items":[],"X":{"a":"b]', '{"X":"unterminated,"Items":[]}',
               '{"Items":[],"X":[[]]]}']
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            for text in bad:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                for chunk in (1, 3, rb.CHUNK):
                    with self.assertRaises(ValueError, msg=(text, chunk)):
                        list(rb.iter_items(path, chunk))

    def test_invalid_json_is_rejected(self):
        # review on #4946: separators and the enclosing object are checked, at every read size
        bad = ['{ "Count":2,"Items":[{}{}]', '{"Items":[{},]}', '{"Items":[,{}]}', '{"Items":[{},,{}]}',
               '{"Items":[{}]', '{"Items":[{}]} x', '{"Items":[{}] "Count":1}', '{"Count":1 "Items":[]}',
               '{"Items":[{}],}', '{"Count":1}', '[{}]', '']
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            for text in bad:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                for chunk in (1, 2, 5, rb.CHUNK):
                    with self.assertRaises(ValueError, msg=(text, chunk)):
                        list(rb.iter_items(path, chunk))

    def test_top_level_count_only(self):
        good = {'{"src":"\\"Items\\": [","Items":[{}],"Count":1}': 1,
                '{"Count":2,"Items":[{},{}],"Fail":[{"Count":7}]}': 2,
                '{"Items":[],"Count":0}': 0, '{"Items":[{}]}': None}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            for text, want in good.items():
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                for chunk in range(1, len(text) + 2):
                    gen = rb.iter_items(path, chunk)
                    while True:
                        try:
                            next(gen)
                        except StopIteration as stop:
                            self.assertEqual(stop.value, want, (text, chunk))
                            break

    def test_number_cut_at_a_read_boundary(self):
        # found by shdann on #4980 (same reader): a read that ends at the e of 1e3 must not give 1
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            for text in ('{"scale":1e3,"Items":[{}],"Count":1}', '{"Items":[{}],"Count":1,"scale":-1.5E-3}'):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                for chunk in range(1, len(text) + 2):
                    gen = rb.iter_items(path, chunk)
                    while True:
                        try:
                            next(gen)
                        except StopIteration as stop:
                            self.assertEqual(stop.value, 1, (text, chunk))
                            break

    def test_empty_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(tmp, [])
            self.assertEqual(list(rb.iter_items(path, 3)), [])

    def test_missing_items_array_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"Count": 0}, f)
            with self.assertRaises(ValueError):
                list(rb.iter_items(path))


class CountTests(unittest.TestCase):
    def test_buckets_flag_outside_and_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = model("a", [0, 1, 0x80 | 2, 6, 7])
            m["Materials"].append({"Shader": "x", "RenderBucket": "3"})   # not an integer
            m["Materials"].append({"Shader": "y"})                         # missing
            path = write_manifest(os.path.join(tmp, "arc", "models"), [m])
            per, total, problems = rb.count([("arc", path)])
            self.assertEqual(problems, [])
            self.assertEqual(total["materials"], 5)
            self.assertEqual(total["bad"], 2)
            self.assertEqual(dict(total["low"]), {0: 1, 1: 1, 2: 1, 6: 1, 7: 1})
            self.assertEqual(total["flag"], 1)
            self.assertEqual(dict(total["flag_by_low"]), {2: 1})
            self.assertEqual(total["outside"], 1)

    def test_unique_models_count_once_across_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = model("shared", [0, 2])
            p1 = write_manifest(os.path.join(tmp, "one", "models"), [shared, model("only1", [1])])
            p2 = write_manifest(os.path.join(tmp, "two", "models"), [shared, shared, model("only2", [6])])
            per, total, _ = rb.count([("one", p1), ("two", p2)])
            self.assertEqual(total["models"], 5)
            self.assertEqual(total["materials"], 8)
            self.assertEqual(total["unique_models"], 3)
            self.assertEqual(total["unique_materials"], 4)
            self.assertEqual(dict(total["unique_low"]), {0: 1, 2: 1, 1: 1, 6: 1})
            # same name, different geometry: a different model
            p3 = write_manifest(os.path.join(tmp, "three", "models"), [model("shared", [0, 2], verts=11)])
            _, total, _ = rb.count([("one", p1), ("three", p3)])
            self.assertEqual(total["unique_models"], 3)

    def test_distinct_shaders_per_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = [{"Name": "a", "Materials": [{"Shader": "AAAA0001", "RenderBucket": 4},
                                                 {"Shader": "aaaa0001", "RenderBucket": 0x80 | 4},
                                                 {"Shader": "bbbb0002", "RenderBucket": 0}]}]
            path = write_manifest(os.path.join(tmp, "arc", "models"), items)
            _, total, _ = rb.count([("arc", path)])
            out = rb.plain(total)
            self.assertEqual(out["distinct_shaders"], {0: 1, 4: 1})
            self.assertEqual(out["shaders_if_few"][4], ["aaaa0001"])

    def test_missing_shader_is_not_a_distinct_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = [{"Name": "a", "Materials": [{"RenderBucket": 6}, {"Shader": None, "RenderBucket": 6},
                                                 {"Shader": "cccc0003", "RenderBucket": 6}]}]
            path = write_manifest(os.path.join(tmp, "arc", "models"), items)
            _, total, _ = rb.count([("arc", path)])
            out = rb.plain(total)
            self.assertEqual(out["distinct_shaders"], {6: 1})
            self.assertEqual(out["no_shader"], 2)
            self.assertEqual(total["materials"], 3)

    def test_same_label_twice_keeps_both_entries(self):
        # review on #4946: a second manifest with the same archive label must not overwrite the first entry
        with tempfile.TemporaryDirectory() as tmp:
            p1 = write_manifest(os.path.join(tmp, "a", "levels", "models"), [model("x", [0])])
            p2 = write_manifest(os.path.join(tmp, "b", "levels", "models"), [model("y", [6]), model("z", [6])])
            per, total, _ = rb.count([("levels", p1), ("levels", p2)])
            self.assertEqual(sorted(per), ["levels", "levels (2)"])
            self.assertEqual((per["levels"]["models"], per["levels (2)"]["models"]), (1, 2))
            self.assertEqual(total["models"], sum(t["models"] for t in per.values()))

    def test_count_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(os.path.join(tmp, "arc", "models"), [model("a", [0])], count=2)
            _, _, problems = rb.count([("arc", path)])
            self.assertEqual(len(problems), 1)
            self.assertIn("1 items read", problems[0])


class CliTests(unittest.TestCase):
    def run_cli(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = rb.main(args)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_directory_finds_every_archive_including_dot_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(os.path.join(tmp, "plain", "models"), [model("a", [0])])
            write_manifest(os.path.join(tmp, ".dotted", "models"), [model("b", [6])])
            os.makedirs(os.path.join(tmp, "nomodels"))
            labels = [label for label, _ in rb.manifests(tmp)]
            self.assertEqual(sorted(labels), [".dotted", "plain"])
            code, out, _ = self.run_cli([tmp])
            self.assertEqual(code, 0)
            self.assertIn("total: 2 models, 2 materials", out)

    def test_json_report_and_refusal_to_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(os.path.join(tmp, "arc", "models"), [model("a", [0x80 | 1])])
            report = os.path.join(tmp, "report.json")
            code, _, _ = self.run_cli([path, "--json", report])
            self.assertEqual(code, 0)
            with open(report, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["total"]["flag"], 1)
            self.assertEqual(data["total"]["low"], {"1": 1})
            code, _, err = self.run_cli([path, "--json", path])
            self.assertNotEqual(code, 0)
            self.assertIn("overwrite", err)

    def test_mismatch_with_count_after_items_sets_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "arc", "models")
            os.makedirs(folder)
            with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
                f.write('{"Items": [{}], "Count": 2}')
            code, out, err = self.run_cli([tmp])
            self.assertEqual(code, 1)
            self.assertIn("(Count 2)", out)
            self.assertIn("Count says 2", err)

    def test_invalid_manifest_exits_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "arc", "models")
            os.makedirs(folder)
            with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
                f.write('{"Items":[{}{}]}')
            code, _, err = self.run_cli([tmp])
            self.assertEqual(code, 2)
            self.assertIn("invalid manifest JSON", err)
            self.assertNotIn("Traceback", err)

    def test_non_scalar_fields_are_counted_not_crashed(self):
        # Flyander on #4946: a list as Shader (and as Name) crashed the unique-model set with TypeError
        items = [{"Name": "x", "Verts": 1, "Tris": 1, "Materials": [{"Shader": ["bad"], "RenderBucket": 0}]},
                 {"Name": ["n"], "Verts": 1, "Tris": 1, "Materials": [{"Shader": {"a": 1}, "RenderBucket": "z"}]},
                 {"Name": "x", "Verts": 1, "Tris": 1, "Materials": [{"Shader": ["bad"], "RenderBucket": 0}]}]
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(os.path.join(tmp, "arc", "models"), items, count=3)
            code, out, err = self.run_cli([tmp])
            self.assertEqual(code, 0, err)
            self.assertNotIn("Traceback", err)
            self.assertIn("unique models 2", out)        # the repeated malformed model is still one model
            self.assertIn("unreadable bucket: 1", out)
            self.assertIn("materials without a shader hash: 2", out)

    def test_non_object_entries_exit_2(self):
        # LILILI2007 on #4946: falsy non-arrays ({} "" 0 false) were read as "no materials"
        for items in ([1], [{"Materials": [7]}], [{"Materials": {"Shader": "s"}}], [{"Materials": {}}],
                      [{"Materials": ""}], [{"Materials": 0}], [{"Materials": False}]):
            with tempfile.TemporaryDirectory() as tmp:
                write_manifest(os.path.join(tmp, "arc", "models"), items)
                code, _, err = self.run_cli([tmp])
                self.assertEqual(code, 2, (items, err))
                self.assertNotIn("Traceback", err)

    def test_mismatch_sets_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(os.path.join(tmp, "arc", "models"), [model("a", [0])], count=5)
            code, _, err = self.run_cli([tmp])
            self.assertEqual(code, 1)
            self.assertIn("Count says 5", err)


if __name__ == "__main__":
    unittest.main()

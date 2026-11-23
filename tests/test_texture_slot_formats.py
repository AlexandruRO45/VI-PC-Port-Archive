"""Synthetic tests for tools/texture_slot_formats.py (no game data)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import texture_slot_formats as tsf  # noqa: E402

BUMP, DIFF, SPEC = "1E77BF4C", "C8E8C282", "C6F46E79"


def dump(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)


def pool(folder, rows, files=True):
    """A pool manifest whose rows point at existing (empty) files unless files is False."""
    textures = []
    for i, (key, fmt) in enumerate(rows):
        name = f"t{i}.dds"
        if files:
            os.makedirs(folder, exist_ok=True)
            open(os.path.join(folder, name), "wb").close()
        textures.append({"AssetKey": key, "DxgiFormat": fmt, "File": name})
    dump(os.path.join(folder, "manifest.json"), {"Count": len(rows), "Textures": textures})


def model(bindings, embedded=None):
    item = {"Name": "m", "Materials": [{"Shader": "00000000", "Textures": [
        {"Param": p, "Id64": key, "Name": h} for p, key, h in bindings]}]}
    if embedded is not None:
        item["Embedded"] = embedded
    return item


def build_export(tmp):
    exp = os.path.join(tmp, "assets")
    pool(os.path.join(exp, "ttpf", "pool1"), [("00000000000000AA", 71), ("00000000000000BB", 83),
                                              ("00000000000000CC", 72)])
    # the same key in a second pool with another format: conflicting
    pool(os.path.join(exp, "ttpf_pack", "pk", "pool2"), [("00000000000000CC", 99), ("00000000000000DD", 98)])
    # dictionaries: hash 11111111 once (candidate), 22222222 twice (ambiguous)
    dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 2, "Items": [
        {"Offset": 0, "Textures": [{"Hash": "11111111", "Format": "BC7_UNORM_SRGB", "Key": None},
                                   {"Hash": "22222222", "Format": "BC1_UNORM", "Key": None}]},
        {"Offset": 4096, "Textures": [{"Hash": "22222222", "Format": "BC1_UNORM", "Key": None}]}]})
    emb = {"Textures": [{"Hash": "33333333", "Format": "BC5_UNORM", "Key": "0000000000000E01"}]}
    dump(os.path.join(exp, "lv", "models", "manifest.json"), {"Count": 3, "Items": [
        model([(BUMP, "00000000000000AA", "AAAAAAAA"), (BUMP, "00000000000000BB", "BBBBBBBB"),
               (DIFF, "00000000000000CC", "CCCCCCCC"), (DIFF, "00000000000000F1", "11111111")]),
        model([(BUMP, "00000000000000AA", "AAAAAAAA"), (DIFF, "00000000000000F2", "22222222"),
               (DIFF, "00000000000000F3", "99999999"), (DIFF, None, "11111111")]),
        model([(SPEC, "0000000000000E01", "33333333"), (None, "00000000000000AA", "AAAAAAAA")], embedded=emb)]})
    # a second archive whose model references a key only the first archive's embedded texture knows
    dump(os.path.join(exp, "other", "models", "manifest.json"), {"Items": [
        model([(SPEC, "00000000000000DD", "DDDDDDDD")])], "Count": 1})
    return exp


class Census(unittest.TestCase):
    def test_per_parameter_resolution_and_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = tsf.census(build_export(tmp))
        self.assertEqual((r["models"], r["bindings"], r["bindings_without_key"], r["bindings_without_param"]),
                         (4, 10, 1, 1))
        p = r["parameters"]
        self.assertEqual(list(p), [DIFF, BUMP, SPEC])       # by bindings, then hash
        b = p[BUMP]
        self.assertEqual((b["bindings"], b["distinct_textures"], b["exact"]), (3, 2, 2))  # AA counted once
        self.assertEqual(b["formats"], {"71": 1, "83": 1})
        d = p[DIFF]
        self.assertEqual((d["bindings"], d["distinct_textures"]), (5, 4))
        self.assertEqual((d["exact"], d["exact_conflicting"], d["candidate"], d["ambiguous"], d["unresolved"]),
                         (1, 1, 1, 1, 1))
        self.assertEqual(d["formats"], {})                  # the one exact key has two formats
        self.assertEqual(d["candidate_formats"], {"99": 1})  # BC7_UNORM_SRGB by name
        s = p[SPEC]
        self.assertEqual((s["exact"], s["formats"]), (2, {"83": 1, "98": 1}))  # embedded Key and pack pool

    def test_archive_filter_counts_only_those_bindings_but_resolves_against_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            r = tsf.census(exp, {"other"})
        self.assertEqual((r["models"], r["bindings"]), (1, 1))
        self.assertEqual(r["parameters"][SPEC]["formats"], {"98": 1})

    def test_embedded_texture_in_a_filtered_out_archive_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            dump(os.path.join(exp, "other", "models", "manifest.json"), {"Items": [
                model([(SPEC, "0000000000000E01", "33333333")])], "Count": 1})
            r = tsf.census(exp, {"other"})
        self.assertEqual((r["parameters"][SPEC]["exact"], r["parameters"][SPEC]["formats"]), (1, {"83": 1}))

    def test_format_and_param_normalisation(self):
        self.assertEqual(tsf.dxgi("BC7_UNORM_SRGB"), 99)
        self.assertEqual(tsf.dxgi("DXGI_FORMAT_BC1_UNORM"), 71)
        self.assertEqual(tsf.dxgi(" 83 "), 83)
        self.assertEqual(tsf.dxgi("NOT_A_FORMAT"), "NOT_A_FORMAT")
        self.assertIsNone(tsf.dxgi(None))
        self.assertIsNone(tsf.dxgi(True))
        self.assertEqual(tsf.param_hash("1e77bf4c"), BUMP)
        self.assertEqual(tsf.param_hash("0x1E77BF4C"), BUMP)
        self.assertEqual(tsf.param_hash(0x1E77BF4C), BUMP)
        self.assertIsNone(tsf.param_hash("bumptex"))


class Cli(unittest.TestCase):
    @staticmethod
    def run_main(argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = tsf.main(argv)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_prints_counts_and_writes_lf_json_outside_the_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            report = os.path.join(tmp, "r.json")
            code, out, _ = self.run_main([exp, "--json", report])
            self.assertEqual(code, 0)
            self.assertIn(f"0x{DIFF} | 5 | 4 | 1 (1 conflicting) | 1 | 1 | 1 | - | 99: 1", out)
            with open(report, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r", raw)
            self.assertEqual(json.loads(raw)["bindings"], 10)

    def test_json_inside_the_export_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            code, _, err = self.run_main([exp, "--json", os.path.join(exp, "new.json")])
            self.assertEqual(code, 2)
            self.assertIn("inside the export", err)
            self.assertFalse(os.path.exists(os.path.join(exp, "new.json")))

    def test_unknown_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = self.run_main([build_export(tmp), "--archive", "nope"])
            self.assertEqual(code, 2)
            self.assertIn("nope", err)

    def test_invalid_manifest_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            with open(os.path.join(exp, "other", "models", "manifest.json"), "w", encoding="utf-8") as f:
                f.write('{"Items":[{} {}]}')
            code, _, err = self.run_main([exp])
            self.assertEqual(code, 2)
            self.assertIn("invalid manifest JSON", err)

    def test_top_limits_the_printed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = self.run_main([build_export(tmp), "--top", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(sum(line.startswith("0x") for line in out.splitlines()), 1)



def minimal(tmp, items, count=None, pools=(), emb_hash_rows=()):
    exp = os.path.join(tmp, "assets")
    for i, rows in enumerate(pools):
        pool(os.path.join(exp, "ttpf", f"p{i}"), rows)
    doc = {"Items": items}
    if count is not None:
        doc["Count"] = count
    dump(os.path.join(exp, "lv", "models", "manifest.json"), doc)
    if emb_hash_rows:
        dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 1, "Items": [
            {"Offset": 0, "Textures": list(emb_hash_rows)}]})
    return exp


class ReviewCases(unittest.TestCase):
    def test_pool_and_embedded_formats_disagreeing_is_a_counted_conflict(self):
        # kacper-serewis on #5067: a key in a pool and as an embedded texture with different formats
        emb = {"Textures": [{"Hash": "44444444", "Format": "BC7_UNORM", "Key": "00000000000000A1"}]}
        with tempfile.TemporaryDirectory() as tmp:
            exp = minimal(tmp, [model([(BUMP, "00000000000000A1", "44444444"),
                                       (BUMP, "00000000000000A2", "55555555")], embedded=emb)],
                          pools=[[("00000000000000A1", 71), ("00000000000000A2", 71)]])
            d = tsf.census(exp)["parameters"][BUMP]
        self.assertEqual((d["exact"], d["exact_conflicting"], d["exact_pool_and_embedded"]), (2, 1, 1))
        self.assertEqual(d["formats"], {"71": 1})           # only the key whose sources agree

    def test_pool_and_embedded_formats_agreeing_are_tallied_once(self):
        emb = {"Textures": [{"Hash": "44444444", "Format": "BC1_UNORM", "Key": "00000000000000A1"}]}
        with tempfile.TemporaryDirectory() as tmp:
            exp = minimal(tmp, [model([(BUMP, "00000000000000A1", "44444444")], embedded=emb)],
                          pools=[[("00000000000000A1", 71)]])
            d = tsf.census(exp)["parameters"][BUMP]
        self.assertEqual((d["exact"], d["exact_conflicting"], d["exact_pool_and_embedded"], d["formats"]),
                         (1, 0, 1, {"71": 1}))

    def test_zero_keys_and_hashes_are_never_joined(self):
        # whattheflop on #5067: zero Id64 bindings, a zero pool key, and a zero name hash on a dictionary entry
        emb = {"Textures": [{"Hash": "00000000", "Format": "BC1_UNORM", "Key": "0000000000000000", "Index": 0},
                            {"Hash": "00000000", "Format": "BC7_UNORM", "Key": "0000000000000000", "Index": 1}]}
        with tempfile.TemporaryDirectory() as tmp:
            exp = minimal(tmp, [model([(BUMP, "0000000000000000", "00000000"),
                                       (BUMP, "0000000000000000", "00000000"),
                                       (DIFF, "00000000000000B1", "00000000")], embedded=emb)],
                          pools=[[("0000000000000000", 71)]],
                          emb_hash_rows=[{"Hash": "00000000", "Format": "BC1_UNORM", "Key": None}])
            r = tsf.census(exp)
        self.assertEqual((r["bindings"], r["bindings_without_key"]), (3, 2))
        self.assertEqual(r["parameters"][BUMP]["distinct_textures"], 0)
        d = r["parameters"][DIFF]
        self.assertEqual((d["exact"], d["candidate"], d["ambiguous"], d["unresolved"]), (0, 0, 0, 1))

    def test_pool_row_without_an_exported_file_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = minimal(tmp, [model([(BUMP, "00000000000000C1", "1"), (BUMP, "00000000000000C2", "2"),
                                       (BUMP, "00000000000000C3", "3")])])
            folder = os.path.join(exp, "ttpf", "nofile")
            pool(folder, [("00000000000000C1", 71), ("00000000000000C2", 71), ("00000000000000C3", 71)], files=False)
            open(os.path.join(folder, "t2.dds"), "wb").close()
            with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
                doc = json.load(f)
            doc["Textures"][0]["File"] = None
            dump(os.path.join(folder, "manifest.json"), doc)
            d = tsf.census(exp)["parameters"][BUMP]
        self.assertEqual((d["exact"], d["exact_pool_without_file"]), (3, 2))   # null File, missing file

    def test_coverage_gap_is_reported_and_extra_rows_are_a_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = minimal(tmp, [model([(BUMP, "00000000000000A1", "1")])], count=2)
            code, out, _ = Cli.run_main([exp])
            self.assertEqual(code, 0)
            self.assertIn("coverage lv/models: 1 of 2 rows (1 not exported)", out)
        with tempfile.TemporaryDirectory() as tmp:
            exp = minimal(tmp, [model([]), model([])], count=1)
            code, _, err = Cli.run_main([exp])
            self.assertEqual(code, 1)
            self.assertIn("more than the manifest Count", err)

    def test_empty_export_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "assets", "lv"))
            code, _, err = Cli.run_main([os.path.join(tmp, "assets")])
        self.assertEqual(code, 2)
        self.assertIn("no <archive>/models/manifest.json", err)

    def test_archive_name_must_match_exactly(self):
        # on Windows a case variant passed a file-exists preflight and then selected no models
        with tempfile.TemporaryDirectory() as tmp:
            exp = minimal(tmp, [model([(BUMP, "00000000000000A1", "1")])])
            code, _, err = Cli.run_main([exp, "--archive", "LV"])
            self.assertEqual(code, 2)
            self.assertIn("--archive LV", err)
            code, out, _ = Cli.run_main([exp, "--archive", "lv"])
            self.assertEqual(code, 0)
            self.assertIn("models 1,", out)

if __name__ == "__main__":
    unittest.main()

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


def model(bindings, embedded=None):
    item = {"Name": "m", "Materials": [{"Shader": "00000000", "Textures": [
        {"Param": p, "Id64": key, "Name": h} for p, key, h in bindings]}]}
    if embedded is not None:
        item["Embedded"] = embedded
    return item


def build_export(tmp):
    exp = os.path.join(tmp, "assets")
    dump(os.path.join(exp, "ttpf", "pool1", "manifest.json"), {"Count": 3, "Textures": [
        {"AssetKey": "00000000000000AA", "DxgiFormat": 71}, {"AssetKey": "00000000000000BB", "DxgiFormat": 83},
        {"AssetKey": "00000000000000CC", "DxgiFormat": 72}]})
    # the same key in a second pool with another format: conflicting
    dump(os.path.join(exp, "ttpf_pack", "pk", "pool2", "manifest.json"), {"Count": 2, "Textures": [
        {"AssetKey": "00000000000000CC", "DxgiFormat": 99}, {"AssetKey": "00000000000000DD", "DxgiFormat": 98}]})
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
    def run_main(self, argv):
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


if __name__ == "__main__":
    unittest.main()

"""Synthetic tests for tools/texture_dicts.py (no game data)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import texture_dicts as td  # noqa: E402
from hashdict import joaat  # noqa: E402


def dump(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)


def dictionary(offset, names, fmt="BC1_UNORM"):
    return {"Dir": f"dictionary_at_{offset:016X}", "Offset": offset,
            "Textures": [{"Index": i, "Name": n, "Hash": f"{joaat(n):08X}", "Format": fmt, "Width": 4, "Height": 4,
                          "MipCount": 1, "Key": None} for i, n in enumerate(names)]}


def model(name, bindings):
    return {"Name": name, "Materials": [{"Shader": "00000000", "Textures": [
        {"Param": "C8E8C282", "Id64": key, "Name": tex} for key, tex in bindings]}]}


def build_export(tmp):
    exp = os.path.join(tmp, "assets")
    # dictionaries in one archive: two single-texture dictionaries at page+16, one with two textures
    dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 3, "Items": [
        dictionary(4096 * 10 + 16, ["dict_a"]), dictionary(4096 * 11 + 16, ["dict_b"]),
        dictionary(4096 * 12 + 16, ["dict_c", "dict_d"], fmt="BC7_UNORM")]})
    # pools: one under ttpf, one under ttpf_pack/<pack>/
    dump(os.path.join(exp, "ttpf", "pool1", "manifest.json"), {"Count": 2, "Textures": [
        {"AssetKey": "00000000000000AA", "DxgiFormat": 71}, {"AssetKey": "00000000000000BB", "DxgiFormat": 71}]})
    dump(os.path.join(exp, "ttpf_pack", "packx", "pool2", "manifest.json"), {"Count": 1, "Textures": [
        {"AssetKey": "00000000000000CC", "DxgiFormat": 98}]})
    # models: references resolving to pool only, dictionary only, both, neither; a repeat; a binding without key
    m3 = model("m3", [("0000000000000F01", "emb_by_key"), ("0000000000000F02", "emb_by_name")])
    m3["Embedded"] = {"Textures": [{"Name": "other_name", "Hash": "0", "Key": "0000000000000F01"},
                                   {"Name": "emb_by_name", "Hash": f"{joaat('emb_by_name'):08X}", "Key": None}]}
    dump(os.path.join(exp, "lv", "models", "manifest.json"), {"Count": 3, "Items": [
        model("m1", [("00000000000000AA", "pool_tex"), ("0000000000000D01", "dict_a"),
                     ("00000000000000BB", "dict_b"), ("0000000000000E01", "lost_tex")]),
        model("m2", [("00000000000000AA", "pool_tex"), (None, "no_key"), ("00000000000000CC", "pack_tex")]), m3]})
    return exp


class CensusTests(unittest.TestCase):
    def test_dictionary_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = td.census(build_export(tmp))
            d = r["dictionaries"]["lv"]
            self.assertEqual((d["dictionaries"], d["textures"], d["single_texture"], d["empty"]), (3, 4, 2, 0))
            self.assertEqual(d["distinct_hashes"], 4)
            self.assertEqual(d["hash_is_joaat_of_name"], 4)
            self.assertEqual(d["with_64bit_key"], 0)
            self.assertEqual(d["formats"], {"BC1_UNORM": 2, "BC7_UNORM": 2})
            self.assertEqual(d["offset_mod_4096"], {16: 3})
            self.assertEqual(r["problems"], [])

    def test_pools_include_pack_pools(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = td.census(build_export(tmp))
            self.assertEqual(r["pools"], {"manifests": 2, "textures": 3, "distinct_keys": 3})

    def test_reference_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = td.census(build_export(tmp))["references"]["lv"]
            self.assertEqual(s["models"], 3)
            self.assertEqual(s["bindings"], 9)
            self.assertEqual(s["bindings_without_key"], 1)
            self.assertEqual(s["distinct_references"], 7)
            # AA, BB, CC in pools; F01 by embedded key, F02 by embedded name; D01 and BB by dictionary name; E01 nowhere
            self.assertEqual(s["resolution"], {"pool": 3, "embedded": 2, "dictionary": 2, "unresolved": 1})
            self.assertEqual(s["dictionary_by_archive"], {"lv": 2})

    def test_count_mismatches_are_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 9, "Items": [dictionary(16, ["x"])]})
            r = td.census(exp)
            self.assertTrue(any("manifest Count 9" in p for p in r["problems"]))

    def test_dot_named_archive_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            dump(os.path.join(exp, ".hidden", "textures", "manifest.json"), {"Count": 1, "Items": [dictionary(16, ["h"])]})
            self.assertIn(".hidden", td.census(exp)["dictionaries"])

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
                for chunk in (1, 2, 5, td.CHUNK):
                    with self.assertRaises(ValueError, msg=(text, chunk)):
                        list(td.iter_items(path, chunk))

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
                    gen = td.iter_items(path, chunk)
                    while True:
                        try:
                            next(gen)
                        except StopIteration as stop:
                            self.assertEqual(stop.value, want, (text, chunk))
                            break

    def test_key64_normalisation(self):
        self.assertEqual(td.key64("0xAbC"), "0000000000000abc")
        self.assertEqual(td.key64(255), "00000000000000ff")
        self.assertIsNone(td.key64("not hex"))
        self.assertIsNone(td.key64(None))


class CliTests(unittest.TestCase):
    def run_cli(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = td.main(args)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_report_written_outside_the_export_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            report = os.path.join(tmp, "report.json")
            code, out, _ = self.run_cli([exp, "--json", report])
            self.assertEqual(code, 0)
            self.assertIn("texture pools: 2 manifests, 3 textures", out)
            with open(report, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["references"]["total"]["distinct_references"], 7)
            inside = os.path.join(exp, "lv", "textures", "manifest.json")
            code, _, err = self.run_cli([exp, "--json", inside])
            self.assertNotEqual(code, 0)
            self.assertIn("inside the export folder", err)

    def test_output_has_no_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = self.run_cli([build_export(tmp)])
            for name in ("dict_a", "pool_tex", "lost_tex", "m1", "emb_by_name"):
                self.assertNotIn(name, out)


if __name__ == "__main__":
    unittest.main()

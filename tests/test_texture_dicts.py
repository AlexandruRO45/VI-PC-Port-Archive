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
            # exact 64-bit: AA, BB, CC (pools) and F01 (embedded Key); name-hash candidates only: D01 (dictionary),
            # F02 (embedded stored Hash); E01 nowhere. The partition adds up; the per-source counts overlap (BB).
            self.assertEqual(s["resolution"], {"exact 64-bit key": 4, "32-bit name-hash candidate only": 2,
                                               "  of which ambiguous (hash on more than one texture)": 0,
                                               "unresolved": 1})
            self.assertEqual(s["by_source"], {"pool key": 3, "embedded key": 1, "pool and embedded key": 0,
                                              "embedded name hash": 1, "dictionary name hash": 2})
            self.assertEqual(s["dictionary_candidates_by_archive"], {"lv": 2})

    def test_stored_hex_name_is_compared_not_hashed_again(self):
        # review on #4980 (whattheflop, kacper-serewis): rpf9dump writes a binding's Name as NameHash.ToString("X8")
        with tempfile.TemporaryDirectory() as tmp:
            exp = os.path.join(tmp, "assets")
            decoy = {"Index": 1, "Name": "decoy", "Hash": f"{joaat('6BACDAD2'):08X}", "Format": "BC1_UNORM", "Key": None}
            real = {"Index": 0, "Name": "real", "Hash": "6BACDAD2", "Format": "BC1_UNORM", "Key": None}
            dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 1, "Items": [
                {"Offset": 16, "Textures": [real, decoy]}]})
            dump(os.path.join(exp, "lv", "models", "manifest.json"), {"Count": 1, "Items": [
                {"Name": "m", "Materials": [{"Textures": [{"Param": "C8E8C282", "Id64": "00000000000000A1",
                                                           "Name": "6BACDAD2"}]}]}]})
            r = td.census(exp)
            s = r["references"]["lv"]
            self.assertEqual(s["by_source"]["dictionary name hash"], 1)
            self.assertEqual(s["resolution"]["32-bit name-hash candidate only"], 1)
            self.assertEqual(s["resolution"]["unresolved"], 0)
            self.assertEqual(r["material_name_hashes"], {"stored": 1})
            # with only the decoy (whose hash is joaat of the hex text), the binding must not match
            dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 1, "Items": [
                {"Offset": 16, "Textures": [decoy]}]})
            s = td.census(exp)["references"]["lv"]
            self.assertEqual((s["by_source"]["dictionary name hash"], s["resolution"]["unresolved"]), (0, 1))

    def test_a_hash_on_two_entries_is_an_ambiguous_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = os.path.join(tmp, "assets")
            a = {"Index": 0, "Name": "one", "Hash": "11111111", "Format": "BC1_UNORM", "Key": None}
            b = {"Index": 0, "Name": "two", "Hash": "11111111", "Format": "BC7_UNORM", "Key": None}
            dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 2, "Items": [
                {"Offset": 16, "Textures": [a]}, {"Offset": 4112, "Textures": [b]}]})
            dump(os.path.join(exp, "lv", "models", "manifest.json"), {"Count": 1, "Items": [
                {"Name": "m", "Materials": [{"Textures": [{"Id64": "00000000000000B2", "Name": "11111111"}]}]}]})
            s = td.census(exp)["references"]["lv"]
            self.assertEqual(s["resolution"]["32-bit name-hash candidate only"], 1)
            self.assertEqual(s["resolution"]["  of which ambiguous (hash on more than one texture)"], 1)

    def test_one_texture_embedded_in_two_models_is_not_ambiguous(self):
        # tedin7 on #4980: rows were counted, so one texture embedded twice looked like two textures
        emb = {"Textures": [{"Name": "e", "Hash": "11111111", "Format": "BC1_UNORM", "Key": "00000000000000A1"}]}
        ref = {"Name": "r", "Materials": [{"Textures": [{"Id64": "00000000000000B2", "Name": "11111111"}]}]}
        for copies in (1, 2):
            with tempfile.TemporaryDirectory() as tmp:
                exp = os.path.join(tmp, "assets")
                items = [dict({"Name": f"m{i}", "Materials": []}, Embedded=emb) for i in range(copies)] + [ref]
                dump(os.path.join(exp, "lv", "models", "manifest.json"), {"Count": len(items), "Items": items})
                s = td.census(exp)["references"]["lv"]
                self.assertEqual(s["resolution"]["32-bit name-hash candidate only"], 1, copies)
                self.assertEqual(s["resolution"]["  of which ambiguous (hash on more than one texture)"], 0, copies)

    def test_count_above_rows_is_coverage_rows_above_count_a_problem(self):
        # review on #4980 (whattheflop): rpf9dump can leave out rows that failed to decode
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 9, "Items": [dictionary(16, ["x"])]})
            r = td.census(exp)
            self.assertEqual(r["problems"], [])
            self.assertEqual(r["coverage"]["lv/textures"], {"declared": 9, "rows": 1, "missing": 8})
            dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 0, "Items": [dictionary(16, ["x"])]})
            r = td.census(exp)
            self.assertTrue(any("more than the manifest Count 0" in p for p in r["problems"]))

    def test_number_cut_at_a_read_boundary(self):
        # review on #4980 (shdann): a read that ends at the e of 1e3 must not give 1
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            for text in ('{"scale":1e3,"Items":[{}],"Count":1}', '{"scale":-1.5E-3,"Items":[{}],"Count":1}',
                         '{"Items":[{}],"Count":1,"scale":12e+3}'):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                for chunk in range(1, len(text) + 2):
                    gen = td.iter_items(path, chunk)
                    while True:
                        try:
                            next(gen)
                        except StopIteration as stop:
                            self.assertEqual(stop.value, 1, (text, chunk))
                            break

    @unittest.skipUnless(os.name == "nt", "drive letters")
    def test_other_drive_is_outside(self):
        self.assertFalse(td.inside(r"Z:\reports\r.json", r"C:\work\assets"))
        self.assertTrue(td.inside(r"C:\work\assets\lv\r.json", r"C:\work\assets"))
        self.assertFalse(td.inside(r"C:\work\assets2\r.json", r"C:\work\assets"))

    def test_dot_named_archive_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            dump(os.path.join(exp, ".hidden", "textures", "manifest.json"), {"Count": 1, "Items": [dictionary(16, ["h"])]})
            self.assertIn(".hidden", td.census(exp)["dictionaries"])

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
                gen = td.iter_items(path)
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
                for chunk in (1, 3, td.CHUNK):
                    with self.assertRaises(ValueError, msg=(text, chunk)):
                        list(td.iter_items(path, chunk))

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

    def test_non_object_entries_exit_2(self):
        # Flyander on #4946, same pattern here: entries that are not objects are input errors, not tracebacks
        cases = [
            (("lv", "models", "manifest.json"), {"Items": [1]}),
            (("lv", "models", "manifest.json"), {"Items": [{"Materials": [7]}]}),
            (("lv", "models", "manifest.json"), {"Items": [{"Materials": [{"Textures": {"Id64": "1"}}]}]}),
            (("lv", "textures", "manifest.json"), {"Items": [{"Textures": ["x"]}]}),
            (("lv", "textures", "manifest.json"), []),
            (("ttpf", "pool1", "manifest.json"), {"Textures": [None]}),
            # LILILI2007's falsy non-arrays, and Embedded that is neither absent, a list nor an object
            (("lv", "models", "manifest.json"), {"Items": [{"Materials": {}}]}),
            (("lv", "models", "manifest.json"), {"Items": [{"Materials": ""}]}),
            (("lv", "models", "manifest.json"), {"Items": [{"Materials": 0}]}),
            (("lv", "models", "manifest.json"), {"Items": [{"Materials": False}]}),
            (("lv", "models", "manifest.json"), {"Items": [{"Embedded": 5}]}),
            (("lv", "models", "manifest.json"), {"Items": [{"Embedded": {"Textures": {}}}]}),
            (("lv", "models", "manifest.json"), {"Items": [{"Embedded": [3]}]}),
            (("lv", "textures", "manifest.json"), {"Items": {}}),
        ]
        for rel, doc in cases:
            with tempfile.TemporaryDirectory() as tmp:
                exp = build_export(tmp)
                dump(os.path.join(exp, *rel), doc)
                code, _, err = self.run_cli([exp])
                self.assertEqual(code, 2, (rel, doc, err))
                self.assertIn("invalid manifest JSON", err)
                self.assertNotIn("Traceback", err)

    def test_non_string_fields_are_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            dump(os.path.join(exp, "lv", "textures", "manifest.json"), {"Count": 1, "Items": [
                {"Offset": 0, "Textures": [{"Name": ["n"], "Hash": ["h"], "Format": {"f": 1}, "Key": [1]}]}]})
            code, _, err = self.run_cli([exp])
            self.assertEqual(code, 0, err)

    def test_report_written_outside_the_export_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = build_export(tmp)
            report = os.path.join(tmp, "report.json")
            code, out, _ = self.run_cli([exp, "--json", report])
            self.assertEqual(code, 0)
            self.assertIn("texture pools: 2 manifests, 3 textures", out)
            with open(report, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["references"]["total"]["distinct_references"], 7)
            code, _, _ = self.run_cli([exp, "--json", report])     # refreshing the same report works
            self.assertEqual(code, 0)
            with open(report, "rb") as f:
                self.assertNotIn(b"\r", f.read())                   # LF only
            for inside in (os.path.join(exp, "lv", "textures", "manifest.json"),   # existing input
                           os.path.join(exp, "new_report.json")):                  # new path inside
                code, _, err = self.run_cli([exp, "--json", inside])
                self.assertNotEqual(code, 0)
                self.assertIn("inside the export folder", err)
            self.assertFalse(os.path.exists(os.path.join(exp, "new_report.json")))

    def test_output_has_no_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = self.run_cli([build_export(tmp)])
            for name in ("dict_a", "pool_tex", "lost_tex", "m1", "emb_by_name"):
                self.assertNotIn(name, out)


if __name__ == "__main__":
    unittest.main()

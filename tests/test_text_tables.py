"""Tests for tools/text_tables.py, built from synthetic resources and conversation data (no game data).

    python -I tests/test_text_tables.py
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "text_tables.py")
    spec = importlib.util.spec_from_file_location("text_tables_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_module()


def ptr(off):
    return struct.pack("<II", 0x50000000 | off, 0)


def make_resource(strings, pad=0):
    """A logical resource: 0DRP at +0x10, a descriptor, then hash table, index array and string blob.

    ``strings`` maps label -> text. ``pad`` adds bytes before the tables so they can cross a 64 KiB boundary.
    """
    labels = list(strings)
    blob = bytearray()
    blob_offsets = []
    for label in labels:
        blob_offsets.append(len(blob))
        blob += strings[label].encode("utf-8") + b"\0"
    head = bytearray(0x10) + b"0DRP" + bytes(0x0C)
    desc_at = len(head)
    desc_len = 8 + 4 + 4 + 8 + 4 + 4 + 8 + 4 + 4 + 8
    index_at = desc_at + desc_len + pad
    hash_at = index_at + 16 * len(labels)
    blob_at = hash_at + 16 * len(labels)
    desc = (ptr(index_at) + struct.pack("<II", 0, 0x10) + ptr(0) + b"\x01\x01\x00\x00"
            + struct.pack("<I", 16 * len(labels)) + ptr(hash_at) + struct.pack("<II", 0x10, len(blob))
            + ptr(blob_at))
    assert len(desc) == desc_len
    index = b"".join(struct.pack("<QQ", 0, off << 20) for off in blob_offsets)
    hashes = bytearray()
    for i, label in enumerate(labels):
        entry = bytearray(16)
        struct.pack_into("<II", entry, 0, T.joaat(label), 0)
        entry[8] = 1
        entry[11:14] = i.to_bytes(3, "little")
        hashes += entry
    return bytes(head + desc + bytes(pad) + index + hashes + blob)


def write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)
    return path


EN = {"ZZ_LINE_A_1": "This is the first line and you know it", "ZZ_LINE_B_2": "What is that, the second?",
      "ZZ_MENU_TITLE": "the menu"}
DE = {"ZZ_LINE_A_1": "Das ist die erste Zeile und du weißt es", "ZZ_LINE_B_2": "Was ist das, die zweite?",
      "ZZ_MENU_TITLE": "das Menü"}
JA = {"ZZ_LINE_A_1": "これは最初の行です"}
CONV = b"head\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_A_1\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_B_2\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_A_1\x00"


class TestTables(unittest.TestCase):
    def test_decode(self):
        tables = list(T.string_tables(make_resource(EN)))
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0], {T.joaat(k): v for k, v in EN.items()})

    def test_bad_bounds_skipped(self):
        res = bytearray(make_resource(EN))
        at = res.index(b"\x01\x01\x00\x00") + 4
        struct.pack_into("<I", res, at, len(res) * 2)      # hash table larger than the resource
        self.assertEqual(list(T.string_tables(bytes(res))), [])

    def test_language(self):
        self.assertEqual(T.language(EN.values()), "en")
        self.assertEqual(T.language(DE.values()), "de")
        self.assertEqual(T.language(JA.values()), "ja")
        self.assertEqual(T.language(["Привет"]), "ru")
        self.assertEqual(T.language(["안녕하세요"]), "ko")
        self.assertEqual(T.language(["你好世界"]), "zh")

    def test_group(self):
        self.assertEqual(T.table_group(["BIL1_A_1", "BIL1_B_2", "BIL1_C_3", "OTHER"]), "BIL1")
        self.assertEqual(T.table_group(["AB_CD_1", "AB_CD_2", "AB_CD_3"]), "AB_CD")
        self.assertEqual(T.table_group(["A_1", "B_2"]), "")


class TestChains(unittest.TestCase):
    def carve(self, tmp, streams):
        """rpf9_carve.py layout: index.json plus one <offset>.bin per inflated stream."""
        files, off = [], 0x1000
        for data in streams:
            write(os.path.join(tmp, f"{off:012x}.bin"), data)
            files.append({"off": off, "stored": len(data), "size": len(data), "pre": "", "head": data[:16].hex()})
            off += 0x20000
        with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"files": files, "summary": {}}, fh)

    def test_resource_across_64k_streams(self):
        res = make_resource(EN, pad=T.BLOCK)            # tables start past the first 64 KiB
        streams = [res[:T.BLOCK], res[T.BLOCK:], make_resource(DE), CONV]
        with tempfile.TemporaryDirectory() as tmp:
            self.carve(tmp, streams)
            scan = T.Scan()
            for label, off, data in T.blobs(tmp):
                scan.add(label, off, data)
        self.assertEqual(sorted(lang for _s, _o, lang, _t in scan.tables), ["de", "en"])
        self.assertEqual(scan.tables[0][1], 0x1000)

    def test_resource_across_a_128k_first_stream(self):
        # A resource's first stream can be any whole number of 64 KiB blocks (docs/page_directory.md): 41 data.rpf
        # resources start with a 128 or 192 KiB stream whose descriptors end in the next stream.
        res = make_resource(EN, pad=2 * T.BLOCK)        # tables start past the first 128 KiB
        streams = [res[:2 * T.BLOCK], res[2 * T.BLOCK:], make_resource(DE), CONV]
        with tempfile.TemporaryDirectory() as tmp:
            self.carve(tmp, streams)
            scan = T.Scan()
            for label, off, data in T.blobs(tmp):
                scan.add(label, off, data)
        self.assertEqual(sorted(lang for _s, _o, lang, _t in scan.tables), ["de", "en"])

    def test_whole_block_streams_continue_and_others_end_the_chain(self):
        for first, joined in ((T.BLOCK, True), (2 * T.BLOCK, True), (3 * T.BLOCK, True),
                              (T.BLOCK + 16, False), (T.BLOCK - 16, False)):
            with self.subTest(first=first):
                rows = [{"off": 0, "size": first}, {"off": 1, "size": 100}]
                data = {0: bytes(first), 1: b"b" * 100}
                offsets = [o for o, _d in T._chains(rows, lambda r: data[r["off"]])]
                self.assertEqual(offsets, [0] if joined else [0, 1])

    def test_short_stream_ends_the_chain(self):
        rows = [{"off": 0, "size": 100}, {"off": 1, "size": 50}]
        data = {0: b"a" * 100, 1: b"b" * 50}
        self.assertEqual([o for o, _d in T._chains(rows, lambda r: data[r["off"]])], [0, 1])

    def test_new_resource_ends_the_chain(self):
        rows = [{"off": 0, "size": T.BLOCK}, {"off": 1, "size": 10}]
        data = {0: bytes(T.BLOCK), 1: bytes(0x10) + b"0DRP"}
        self.assertEqual([o for o, _d in T._chains(rows, lambda r: data[r["off"]])], [0, 1])


class TestDialogue(unittest.TestCase):
    def test_waves_and_text(self):
        scan = T.Scan()
        scan.add("a", 0, make_resource(EN))
        scan.add("b", 0, make_resource(DE))
        scan.add("c", 0, CONV)
        rows = list(scan.dialogue_rows())
        self.assertEqual([(r["wave"], r["label"], r["speaker"]) for r in rows],
                         [("zz_rootau_abcd_01", "ZZ_LINE_A_1", "A"), ("zz_rootau_abcd_02", "ZZ_LINE_B_2", "B")])
        self.assertEqual(rows[0]["text_en"], EN["ZZ_LINE_A_1"])
        self.assertEqual(rows[1]["text_de"], DE["ZZ_LINE_B_2"])
        self.assertEqual(scan.languages(), ["en", "de"])

    def test_lowercase_in_label(self):
        scan = T.Scan()
        scan.add("c", 0, b"ZZ_KEYAU_ABCD\x00ZZ_TIMELEFT_90s_1\x00")
        self.assertEqual(scan.pairs["ZZ_KEYAU_ABCD"], ["ZZ_TIMELEFT_90s_1"])


def make_variant_resource(groups, pad_before_variants=0):
    """Build a synthetic 0DRP string table with 0x956a9df5 groups and 0x10075946 variant records."""
    blob = bytearray()
    var_bytes = bytearray(pad_before_variants)
    grp_bytes = bytearray()
    hash_bytes = bytearray()
    start_rec = len(var_bytes) // 32
    for g_idx, (label, g_src_a, g_src_b, variants) in enumerate(groups):
        rec_byte_off = len(var_bytes)
        for text, gender, plurality, case in variants:
            str_off = len(blob)
            blob += text.encode("utf-8") + b"\0"
            vrec = bytearray(32)
            struct.pack_into("<QBBB", vrec, 0, (str_off << 20) | 4, gender, plurality, case)
            var_bytes += vrec
        grec = bytearray(32)
        struct.pack_into("<IIIIQ", grec, 0, 0, 0, len(variants), len(variants), (rec_byte_off << 20) | 6)
        grec[26] = g_src_a
        grec[27] = g_src_b
        grp_bytes += grec

        hent = bytearray(16)
        struct.pack_into("<II", hent, 0, T.joaat(label), 0)
        hent[8] = 5  # 1-based q38 block index of 0x956a9df5
        hent[11:14] = (g_idx * 2).to_bytes(3, "little")
        hash_bytes += hent

    head = bytearray(0x10) + b"0DRP" + bytes(0x0C)
    desc_at = len(head)
    desc_len = 0x28 + 16 * 3
    index_at = desc_at + desc_len
    hash_at = index_at + 16
    blob_at = hash_at + len(hash_bytes)
    grp_at = blob_at + len(blob)
    var_at = grp_at + len(grp_bytes)
    desc = (ptr(index_at) + struct.pack("<II", 0, 0x10) + ptr(0) + b"\x01\x01\x00\x00"
            + struct.pack("<I", len(hash_bytes)) + ptr(hash_at)
            + struct.pack("<II", 0x10, len(blob)) + ptr(blob_at)
            + struct.pack("<II", T.VARIANT_GROUP_TAG, len(grp_bytes)) + ptr(grp_at)
            + struct.pack("<II", T.VARIANT_RECORD_TAG, len(var_bytes)) + ptr(var_at))
    return bytes(head + desc + bytes(16) + hash_bytes + blob + grp_bytes + var_bytes)


class TestVariants(unittest.TestCase):
    def test_variant_groups_past_4k_byte_offset(self):
        # 130 dummy records push byte offset to 4160 (>= 4096), verifying (ptr4 >> 20) is a byte offset.
        res = make_variant_resource([
            ("ZZ_NOUN_SG_PL", 12, 0, [
                ("un amigo", 1, 1, 0),
                ("una amiga", 2, 1, 0),
                ("unos amigos", 1, 3, 2),
            ]),
        ], pad_before_variants=130 * 32)
        tables = list(T.variant_tables(res))
        self.assertEqual(len(tables), 1)
        items = tables[0][T.joaat("ZZ_NOUN_SG_PL")]
        self.assertEqual(
            [(x["text"], x["gender"], x["plurality"], x["case"], x["source_a"]) for x in items],
            [
                ("un amigo", "Male", "Singular", "None", "Target"),
                ("una amiga", "Female", "Singular", "None", "Target"),
                ("unos amigos", "Male", "Many", "Case2", "Target"),
            ],
        )

    def test_variants_cli_csv(self):
        res = make_variant_resource([("ZZ_GREET", 10, 11, [("Bonjour", 0, 1, 0)])])
        with tempfile.TemporaryDirectory() as tmp:
            src = write(os.path.join(tmp, "fr.bin"), res)
            out = os.path.join(tmp, "variants.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = T.main([src, "--variants", out])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["lang"], rows[0]["gender"], rows[0]["plurality"], rows[0]["case"],
                              rows[0]["source_a"], rows[0]["source_b"], rows[0]["text"]),
                             ("xx", "None", "Singular", "None", "Player", "Speaker", "Bonjour"))

    def test_variants_uses_resolved_archive_language(self):
        # Short variant tables ("Bonjour") guess "xx" alone; when Scan provides archive_lang, variants inherit it.
        # Subclass instead of assigning scan.archive_lang because on main Scan.archive_lang is a read-only @property.
        class _ScanWithArchiveLang(T.VariantScan):
            @property
            def archive_lang(self):
                return {("arc", 0x1000): "fr"}

        res = make_variant_resource([("ZZ_GREET", 10, 11, [("Bonjour", 0, 1, 0)])])
        scan = _ScanWithArchiveLang()
        scan.add("src", 0x1100, res, archive=("arc", 0x1000))
        self.assertEqual([(s, o, lang) for s, o, lang, _vt in scan.variants], [("src", 0x1100, "fr")])

    def test_variants_read_the_source_once(self):
        # A source that disappears as soon as its scan ends must not be reopened for the variant rows: they
        # come from the bytes already read. A second read used to be an uncaught FileNotFoundError, raised
        # after the other reports had been staged.
        res = make_variant_resource([("ZZ_GREET", 10, 11, [("Bonjour", 0, 1, 0)])])
        with tempfile.TemporaryDirectory() as tmp:
            src = write(os.path.join(tmp, "fr.bin"), res)
            out = os.path.join(tmp, "variants.csv")
            read, scans = T.blobs, []

            def once(path):
                scans.append(path)
                try:
                    yield from read(path)
                finally:
                    os.remove(path)

            with patch.object(T, "blobs", once), contextlib.redirect_stdout(io.StringIO()):
                rc = T.main([src, "--variants", out])
            self.assertEqual(rc, 0)
            self.assertEqual(len(scans), 1)
            with open(out, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([row["text"] for row in rows], ["Bonjour"])

    def test_variants_read_error_is_the_input_error_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write(os.path.join(tmp, "fr.bin"), make_variant_resource(
                [("ZZ_GREET", 10, 11, [("Bonjour", 0, 1, 0)])]))
            out = os.path.join(tmp, "variants.csv")

            def vanishing(path):
                raise OSError("source vanished")
                yield  # unreachable; keeps this a generator like blobs()

            with patch.object(T, "blobs", vanishing), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(T.main([src, "--variants", out]), 2)
            self.assertIn("cannot read input", err.getvalue())
            self.assertFalse(os.path.exists(out))


class TestCli(unittest.TestCase):
    def test_csv_exports_explicitly_empty_strings(self) -> None:
        strings = {"ZZ_SYNTHETIC_BLANK": "", "ZZ_SYNTHETIC_LINE": "This is the text you know"}
        with tempfile.TemporaryDirectory() as tmp:
            source = write(os.path.join(tmp, "strings.bin"), make_resource(strings))
            tables_csv = os.path.join(tmp, "tables.csv")
            strings_csv = os.path.join(tmp, "strings.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(T.main([source, "--tables", tables_csv, "--strings", strings_csv]), 0)
            with open(tables_csv, encoding="utf-8", newline="") as f:
                self.assertEqual([row["strings"] for row in csv.DictReader(f)], ["2"])
            with open(strings_csv, encoding="utf-8", newline="") as f:
                self.assertEqual({row["hash"]: row["text"] for row in csv.DictReader(f)},
                                 {f"{T.joaat(label):08x}": text for label, text in strings.items()})

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "en.bin"), make_resource(EN))
            write(os.path.join(src, "de.bin"), make_resource(DE))
            write(os.path.join(src, "conv.pso"), CONV)
            labels = os.path.join(tmp, "labels.json")
            with open(labels, "w", encoding="utf-8") as fh:
                json.dump({f"{T.joaat('zz_menu_title'):08x}": "zz_menu_title"}, fh)
            outs = {k: os.path.join(tmp, f"{k}.csv") for k in ("dialogue", "tables", "strings")}
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(T.main([src, "--labels", labels, "--dialogue", outs["dialogue"],
                                         "--tables", outs["tables"], "--strings", outs["strings"]]), 0)
            self.assertIn("2 string tables", log.getvalue())
            self.assertIn("1 dialogue keys, 2 lines", log.getvalue())
            with open(outs["tables"], encoding="utf-8", newline="") as fh:
                tables = list(csv.DictReader(fh))
            self.assertEqual({(t["lang"], t["strings"], t["resolved"], t["group"]) for t in tables},
                             {("en", "3", "3", "ZZ_LINE"), ("de", "3", "3", "ZZ_LINE")})  # 2 of 3 share ZZ_LINE
            with open(outs["strings"], encoding="utf-8", newline="") as fh:
                strings = list(csv.DictReader(fh))
            self.assertIn(("ZZ_MENU_TITLE", "das Menü"), {(s["label"], s["text"]) for s in strings})
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")],
                             "staging files must not be left behind")

    def test_failed_later_write_leaves_earlier_reports_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            dialogue = os.path.join(tmp, "dialogue.csv")
            old = "OLD GOOD REPORT\n"
            with open(dialogue, "w", encoding="utf-8") as fh:
                fh.write(old)
            tables = os.path.join(tmp, "tables.csv")
            os.mkdir(tables)                        # a later output that cannot be written
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(T.main([src, "--dialogue", dialogue, "--tables", tables]), 2)
            self.assertIn("cannot write", err.getvalue())
            with open(dialogue, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old)
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")])

    @unittest.skipUnless(os.name == "nt", "Windows denies os.replace on a file held open by a reader")
    def test_reader_holding_a_later_report_keeps_all_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            dialogue = os.path.join(tmp, "dialogue.csv")
            tables = os.path.join(tmp, "tables.csv")
            old_dialogue, old_tables = "OLD DIALOGUE\n", "OLD TABLES\n"
            for path, text in ((dialogue, old_dialogue), (tables, old_tables)):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            with open(tables, "rb"):       # a reader keeps the second output open
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(T.main([src, "--dialogue", dialogue, "--tables", tables]), 2)
            self.assertIn("cannot write", err.getvalue())
            with open(dialogue, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old_dialogue)
            with open(tables, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old_tables)
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")],
                             "staging and backup files must not be left behind")

    def test_two_outputs_naming_one_file_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            out = os.path.join(tmp, "same.csv")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as cm:
                    T.main([src, "--dialogue", out, "--tables", out])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("distinct", err.getvalue())
            self.assertFalse(os.path.exists(out), "no report may be written before the check")

    def test_missing_input(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(T.main(["/nonexistent"]), 2)

    def test_missing_parent_and_hardlinked_outputs_rejected_upfront(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write(os.path.join(tmp, "en.bin"), make_resource(EN))
            missing_parent = os.path.join(tmp, "no_such_dir", "out.csv")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                T.main([src, "--tables", missing_parent])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("parent directory does not exist", err.getvalue())

            first = write(os.path.join(tmp, "first.csv"), b"old\n")
            alias = os.path.join(tmp, "alias.csv")
            try:
                os.link(first, alias)
            except OSError:
                self.skipTest("hard links unavailable")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                T.main([src, "--dialogue", first, "--tables", alias])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("distinct", err.getvalue())

    def test_invalid_json_labels_is_clean_error_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write(os.path.join(tmp, "en.bin"), make_resource(EN))
            bad_labels = os.path.join(tmp, "bad.json")
            with open(bad_labels, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = T.main([src, "--labels", bad_labels])
            self.assertEqual(rc, 2)
            self.assertIn("cannot read input", err.getvalue())


if __name__ == "__main__":
    unittest.main()

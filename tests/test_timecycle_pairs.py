"""Synthetic unit tests for tools/timecycle_pairs.py (no game files)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import timecycle_pairs as tp  # noqa: E402

SYNTH_KF = b"""<?xml version="1.0" encoding="UTF-8"?>
<timecycle_keyframe_data version="1.000000">
  <cycle name="SYNTH_WEATHER" regions="2">
    <region name="REG_A">
      <far_clip>500 600 700</far_clip>
      <light_dir_col_r>0.1 0.5 1.0</light_dir_col_r>
      <light_dir_col_g>0.1 0.5 1.0</light_dir_col_g>
      <light_dir_col_b>0.1 0.5 1.0</light_dir_col_b>
      <postfx_exposure>-1.0 0.0 2.0</postfx_exposure>
    </region>
  </cycle>
</timecycle_keyframe_data>
"""

SYNTH_MOD = b"""<?xml version="1.0" encoding="UTF-8"?>
<timecycle_modifier_data version="1.000000">
  <modifier name="SYNTH_PLAIN_1" userFlags="0">
    <postfx_exposure>0.5 0.0</postfx_exposure>
    <light_dir_col_r>0.8 0.25</light_dir_col_r>
    <light_dir_col_g>0.4 0.25</light_dir_col_g>
    <light_dir_col_b>0.2 0.25</light_dir_col_b>
    <far_clip>100.0 500.0</far_clip>
    <empty_var></empty_var>
  </modifier>
  <modifier name="SYNTH_PLAIN_2" userFlags="0">
    <postfx_exposure>1.2 1.0</postfx_exposure>
    <far_clip>250.0 250.0</far_clip>
    <submodifiers><item>SYNTH_PLAIN_1</item></submodifiers>
  </modifier>
  <modifier name="SYNTH_AXIS" numAxes="1" numKeyframes="2">
    <axis controlValue="SYNTH_CTRL">
      <keyframe_t_value>0.0 1.0</keyframe_t_value>
      <postfx_exposure>0.0 1.0</postfx_exposure>
    </axis>
  </modifier>
</timecycle_modifier_data>
"""


class TimecyclePairsTests(unittest.TestCase):
    def test_summarize_synthetic_documents(self):
        summary = tp.summarize_documents([
            ("timecycle_keyframe_data", SYNTH_KF),
            ("timecycle_modifier_data", SYNTH_MOD),
        ])
        self.assertEqual(summary["keyframe_docs"], 1)
        self.assertEqual(summary["modifier_docs"], 1)
        self.assertEqual(summary["modifiers"], {"total": 3, "plain": 2, "axis": 1})
        self.assertEqual(summary["plain_token_counts"], {"0": 1, "2": 7})
        self.assertEqual(summary["second_number"]["eq0"], 1)
        self.assertEqual(summary["second_number"]["between_0_1"], 3)
        self.assertEqual(summary["second_number"]["eq1"], 1)
        self.assertEqual(summary["second_number"]["gt1"], 2)
        self.assertEqual(summary["rgb_triplet_b_equality"], {"frac_all_equal": 1})
        self.assertEqual(summary["per_variable_b_constancy"]["both_b0_and_b1"], 1)
        self.assertEqual(summary["per_variable_b_constancy"]["mixed_within_single_doc"], 1)
        fc = summary["vars_with_b_outside_0_1"]["far_clip"]
        self.assertEqual(fc["count"], 2)
        self.assertEqual(fc["a_eq_b"], 1)
        self.assertEqual(fc["a_lt_b"], 1)
        self.assertEqual(fc["keyframe_range"], [500.0, 700.0])

    def test_cli_xml_directory_and_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "kf.xml"), "wb") as f:
                f.write(SYNTH_KF)
            with open(os.path.join(tmp, "mod.xml"), "wb") as f:
                f.write(SYNTH_MOD)
            out_path = os.path.join(tmp, "report.json")
            rc = tp.main([tmp, "--out", out_path])
            self.assertEqual(rc, 0)
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["modifiers"]["plain"], 2)
            self.assertEqual(data["second_number"]["count"], 7)

    def write_carve(self, tmp, files):
        """A synthetic carve directory: one row per stream in index.json, its bytes in <off>.bin."""
        rows = []
        for name, data in files:
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(data)
            rows.append({"off": int(name[:12], 16), "stored": len(data), "size": len(data)})
        with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(rows)}, "files": rows}, f)

    def test_carve_dir_reads_document_without_declaration(self):
        # A complete document with no XML declaration is a "text" stream, not "decl", so
        # _iter_source_docs used to drop it: the same bytes read fine as a plain .xml file.
        # docs/gamedata.md:689 counts complete documents of that shape in the corpus.
        declless = SYNTH_MOD.split(b"\n", 1)[1]
        self.assertFalse(declless.lstrip().startswith(b"<?xml"))
        with tempfile.TemporaryDirectory() as tmp:
            self.write_carve(tmp, [("000000001000.bin", declless)])
            docs = tp.load_documents([tmp])
            self.assertEqual([r for r, _ in docs], ["timecycle_modifier_data"])
            summary = tp.summarize_documents(docs)
            self.assertEqual(summary["modifier_docs"], 1)
            self.assertEqual(summary["modifiers"], {"total": 3, "plain": 2, "axis": 1})

    def test_carve_dir_ignores_streams_that_are_not_timecycle_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_carve(tmp, [
                ("000000001000.bin", SYNTH_MOD),                    # declared document: the "decl" branch
                ("000000002000.bin", b"<synth_other_root/>"),       # standalone, but not a target root
                ("000000003000.bin", b"\x00\x01\x02\x03"),          # binary row, no tags
                ("000000004000.bin", b"just some text, no tags."),  # text row, no tags
            ])
            docs = tp.load_documents([tmp])
            self.assertEqual([r for r, _ in docs], ["timecycle_modifier_data"])

    def test_carve_and_file_agree_on_a_document_with_content_after_the_root(self):
        # A document is complete once its root is closed, whatever trails it: a trailing
        # comment or 600 newlines put the closing tag outside gi.standalone_xml()'s 512-byte
        # tail window, so the carve path dropped a document the .xml file path reads.
        body = SYNTH_MOD.split(b"\n", 1)[1]
        for suffix in (b"", b"<!-- end -->", b"\n" * 600):
            with self.subTest(trailing=suffix[:12]):
                data = body + b"\n" + suffix
                with tempfile.TemporaryDirectory() as tmp:
                    file_path = os.path.join(tmp, "doc.xml")
                    with open(file_path, "wb") as f:
                        f.write(data)
                    as_file = tp.load_documents([file_path])
                    carve = os.path.join(tmp, "carve")
                    os.makedirs(carve)
                    self.write_carve(carve, [("000000001000.bin", data)])
                    as_carve = tp.load_documents([carve])
                    self.assertEqual([r for r, _ in as_file], ["timecycle_modifier_data"])
                    self.assertEqual([r for r, _ in as_carve], [r for r, _ in as_file])
                    self.assertEqual(tp.summarize_documents(as_carve),
                                     tp.summarize_documents(as_file))
                    self.assertEqual(tp.summarize_documents(as_file)["modifier_docs"], 1)

    def test_carve_dir_joins_a_document_split_over_two_text_streams(self):
        # RPF9 splits a file over 64 KiB into whole 64 KiB streams (docs/gamedata.md:20), so
        # the first stream of a document stored without a declaration is exactly 65 536 bytes
        # and the closing root tag sits in the next stream. The old branch tested gi.
        # xml_complete() on the first stream alone and dropped the document: the same bytes
        # as a plain .xml file are one document, so the two paths must agree.
        body = SYNTH_MOD.split(b"\n", 1)[1]
        close = b"</timecycle_modifier_data>\n"
        core = body[: body.rindex(close)]
        pad = b'  <modifier name="SYNTH_PAD" userFlags="0"><far_clip>1 2</far_clip></modifier>\n'
        doc = core + pad * 2000 + close
        first, rest = doc[:65536], doc[65536:]
        self.assertEqual(len(first), 65536)
        # Premise (the reviewer's input): both streams read as "text", the first opens the
        # target root and does not close it, so completeness is only reached by joining.
        self.assertEqual(tp.gi.stream_kind(first, {}), "text")
        self.assertEqual(tp.gi.stream_kind(rest, {}), "text")
        self.assertEqual(tp.gi.xml_root(first), "timecycle_modifier_data")
        self.assertFalse(tp.gi.xml_complete("timecycle_modifier_data", first))
        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "doc.xml")
            with open(file_path, "wb") as f:
                f.write(doc)
            as_file = tp.load_documents([file_path])
            carve = os.path.join(tmp, "carve")
            os.makedirs(carve)
            self.write_carve(carve, [
                ("000000001000.bin", first),
                ("000000002000.bin", rest),
            ])
            as_carve = tp.load_documents([carve])
            self.assertEqual([r for r, _ in as_file], ["timecycle_modifier_data"])
            self.assertEqual([r for r, _ in as_carve], [r for r, _ in as_file])
            self.assertEqual(tp.summarize_documents(as_carve),
                             tp.summarize_documents(as_file))
            self.assertEqual(tp.summarize_documents(as_carve)["modifier_docs"], 1)

    def test_carve_dir_skips_a_binary_document_stream_and_still_measures_the_run(self):
        # A stream is "binary" once its first 4 KiB is under 95% printable, so 400 NUL bytes
        # trailing the closing root tag make a document stream binary: it passes the counting
        # test gi.xml_complete() but ET.fromstring rejects the NULs with "not well-formed",
        # and one such stream ended the whole run with exit 2, measuring no document at all.
        # main skipped binary streams; they must keep being skipped, leaving the run at 0.
        nul_doc = (b"<timecycle_modifier_data><modifier name='test' userFlags='0'/>"
                   b"</timecycle_modifier_data>") + b"\x00" * 400
        declless = SYNTH_MOD.split(b"\n", 1)[1]
        # Premise (the reviewer's input): the stream is binary, yet the counting test calls
        # it a closed document — it is the ET.fromstring call that rejects it.
        self.assertEqual(tp.gi.stream_kind(nul_doc, {}), "binary")
        self.assertTrue(tp.gi.xml_complete("timecycle_modifier_data", nul_doc))
        with tempfile.TemporaryDirectory() as tmp:
            self.write_carve(tmp, [
                ("000000001000.bin", SYNTH_MOD),   # declared document: measured
                ("000000002000.bin", nul_doc),     # binary stream: skipped
                ("000000003000.bin", declless),    # complete text document: measured
            ])
            out_path = os.path.join(tmp, "report.json")
            rc = tp.main([tmp, "--out", out_path])
            self.assertEqual(rc, 0)
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["modifier_docs"], 2)

    def test_cli_validation_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_xml = os.path.join(tmp, "mod.xml")
            with open(in_xml, "wb") as f:
                f.write(SYNTH_MOD)
            with self.assertRaises(SystemExit) as cm:
                tp.main([in_xml, "--out", in_xml])
            self.assertEqual(cm.exception.code, 2)
            with self.assertRaises(SystemExit) as cm:
                tp.main([in_xml, "--out", tmp])
            self.assertEqual(cm.exception.code, 2)
            with self.assertRaises(SystemExit) as cm:
                tp.main([os.path.join(tmp, "missing.rpf")])
            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

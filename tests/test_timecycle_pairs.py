"""Synthetic unit tests for tools/timecycle_pairs.py (no game files)."""
from __future__ import annotations

import contextlib
import io
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

    def test_non_finite_modifier_and_keyframe_numbers_are_refused(self):
        for token in (b"NaN", b"inf", b"-Infinity", b"1e400", b"-1e400"):
            cases = (
                ("timecycle_modifier_data", SYNTH_MOD.replace(b"0.5 0.0", token + b" 0.0")),
                ("timecycle_modifier_data", SYNTH_MOD.replace(b"0.5 0.0", b"0.5 " + token)),
                ("timecycle_keyframe_data", SYNTH_KF.replace(b"500 600 700", b"500 " + token + b" 700")),
            )
            for position, (kind, blob) in enumerate(cases):
                with self.subTest(token=token, kind=kind, position=position):
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        tp.summarize_documents([(kind, blob)])

    def test_cli_non_finite_input_preserves_report_and_emits_no_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "mod.xml")
            report = os.path.join(tmp, "report.json")
            with open(source, "wb") as f:
                f.write(SYNTH_MOD.replace(b"0.5 0.0", b"NaN 0.0"))
            with open(report, "wb") as f:
                f.write(b"previous report\n")
            for extra in ([], ["--out", report]):
                with self.subTest(extra=extra):
                    out, err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        status = tp.main([source] + extra)
                    self.assertEqual(status, 2)
                    self.assertEqual(out.getvalue(), "")
                    self.assertIn("non-finite", err.getvalue())
                    self.assertEqual(len(err.getvalue().splitlines()), 1)
                    with open(report, "rb") as f:
                        self.assertEqual(f.read(), b"previous report\n")

    def test_finite_scientific_notation_is_still_accepted(self):
        summary = tp.summarize_documents([
            ("timecycle_modifier_data", SYNTH_MOD.replace(b"0.5 0.0", b"-5e-1 1e-3")),
            ("timecycle_keyframe_data", SYNTH_KF.replace(b"500 600 700", b"5e2 6e2 7e2")),
        ])
        self.assertEqual(summary["first_number"]["min"], -0.5)
        self.assertEqual(summary["second_number"]["between_0_1"], 4)
        self.assertEqual(summary["vars_with_b_outside_0_1"]["far_clip"]["keyframe_range"], [500, 700])
        json.dumps(summary, allow_nan=False)


    def test_large_finite_pair_medians_remain_finite(self):
        for first, second, expected in (
                ("1e308", "1e308", 1e308),
                ("-1e308", "-1e308", -1e308),
                ("1e308", "1.6e308", 1.3e308),
                ("-1e308", "1e308", 0.0),
                ("5e-324", "5e-324", 5e-324)):
            with self.subTest(first=first, second=second), tempfile.TemporaryDirectory() as tmp:
                source = os.path.join(tmp, "in.xml")
                report = os.path.join(tmp, "report.json")
                xml = ('<?xml version="1.0"?><timecycle_modifier_data><modifier>'
                       f'<x>{first} {first}</x><y>{second} {second}</y>'
                       '</modifier></timecycle_modifier_data>')
                with open(source, "w", encoding="utf-8", newline="\n") as f:
                    f.write(xml)
                self.assertEqual(tp.main([source, "--out", report]), 0)
                with open(report, encoding="utf-8") as f:
                    result = json.load(f, parse_constant=lambda token: self.fail(token))
                for key in ("first_number", "second_number"):
                    self.assertEqual(result[key]["median"], round(expected, 6))

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

#!/usr/bin/env python3
"""Unit tests for tools/coverage_dashboard.py."""
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import coverage_dashboard


class TestCoverageDashboard(unittest.TestCase):

    def test_git_metadata(self):
        sha, date_str = coverage_dashboard.get_git_metadata(REPO_ROOT)
        self.assertIsInstance(sha, str)
        self.assertGreater(len(sha), 0)
        self.assertIsInstance(date_str, str)
        self.assertGreater(len(date_str), 0)

    def test_collectors_integrity(self):
        collectors = [
            coverage_dashboard.get_rdna2_coverage,
            coverage_dashboard.get_x64_coverage,
            coverage_dashboard.get_runtime_coverage,
            coverage_dashboard.get_tools_coverage,
        ]

        for col in collectors:
            data = col(REPO_ROOT)
            self.assertIn("title", data)
            self.assertIn("covered", data)
            self.assertIn("total", data)
            self.assertIn("categories", data)

            self.assertGreater(data["total"], 0, f"{data['title']} total must be > 0")
            self.assertGreaterEqual(data["covered"], 0, f"{data['title']} covered must be >= 0")
            self.assertLessEqual(data["covered"], data["total"], f"{data['title']} covered must be <= total")

            for name, cov, tot in data["categories"]:
                self.assertGreater(tot, 0, f"Category {name} total must be > 0")
                self.assertGreaterEqual(cov, 0, f"Category {name} covered must be >= 0")
                self.assertLessEqual(cov, tot, f"Category {name} covered must be <= total")

    def test_render_category_grid_xml_validity(self):
        # Render a snippet and wrap in valid XML
        snippet = coverage_dashboard.render_category_grid("VOP3", 10, 20, 10, 10, 100, 80)
        xml_wrap = f'<svg xmlns="http://www.w3.org/2000/svg">{snippet}</svg>'
        root = ET.fromstring(xml_wrap)
        self.assertIsNotNone(root)

    def test_render_card_xml_validity(self):
        panel = {
            "title": "Test Panel",
            "covered": 15,
            "total": 30,
            "skipped": 0,
            "categories": [("Cat A", 10, 20), ("Cat B", 5, 10)],
            "small_items": [("Small A", 2, 2)],
        }
        snippet = coverage_dashboard.render_card(panel, 20, 20, 940, 200)
        xml_wrap = f'<svg xmlns="http://www.w3.org/2000/svg">{snippet}</svg>'
        root = ET.fromstring(xml_wrap)
        self.assertIsNotNone(root)

    def test_render_dashboard_svg_well_formed(self):
        panels = [
            coverage_dashboard.get_rdna2_coverage(REPO_ROOT),
            coverage_dashboard.get_x64_coverage(REPO_ROOT),
            coverage_dashboard.get_runtime_coverage(REPO_ROOT),
            coverage_dashboard.get_tools_coverage(REPO_ROOT),
        ]
        svg_content = coverage_dashboard.render_dashboard_svg(panels, "abc1234", "2026-11-23 12:00 UTC")

        # Must parse cleanly without throwing
        root = ET.fromstring(svg_content)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        self.assertIn("viewBox", root.attrib)

        # Must contain title text
        self.assertIn("Leonida", svg_content)
        self.assertIn("abc1234", svg_content)

    def test_cli_output_and_check(self):
        tool_script = os.path.join(REPO_ROOT, "tools", "coverage_dashboard.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_svg = os.path.join(tmpdir, "dashboard.svg")

            # 1. Generate output
            r = subprocess.run([sys.executable, "-I", tool_script, "--output", out_svg], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"Command failed: {r.stderr}")
            self.assertTrue(os.path.exists(out_svg))

            # 2. Check mode passes on freshly generated file
            r = subprocess.run([sys.executable, "-I", tool_script, "--output", out_svg, "--check"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"Check mode failed: {r.stderr}")
            self.assertIn("is up to date", r.stdout)

            # 3. Check mode fails if file is corrupted / drifted
            with open(out_svg, "w", encoding="utf-8") as f:
                f.write("<svg>drifted content</svg>")
            r = subprocess.run([sys.executable, "-I", tool_script, "--output", out_svg, "--check"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, "Check mode should fail on drifted file")

            # 4. Check mode fails if file missing
            os.remove(out_svg)
            r = subprocess.run([sys.executable, "-I", tool_script, "--output", out_svg, "--check"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, "Check mode should fail if file is missing")

    def test_empty_files_produce_zero_change(self):
        """Replicate shdann's scenario: 0, 380, 381, 400 empty test files produce zero change."""
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            sem = Path(tmp) / "tests" / "semantic"
            sem.mkdir(parents=True)
            results = []
            for n in (0, 380, 381, 400):
                for i in range(n):
                    (sem / f"test_{i}.py").touch()
                cov = coverage_dashboard.get_rdna2_coverage(tmp)
                vop3_cov = cov["categories"][0][1]
                vop3_tot = cov["categories"][0][2]
                results.append((n, vop3_cov, vop3_tot, cov["covered"], cov["total"]))

            # Assert VOP3 and total counts are completely identical across all empty file counts
            baseline = results[0]
            for n, v_cov, v_tot, t_cov, t_tot in results:
                self.assertEqual(v_cov, baseline[1], f"VOP3 covered drifted at n={n}: {v_cov} vs {baseline[1]}")
                self.assertEqual(v_tot, baseline[2], f"VOP3 total drifted at n={n}: {v_tot} vs {baseline[2]}")
                self.assertEqual(t_cov, baseline[3], f"Total covered drifted at n={n}: {t_cov} vs {baseline[3]}")
                self.assertEqual(t_tot, baseline[4], f"Total forms drifted at n={n}: {t_tot} vs {baseline[4]}")

    def test_unrelated_code_and_comments_produce_zero_change(self):
        """Verify that Python comments, docstrings, or non-opcode variables produce zero change."""
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            sem = Path(tmp) / "tests" / "semantic"
            sem.mkdir(parents=True)
            base_cov = coverage_dashboard.get_rdna2_coverage(tmp)

            # File with comments, docstring, and arbitrary python
            (sem / "test_unrelated.py").write_text(
                '# Just a comment\n"""Module docstring with words."""\nx = 100\ndef helper():\n    return x\n',
                encoding="utf-8"
            )
            new_cov = coverage_dashboard.get_rdna2_coverage(tmp)
            self.assertEqual(base_cov["categories"], new_cov["categories"])
            self.assertEqual(base_cov["covered"], new_cov["covered"])

    def test_genuine_lowered_opcode_exercises_category(self):
        """Verify that genuine unexercised lowered RDNA2 opcodes increment their respective category."""
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            sem = Path(tmp) / "tests" / "semantic"
            sem.mkdir(parents=True)
            base_cov = coverage_dashboard.get_rdna2_coverage(tmp)

            trans_ops = coverage_dashboard.get_translator_lowered_opcodes()
            unexercised_vop3 = [
                op for op in trans_ops
                if coverage_dashboard.classify_rdna2_instruction(op) == "VOP3"
                and op not in coverage_dashboard._BASE_EXERCISED_OPCODES
            ]
            if unexercised_vop3:
                target_op = unexercised_vop3[0]
                (sem / "test_new_vop3.py").write_text(f'asm = ["{target_op} v0, v1, v2"]\n', encoding="utf-8")
                new_cov = coverage_dashboard.get_rdna2_coverage(tmp)
                self.assertEqual(new_cov["categories"][0][1], base_cov["categories"][0][1] + 1)


if __name__ == "__main__":
    unittest.main()

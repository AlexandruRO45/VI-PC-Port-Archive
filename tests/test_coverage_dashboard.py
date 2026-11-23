#!/usr/bin/env python3
"""Unit tests for tools/coverage_dashboard.py."""
import os
import re
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

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
        data = coverage_dashboard.get_rdna2_coverage(REPO_ROOT)
        self.assertEqual(data["title"], "gfx1030 (RDNA2) forms")
        self.assertIn("covered", data)
        self.assertEqual(data["total"], 1865, "Total catalog denominator must remain strictly fixed at 1,865")
        self.assertGreaterEqual(data["covered"], 0)
        self.assertLessEqual(data["covered"], data["total"])

        for name, cov, tot in data["categories"]:
            self.assertGreater(tot, 0, f"Category {name} total must be > 0")
            self.assertGreaterEqual(cov, 0, f"Category {name} covered must be >= 0")
            self.assertLessEqual(cov, tot, f"Category {name} covered ({cov}) must be <= total ({tot})")

    def test_render_category_grid_xml_validity(self):
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
        ]
        svg_content = coverage_dashboard.render_dashboard_svg(panels, "abc1234", "2026-11-23T12:00:00Z")

        root = ET.fromstring(svg_content)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        self.assertIn("viewBox", root.attrib)

        self.assertIn("Leonida", svg_content)
        self.assertIn("abc1234", svg_content)
        self.assertIn("Distinct translator mnemonics", svg_content)
        self.assertNotIn("Amber", svg_content)

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

    def test_cli_check_detects_category_shift_with_same_total(self):
        """Verify --check detects category count shifts even when the overall total is unchanged (tedin7 Issue 2)."""
        tool_script = os.path.join(REPO_ROOT, "tools", "coverage_dashboard.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_svg = os.path.join(tmpdir, "dashboard.svg")

            subprocess.run([sys.executable, "-I", tool_script, "--output", out_svg], check=True, capture_output=True, text=True)
            with open(out_svg, "r", encoding="utf-8") as f:
                content = f.read()

            # Mutate category counts in opposite directions while keeping total identical
            mutated = re.sub(r"VOP2 \d+/47", "VOP2 46/47", content)
            mutated = re.sub(r"VOP3 \d+/425", "VOP3 110/425", mutated)
            self.assertNotEqual(content, mutated)

            with open(out_svg, "w", encoding="utf-8") as f:
                f.write(mutated)

            r = subprocess.run([sys.executable, "-I", tool_script, "--output", out_svg, "--check"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, "--check should fail when category counts shift with same total")
            self.assertIn("Check failed", r.stderr)

    def test_svg_subtitle_omits_unused_amber_legend(self):
        """Verify that 'Amber: held back by skipped tests.' does not appear in generated SVG (tedin7 note)."""
        tool_script = os.path.join(REPO_ROOT, "tools", "coverage_dashboard.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_svg = os.path.join(tmpdir, "dashboard.svg")
            subprocess.run([sys.executable, "-I", tool_script, "--output", out_svg], check=True, capture_output=True, text=True)
            with open(out_svg, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertNotIn("Amber", content)
            self.assertNotIn("held back by skipped tests", content)
            self.assertIn("Green: covered. Grey: not covered.", content)

    def test_rdna2_vop3_and_vop2_classification(self):
        """Verify table-driven classification for VOP3 multi-operand instructions and VOP2 (tedin7 Issue 1)."""
        classify = coverage_dashboard.classify_rdna2_instruction

        # VOP3 multi-operand instructions (never VOP2)
        self.assertEqual(classify("v_bfe_u32"), "VOP3")
        self.assertEqual(classify("v_med3_f32"), "VOP3")
        self.assertEqual(classify("v_min3_f32"), "VOP3")
        self.assertEqual(classify("v_sad_u8"), "VOP3")
        self.assertEqual(classify("v_mul_hi_u32"), "VOP3")

        # VOP2 instructions
        self.assertEqual(classify("v_cvt_pkrtz_f16_f32"), "VOP2")
        self.assertEqual(classify("v_add_f32"), "VOP2")

        # Architectural _e64 suffix is always VOP3
        self.assertEqual(classify("v_cmp_eq_f32_e64"), "VOP3")
        self.assertEqual(classify("v_add_f32_e64"), "VOP3")

        # Arbitrary unknown v_ mnemonics must not fall back to VOP2
        self.assertIsNone(classify("v_nonexistent_unknown_mnemonic"))

    def test_empty_tree_and_comment_only_files_score_zero(self):
        """Assert that empty tests/semantic/, empty files, and comment-only files all score 0."""
        with tempfile.TemporaryDirectory() as tmp:
            cov = coverage_dashboard.get_rdna2_coverage(tmp)
            self.assertEqual(cov["covered"], 0)
            self.assertEqual(cov["total"], 1865)

            sem = Path(tmp) / "tests" / "semantic"
            sem.mkdir(parents=True)
            for n in (0, 380, 381, 400):
                for i in range(n):
                    (sem / f"test_{i}.py").touch()
                cov = coverage_dashboard.get_rdna2_coverage(tmp)
                self.assertEqual(cov["covered"], 0, f"Expected 0 covered for n={n}, got {cov['covered']}")

            (sem / "test_comment.py").write_text("# ds_swizzle\n# v_add3_u32\n# s_mov_b32\n", encoding="utf-8")
            (sem / "test_asm_comment.asm").write_text("; ds_swizzle\n; v_add3_u32\n// s_mov_b32\n", encoding="utf-8")
            cov_comment = coverage_dashboard.get_rdna2_coverage(tmp)
            self.assertEqual(cov_comment["covered"], 0, f"Comments with opcodes scored {cov_comment['covered']}")

    def test_docstrings_score_zero(self):
        """Verify that module/function/class docstrings mentioning opcodes without test code score 0."""
        with tempfile.TemporaryDirectory() as tmp:
            sem = Path(tmp) / "tests" / "semantic"
            sem.mkdir(parents=True)

            (sem / "test_doc_mod.py").write_text('"""Mentions v_mul_hi_u32 without any test code."""\n', encoding="utf-8")
            self.assertEqual(coverage_dashboard.get_rdna2_coverage(tmp)["covered"], 0)

            (sem / "test_doc_s_mov.py").write_text('"""s_mov_b32"""\n', encoding="utf-8")
            self.assertEqual(coverage_dashboard.get_rdna2_coverage(tmp)["covered"], 0)

            doc_func = '''
class TestDoc(unittest.TestCase):
    """Class docstring mentioning ds_swizzle."""
    def test_doc(self):
        """Function docstring mentioning v_add3_u32."""
        pass
'''
            (sem / "test_doc_func.py").write_text(doc_func, encoding="utf-8")
            self.assertEqual(coverage_dashboard.get_rdna2_coverage(tmp)["covered"], 0)

    def test_fixed_denominator_invariant(self):
        """Verify that covered <= total always holds and denominator remains fixed at 1,865."""
        with tempfile.TemporaryDirectory() as tmp:
            sem = Path(tmp) / "tests" / "semantic"
            sem.mkdir(parents=True)

            asm_lines = [f'asm_{i} = ["v_add_f32 v0, v1, v2"]' for i in range(60)]
            (sem / "test_many_vop2.py").write_text("\n".join(asm_lines) + "\n", encoding="utf-8")

            cov = coverage_dashboard.get_rdna2_coverage(tmp)
            self.assertEqual(cov["total"], 1865, "Total catalog denominator must stay 1,865")
            self.assertLessEqual(cov["covered"], cov["total"])

            vop2_entry = [c for c in cov["categories"] if c[0] == "VOP2"][0]
            self.assertLessEqual(vop2_entry[1], vop2_entry[2], "VOP2 covered must be <= 47")
            self.assertEqual(vop2_entry[2], 47, "VOP2 catalog total must remain 47")

    def test_asm_fixture_support(self):
        """Verify scanning of .asm fixtures (such as bfi_known_mask.asm) with comments ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            fixtures_dir = Path(tmp) / "tests" / "semantic" / "fixtures"
            fixtures_dir.mkdir(parents=True)

            asm_content = """
; Zero, full, shifted contiguous SGPR and sparse literal masks.
; Expected MRT0 words: 3f123456 4055aa7e
\tv_mov_b32 v0, 0x4055aa7e
\tv_bfi_b32 v2, 0, v0, v1
\ts_mov_b32 s8, 0x00ff0000
\texp mrt0 v2, v2, v2, v2 done vm
\ts_endpgm
"""
            (fixtures_dir / "bfi_known_mask.asm").write_text(asm_content, encoding="utf-8")

            test_py = '''
from pathlib import Path
fixture = Path(__file__).parent / "fixtures" / "bfi_known_mask.asm"
text = fixture.read_bytes()
'''
            (Path(tmp) / "tests" / "semantic" / "test_bfi.py").write_text(test_py, encoding="utf-8")

            cov = coverage_dashboard.get_rdna2_coverage(tmp)
            self.assertGreaterEqual(cov["covered"], 4)

    def test_shallow_clone_after_svg_only_commits_is_idempotent(self):
        """Verify workflow has fetch-depth: 0 and shallow clone guards against SVG-only commits (nanyoudegege & tedin7 Issue 3)."""
        # 1. Verify workflow checkout configuration
        wf_path = os.path.join(REPO_ROOT, ".github", "workflows", "coverage-dashboard.yml")
        with open(wf_path, "r", encoding="utf-8") as f:
            wf_content = f.read()
        self.assertIn("fetch-depth: 0", wf_content, "Workflow must use fetch-depth: 0 to access full commit history")

        # 2. Simulate shallow clone after SVG-only commits
        with tempfile.TemporaryDirectory() as base_tmp:
            base_path = Path(base_tmp)
            subprocess.run(["git", "init"], cwd=base_tmp, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=base_tmp, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=base_tmp, check=True, capture_output=True)

            # Metric-input commit
            (base_path / "tools").mkdir(parents=True)
            (base_path / "docs" / "coverage").mkdir(parents=True)
            (base_path / "tools" / "rdna2spirv.py").write_text("# metric input\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=base_tmp, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "metric input"], cwd=base_tmp, check=True, capture_output=True)

            # Two SVG-only commits
            svg_file = base_path / "docs" / "coverage" / "dashboard.svg"
            svg_file.write_text("<svg>first</svg>", encoding="utf-8")
            subprocess.run(["git", "add", str(svg_file)], cwd=base_tmp, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "docs: update 1 [skip ci]"], cwd=base_tmp, check=True, capture_output=True)

            svg_file.write_text("<svg>second</svg>", encoding="utf-8")
            subprocess.run(["git", "add", str(svg_file)], cwd=base_tmp, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "docs: update 2 [skip ci]"], cwd=base_tmp, check=True, capture_output=True)

            with tempfile.TemporaryDirectory() as shallow_tmp:
                subprocess.run(["git", "clone", "--depth", "1", f"file://{base_tmp}", shallow_tmp],
                               check=True, capture_output=True)

                tool_script = os.path.join(REPO_ROOT, "tools", "coverage_dashboard.py")
                out_svg = Path(shallow_tmp) / "docs" / "coverage" / "dashboard.svg"

                r1 = subprocess.run([sys.executable, "-I", tool_script, "--repo-root", shallow_tmp, "--output", str(out_svg)],
                                    check=True, capture_output=True, text=True)
                svg_run1 = out_svg.read_bytes()

                r2 = subprocess.run([sys.executable, "-I", tool_script, "--repo-root", shallow_tmp, "--output", str(out_svg)],
                                    check=True, capture_output=True, text=True)
                svg_run2 = out_svg.read_bytes()

                self.assertEqual(svg_run1, svg_run2, "SVG output drifted between successive runs in shallow clone")
                tip_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=shallow_tmp,
                                         capture_output=True, text=True).stdout.strip()
                self.assertNotIn(tip_sha.encode("utf-8"), svg_run1, "Embedded SVG-only tip commit SHA in shallow clone")


if __name__ == "__main__":
    unittest.main()

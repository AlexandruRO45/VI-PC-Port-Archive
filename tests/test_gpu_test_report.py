"""Synthetic CPU tests for structured GPU test JSON reports and baseline-vs-PR comparison.

    python -I tests/test_gpu_test_report.py

Covers:
  - Outcome normalization in scripts/run_gpu_tests.py (including check.Skip with ok=False).
  - Default mode preserving terminal output, exit codes, and performing no extra metadata probes.
  - Opt-in --json output recording revision, GPU, driver, effective run settings, summary, and per-test outcomes.
  - Refusal of --json paths that collide with inputs or runner artifacts (via tools/report_paths.same_report_path).
  - scripts/compare_gpu_reports.py distinguishing pass->fail, pass->skip, missing, skip->fail, added-fail,
    fail->pass, skip->pass, fail->skip, added pass/skip, unchanged fails, duplicate warnings, and surfacing
    incompatible run settings, GPU, or driver mismatches.

Standard library only; no GPU, no Vulkan driver, no llvm-mc, and no game data.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_gpu_reports as cmp_gpu  # noqa: E402
import run_gpu_tests as rgt  # noqa: E402
import check  # noqa: E402  # runtime/test/shaders, added to sys.path by run_gpu_tests


FAKE_REVISION = {
    "commit": "dde82f27e994337094afc476ded0b1426f6e87e2",
    "short_commit": "dde82f27e994",
    "dirty": False,
}

FAKE_DEVICE_INFO = {
    "device": "Synthetic Radeon RX 7600 XT",
    "subgroup_size": 64,
    "min_subgroup_size": 32,
    "max_subgroup_size": 64,
    "subgroup_size_control": True,
    "compute_in_required_stages": True,
    "mesh_in_required_stages": True,
    "compute_in_supported_stages": True,
    "mesh_in_supported_stages": True,
    "native64_compute": True,
    "native64": True,
    "per_vertex_inputs": True,
    "barycentric_fallback": False,
    "max_push_constants_size": 256,
    "vendor_id": 0x1002,
    "device_id": 0x7480,
    "driver_id": 3,
    "driver_name": "radv",
    "driver_info": "Mesa 26.2.6",
    "driver_version": 109060102,
    "api_version": "1.3.296",
    "conformance_version": "1.3.8.2",
    "rdna1_amd_windows": False,
    "rdna2_amd_windows": False,
    "harness_device": "Synthetic Radeon RX 7600 XT",
    "harness_device_matches": True,
}


def make_report(
    tests,
    *,
    commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    dirty=False,
    gpu=None,
    driver=None,
    settings=None,
    exit_code=None,
    error=None,
    runner="scripts/run_gpu_tests.py",
):
    base_gpu = {
        "name": "Synthetic Radeon RX 7600 XT",
        "vendor_id": 0x1002,
        "device_id": 0x7480,
        "subgroup_size": 64,
        "min_subgroup_size": 32,
        "max_subgroup_size": 64,
        "native64": True,
        "native64_compute": True,
        "max_push_constants_size": 256,
    }
    if gpu:
        base_gpu.update(gpu)

    base_driver = {
        "id": 3,
        "name": "radv",
        "info": "Mesa 26.2.6",
        "version": 109060102,
        "api_version": "1.3.296",
        "conformance_version": "1.3.8.2",
    }
    if driver:
        base_driver.update(driver)

    base_settings = {
        "filter": "",
        "disasm_mode": "llvm-mc",
        "assembler": "spirv-as",
        "validate": True,
        "require_pass": False,
        "wave64_emulate": False,
        "bary_fallback": False,
        "f16_interp_fallback": False,
        "texel_offset_fallback": False,
        "mesh_compute": False,
        "pc_tail_ubo": False,
        "fma": "glsl",
        "vk_device_filter": "",
        "vk_validation": False,
        "gpu_test_timeout": 120.0,
    }
    if settings:
        base_settings.update(settings)

    passed = sum(1 for t in tests if t["status"] == "pass")
    failed = sum(1 for t in tests if t["status"] == "fail")
    skipped = sum(1 for t in tests if t["status"] == "skip")
    if exit_code is None:
        exit_code = 1 if (failed or error) else 0

    doc = {
        "schema_version": 1,
        "runner": runner,
        "revision": {
            "commit": commit,
            "short_commit": commit[:12] if commit else None,
            "dirty": dirty,
        },
        "gpu": base_gpu,
        "driver": base_driver,
        "settings": base_settings,
        "summary": {
            "total": len(tests),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "exit_code": exit_code,
        },
        "tests": tests,
    }
    if error is not None:
        doc["error"] = error
    return doc


def write_json(path: Path, data: dict) -> str:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return str(path)


class OutcomeNormalizationTests(unittest.TestCase):
    def test_normalize_result_pass_fail_and_skip_variants(self):
        self.assertEqual(
            rgt.normalize_result("cs_lds", True, "cs_lds: 64 lanes mirrored"),
            {"name": "cs_lds", "status": "pass", "detail": "cs_lds: 64 lanes mirrored"},
        )
        self.assertEqual(
            rgt.normalize_result("cs_lds", False, "exit 1: lane 3 differs"),
            {"name": "cs_lds", "status": "fail", "detail": "exit 1: lane 3 differs"},
        )
        self.assertEqual(
            rgt.normalize_result("cs_wavemask", None, "SKIP (no 64-wide compute subgroup on this device)"),
            {
                "name": "cs_wavemask",
                "status": "skip",
                "detail": "SKIP (no 64-wide compute subgroup on this device)",
                "skip_reason": "no 64-wide compute subgroup on this device",
            },
        )
        # cs_fmt_* rows store (name, not problem, problem) where problem is a check.Skip (truthy -> ok=False).
        skip_detail = check.Skip("SKIP (unsupported: m.spv declares Int64 and the device has no shaderInt64)")
        self.assertEqual(
            rgt.normalize_result("cs_fmt_1010102", False, skip_detail),
            {
                "name": "cs_fmt_1010102",
                "status": "skip",
                "detail": "SKIP (unsupported: m.spv declares Int64 and the device has no shaderInt64)",
                "skip_reason": "unsupported: m.spv declares Int64 and the device has no shaderInt64",
            },
        )
        self.assertEqual(
            rgt.normalize_result("ps_m15", None, "SKIP (run by d3d12_translated_draw_test)"),
            {
                "name": "ps_m15",
                "status": "skip",
                "detail": "SKIP (run by d3d12_translated_draw_test)",
                "skip_reason": "run by d3d12_translated_draw_test",
            },
        )
        self.assertEqual(
            rgt.normalize_result("custom_skip", None, ""),
            {"name": "custom_skip", "status": "skip", "detail": ""},
        )


class RunGpuTestsJsonReportTests(unittest.TestCase):
    def _run_synthetic(
        self,
        *,
        shader_filter: str,
        outcomes: dict[str, tuple[int, str]],
        json_out: str | None = None,
        require_pass: bool = False,
        assembler: str = "python",
        no_validate: bool = True,
        disasm_dir: str | None = None,
        env: dict[str, str] | None = None,
        translator_rc: int = 0,
        revision=FAKE_REVISION,
        device_info=FAKE_DEVICE_INFO,
        fmt_1010102_result=None,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            args = SimpleNamespace(
                filter=shader_filter,
                build=str(Path(tmp) / "build"),
                llvm_mc="unused",
                require_pass=require_pass,
                assembler=assembler,
                no_validate=no_validate,
                disasm_dir=disasm_dir,
                json_out=json_out,
            )

            def fake_run_cmd(cmd, env=None, gpu=True, cwd=None):
                if not gpu:
                    if translator_rc != 0:
                        return translator_rc, "translator failed synthetically"
                    out_dir = Path(cmd[cmd.index("--out") + 1])
                    in_dir = Path(cmd[3])
                    for listing in in_dir.glob("*.s"):
                        (out_dir / (listing.stem + ".spv")).touch()
                    return 0, "translated ok"
                spv_name = Path(cmd[1]).stem
                return outcomes.get(spv_name, (0, f"{spv_name}: ok"))

            stdout, stderr = io.StringIO(), io.StringIO()
            base_env = {
                "LEONIDA_W64_EMULATE": "",
                "LEONIDA_BARY_FALLBACK": "0",
                "LEONIDA_F16_INTERP_FALLBACK": "0",
                "LEONIDA_PC_TAIL_UBO": "",
                "LEONIDA_FMA": "",
                "LEONIDA_VK_DEVICE": "",
                "LEONIDA_VK_VALIDATION": "0",
                "LEONIDA_GPU_TEST_TIMEOUT": "120",
            }
            if env:
                base_env.update(env)

            with (
                patch.dict(rgt.os.environ, base_env, clear=False),
                patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                patch.object(rgt, "spirv_as"),
                patch.object(rgt, "run_cmd", side_effect=fake_run_cmd),
                patch.object(rgt, "fmt_1010102_test", return_value=fmt_1010102_result),
                patch.object(rgt, "probe_git_revision", return_value=revision) as rev_probe,
                patch.object(rgt, "probe_device_info", return_value=device_info) as dev_probe,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                rc = rgt.run_suite(args, str(work))
            return rc, stdout.getvalue(), stderr.getvalue(), rev_probe.call_count, dev_probe.call_count

    def test_default_mode_preserves_output_exit_code_and_skips_extra_probes(self):
        outcomes = {"cs_lds": (0, "device: Synthetic\ncs_lds: 64 lanes mirrored")}
        rc_default, out_default, err_default, rev_calls, dev_calls = self._run_synthetic(
            shader_filter="cs_lds.asm", outcomes=outcomes, json_out=None
        )
        self.assertEqual((rc_default, err_default), (0, ""))
        self.assertEqual((rev_calls, dev_calls), (0, 0))

        with tempfile.TemporaryDirectory() as report_dir:
            report_path = Path(report_dir) / "out.json"
            rc_json, out_json, err_json, rev_calls_j, dev_calls_j = self._run_synthetic(
                shader_filter="cs_lds.asm", outcomes=outcomes, json_out=str(report_path)
            )
            self.assertEqual(rc_json, rc_default)
            self.assertEqual(out_json, out_default)
            self.assertEqual(err_json, err_default)
            self.assertEqual((rev_calls_j, dev_calls_j), (1, 1))
            raw_bytes = report_path.read_bytes()
            self.assertNotIn(b"\r\n", raw_bytes)
            report = json.loads(raw_bytes.decode("utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["revision"], FAKE_REVISION)
            self.assertEqual(report["gpu"]["name"], "Synthetic Radeon RX 7600 XT")
            self.assertEqual(report["gpu"]["vendor_id"], 0x1002)
            self.assertEqual(report["driver"]["name"], "radv")
            self.assertEqual(report["driver"]["info"], "Mesa 26.2.6")
            self.assertEqual(report["summary"], {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "exit_code": 0})
            self.assertEqual(
                report["tests"],
                [{"name": "cs_lds", "status": "pass", "detail": "cs_lds: 64 lanes mirrored"}],
            )

    def test_records_effective_settings_and_mixed_outcomes(self):
        with tempfile.TemporaryDirectory() as report_dir:
            report_path = Path(report_dir) / "nested" / "report.json"
            rc, out, err, _, _ = self._run_synthetic(
                shader_filter="cs_fmt_1010102.asm",
                outcomes={},
                json_out=str(report_path),
                require_pass=True,
                assembler="python",
                no_validate=False,
                env={
                    "LEONIDA_W64_EMULATE": "1",
                    "LEONIDA_BARY_FALLBACK": "1",
                    "LEONIDA_F16_INTERP_FALLBACK": "1",
                    "LEONIDA_PC_TAIL_UBO": "1",
                    "LEONIDA_FMA": "exact",
                    "LEONIDA_VK_DEVICE": "7600",
                    "LEONIDA_VK_VALIDATION": "1",
                    "LEONIDA_GPU_TEST_TIMEOUT": "45",
                },
                fmt_1010102_result=check.Skip("SKIP (unsupported: missing Int64)"),
            )
            self.assertEqual(rc, 1)
            self.assertIn("--require-pass: no GPU tests passed", err)
            self.assertIn("SKIP  cs_fmt_1010102", out)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"], {"total": 1, "passed": 0, "failed": 0, "skipped": 1, "exit_code": 1})
            self.assertEqual(report["tests"][0]["status"], "skip")
            self.assertEqual(report["tests"][0]["skip_reason"], "unsupported: missing Int64")
            settings = report["settings"]
            self.assertEqual(settings["filter"], "cs_fmt_1010102.asm")
            self.assertEqual(settings["disasm_mode"], "llvm-mc")
            self.assertEqual(settings["assembler"], "python")
            self.assertTrue(settings["require_pass"])
            self.assertTrue(settings["wave64_emulate"])
            self.assertTrue(settings["bary_fallback"])
            self.assertTrue(settings["f16_interp_fallback"])
            self.assertTrue(settings["pc_tail_ubo"])
            self.assertEqual(settings["fma"], "exact")
            self.assertEqual(settings["vk_device_filter"], "7600")
            self.assertTrue(settings["vk_validation"])
            self.assertEqual(settings["gpu_test_timeout"], 45.0)

    def test_writes_report_on_failing_test_and_on_translator_failure(self):
        with tempfile.TemporaryDirectory() as report_dir:
            fail_path = Path(report_dir) / "fail.json"
            rc, out, _, _, _ = self._run_synthetic(
                shader_filter="cs_lds.asm",
                outcomes={"cs_lds": (1, "device: X\ncs_lds: lane 3 differs")},
                json_out=str(fail_path),
            )
            self.assertEqual(rc, 1)
            self.assertIn("FAIL  cs_lds", out)
            fail_report = json.loads(fail_path.read_text(encoding="utf-8"))
            self.assertEqual(fail_report["summary"], {"total": 1, "passed": 0, "failed": 1, "skipped": 0, "exit_code": 1})
            self.assertEqual(fail_report["tests"][0]["status"], "fail")
            self.assertIn("exit 1: cs_lds: lane 3 differs", fail_report["tests"][0]["detail"])

            trans_path = Path(report_dir) / "translator_fail.json"
            rc2, _, _, _, _ = self._run_synthetic(
                shader_filter="cs_lds.asm",
                outcomes={},
                json_out=str(trans_path),
                translator_rc=1,
            )
            self.assertEqual(rc2, 1)
            trans_report = json.loads(trans_path.read_text(encoding="utf-8"))
            self.assertEqual(trans_report["summary"]["exit_code"], 1)
            self.assertEqual(trans_report["tests"], [])
            self.assertIn("rdna2spirv.py failed", trans_report["error"])

    def test_device_probe_failure_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as report_dir:
            report_path = Path(report_dir) / "no_gpu_probe.json"
            rc, _, _, _, _ = self._run_synthetic(
                shader_filter="cs_lds.asm",
                outcomes={"cs_lds": (0, "cs_lds: ok")},
                json_out=str(report_path),
                revision={"commit": None, "short_commit": None, "dirty": None},
                device_info={"error": "RuntimeError: no Vulkan loader found"},
            )
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIsNone(report["gpu"]["name"])
            self.assertIsNone(report["driver"]["name"])
            self.assertEqual(report["device_probe_error"], "RuntimeError: no Vulkan loader found")

    def test_unverified_multi_device_vulkan_12_and_13_reports_unknown_gpu_and_driver(self):
        """F4 regression: if vk_subgroup_info picks a Vulkan 1.3 device while vk_device_pick.h selects a higher-ranked
        Vulkan 1.2 device, report GPU and driver as unknown instead of misattributing the wrong device."""
        sys.path.insert(0, str(ROOT / "tests"))
        import test_vk_subgroup_info as tvk
        import vk_subgroup_info as vk_info

        dgpu_12 = tvk.FakeDevice(
            name="Discrete Vulkan 1.2 GPU",
            api_version=vk_info.VK_API_VERSION_1_2,
            device_type=vk_info.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            dynamic_rendering=False,
            vendor_id=0x10DE,
            device_id=0x1234,
            driver_id=4,
            driver_name="NVIDIA",
            driver_info="550.0",
        )
        cpu_13 = tvk.FakeDevice(
            name="llvmpipe (LLVM 18.1)",
            api_version=vk_info.VK_API_VERSION_1_3,
            device_type=4,
            vendor_id=0x10005,
            device_id=0,
            driver_id=13,
            driver_name="llvmpipe",
            driver_info="Mesa 25.0",
        )
        vk = tvk.FakeVulkan(devices=(dgpu_12, cpu_13))
        with patch.dict(vk_info.os.environ, {"LEONIDA_VK_DEVICE": ""}), \
                patch.object(vk_info, "load_vulkan", return_value=vk):
            mismatched_info = vk_info.query(driver_info=True)
        self.assertFalse(mismatched_info["harness_device_matches"])

        with tempfile.TemporaryDirectory() as report_dir:
            report_path = Path(report_dir) / "unverified_gpu.json"
            rc, _, _, _, _ = self._run_synthetic(
                shader_filter="cs_lds.asm",
                outcomes={"cs_lds": (0, "cs_lds: ok")},
                json_out=str(report_path),
                device_info=mismatched_info,
            )
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIsNone(report["gpu"]["name"])
            self.assertIsNone(report["gpu"]["vendor_id"])
            self.assertIsNone(report["driver"]["name"])
            self.assertIsNone(report["driver"]["info"])
            self.assertIn("Discrete Vulkan 1.2 GPU", report["device_probe_error"])
            self.assertIn("llvmpipe (LLVM 18.1)", report["device_probe_error"])

        # Missing harness_device_matches must also fail closed to unknown.
        stripped = {k: v for k, v in FAKE_DEVICE_INFO.items() if k != "harness_device_matches"}
        gpu, driver, probe_err = rgt.format_device_metadata(stripped)
        self.assertIsNone(gpu["name"])
        self.assertIsNone(driver["name"])
        self.assertIn("unverified GPU metadata", probe_err)

    def test_refuses_json_output_overwriting_input_or_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm_dir = Path(tmp) / "disasm"
            disasm_dir.mkdir()
            listing = disasm_dir / "cs_lds.s"
            sentinel = "; synthetic\n\ts_endpgm\n"
            listing.write_text(sentinel, encoding="utf-8", newline="\n")

            # Colliding with a .s input in --disasm-dir must fail with exit 2 and leave the file untouched.
            rc, _, err, _, _ = self._run_synthetic(
                shader_filter="cs_lds.asm",
                outcomes={},
                disasm_dir=str(disasm_dir),
                json_out=str(listing),
            )
            self.assertEqual(rc, 2)
            self.assertIn("must not overwrite", err)
            self.assertEqual(listing.read_text(encoding="utf-8"), sentinel)

            # Passing an existing directory as --json must fail with exit 2.
            rc_dir, _, err_dir, _, _ = self._run_synthetic(
                shader_filter="cs_lds.asm",
                outcomes={},
                json_out=str(disasm_dir),
            )
            self.assertEqual(rc_dir, 2)
            self.assertIn("is a directory", err_dir)

    def test_probe_git_revision_handles_missing_git_cleanly(self):
        with patch.object(rgt.subprocess, "run", side_effect=OSError("git not found")):
            self.assertEqual(rgt.probe_git_revision(), {"commit": None, "short_commit": None, "dirty": None})

        def fake_git(cmd, **_kwargs):
            if "rev-parse" in cmd:
                return SimpleNamespace(returncode=0, stdout="0123456789abcdef0123456789abcdef01234567\n")
            return SimpleNamespace(returncode=0, stdout=" M scripts/run_gpu_tests.py\n")

        with patch.object(rgt.subprocess, "run", side_effect=fake_git):
            self.assertEqual(
                rgt.probe_git_revision(),
                {
                    "commit": "0123456789abcdef0123456789abcdef01234567",
                    "short_commit": "0123456789ab",
                    "dirty": True,
                },
            )


class CompareGpuReportsTests(unittest.TestCase):
    def test_identical_compatible_reports_exit_0(self):
        tests = [
            {"name": "cs_bounds", "status": "pass", "detail": "ok"},
            {"name": "cs_wavemask", "status": "skip", "detail": "SKIP (no 64-wide)", "skip_reason": "no 64-wide"},
        ]
        base = make_report(tests, commit="1111111111111111111111111111111111111111")
        cand = make_report(tests, commit="2222222222222222222222222222222222222222")
        res = cmp_gpu.compare_reports(base, cand)
        self.assertTrue(res["compatible"])
        self.assertFalse(res["has_regressions"])
        self.assertEqual(res["exit_code"], 0)
        self.assertEqual(res["counts"]["unchanged_pass"], 1)
        self.assertEqual(res["counts"]["unchanged_skip"], 1)

    def test_distinguishes_pass_to_fail_pass_to_skip_and_missing_tests(self):
        base = make_report([
            {"name": "t_pass_to_fail", "status": "pass", "detail": "4096/4096 exact"},
            {"name": "t_pass_to_skip", "status": "pass", "detail": "4096/4096 exact"},
            {"name": "t_missing_pass", "status": "pass", "detail": "ok"},
            {"name": "t_missing_skip", "status": "skip", "detail": "SKIP (old skip)"},
            {"name": "t_skip_to_fail", "status": "skip", "detail": "SKIP (no feature)"},
            {"name": "t_fail_to_pass", "status": "fail", "detail": "old bug"},
            {"name": "t_skip_to_pass", "status": "skip", "detail": "SKIP (old skip)"},
            {"name": "t_fail_to_skip", "status": "fail", "detail": "exit 77"},
            {"name": "t_still_fail", "status": "fail", "detail": "pre-existing"},
            {"name": "t_still_pass", "status": "pass", "detail": "ok"},
        ])
        cand = make_report([
            {"name": "t_pass_to_fail", "status": "fail", "detail": "12 pixels differ"},
            {"name": "t_pass_to_skip", "status": "skip", "detail": "SKIP (unsupported: feature)", "skip_reason": "unsupported: feature"},
            {"name": "t_skip_to_fail", "status": "fail", "detail": "exit 1: crash"},
            {"name": "t_fail_to_pass", "status": "pass", "detail": "fixed"},
            {"name": "t_skip_to_pass", "status": "pass", "detail": "now passes"},
            {"name": "t_fail_to_skip", "status": "skip", "detail": "SKIP (unsupported: no Int64)", "skip_reason": "unsupported: no Int64"},
            {"name": "t_still_fail", "status": "fail", "detail": "pre-existing"},
            {"name": "t_still_pass", "status": "pass", "detail": "ok"},
            {"name": "t_added_pass", "status": "pass", "detail": "new test ok"},
            {"name": "t_added_skip", "status": "skip", "detail": "SKIP (optional)"},
            {"name": "t_added_fail", "status": "fail", "detail": "new test broken"},
        ])
        res = cmp_gpu.compare_reports(base, cand)
        self.assertTrue(res["compatible"])
        self.assertTrue(res["has_regressions"])
        self.assertEqual(res["exit_code"], 1)

        self.assertEqual([r["name"] for r in res["regressions"]["pass_to_fail"]], ["t_pass_to_fail"])
        self.assertEqual([r["name"] for r in res["regressions"]["pass_to_skip"]], ["t_pass_to_skip"])
        self.assertEqual(res["regressions"]["pass_to_skip"][0]["candidate_skip_reason"], "unsupported: feature")
        self.assertEqual(
            [(r["name"], r["baseline_status"]) for r in res["regressions"]["missing"]],
            [("t_missing_pass", "pass"), ("t_missing_skip", "skip")],
        )
        self.assertEqual([r["name"] for r in res["regressions"]["skip_to_fail"]], ["t_skip_to_fail"])
        self.assertEqual([r["name"] for r in res["regressions"]["added_fail"]], ["t_added_fail"])

        self.assertEqual([r["name"] for r in res["changes"]["fail_to_pass"]], ["t_fail_to_pass"])
        self.assertEqual([r["name"] for r in res["changes"]["skip_to_pass"]], ["t_skip_to_pass"])
        self.assertEqual([r["name"] for r in res["changes"]["fail_to_skip"]], ["t_fail_to_skip"])
        self.assertEqual([r["name"] for r in res["changes"]["added_pass"]], ["t_added_pass"])
        self.assertEqual([r["name"] for r in res["changes"]["added_skip"]], ["t_added_skip"])
        self.assertEqual([r["name"] for r in res["unchanged_fail"]], ["t_still_fail"])

        text = cmp_gpu.format_comparison(res)
        self.assertIn("pass -> fail (1):", text)
        self.assertIn("pass -> skip (1):", text)
        self.assertIn("missing from candidate (2):", text)
        self.assertIn("skip -> fail (1):", text)
        self.assertIn("added fail (1):", text)
        self.assertIn("fail -> skip (1):", text)
        self.assertIn("1 pre-existing fail", text)

    def test_pass_to_skip_and_missing_each_fail_independently(self):
        base = make_report([{"name": "cs_lds", "status": "pass", "detail": "ok"}])
        cand_skip = make_report([{"name": "cs_lds", "status": "skip", "detail": "SKIP (regressed)"}])
        res_skip = cmp_gpu.compare_reports(base, cand_skip)
        self.assertEqual(res_skip["exit_code"], 1)
        self.assertEqual(res_skip["counts"]["pass_to_skip"], 1)
        self.assertEqual(res_skip["counts"]["pass_to_fail"], 0)
        self.assertEqual(res_skip["counts"]["missing"], 0)

        cand_missing = make_report([])
        res_missing = cmp_gpu.compare_reports(base, cand_missing)
        self.assertEqual(res_missing["exit_code"], 1)
        self.assertEqual(res_missing["counts"]["missing"], 1)
        self.assertEqual(res_missing["counts"]["pass_to_fail"], 0)
        self.assertEqual(res_missing["counts"]["pass_to_skip"], 0)

    def test_fail_to_skip_and_improvements_do_not_trigger_regression(self):
        base = make_report([
            {"name": "t_fixed", "status": "fail", "detail": "wrong"},
            {"name": "t_unskipped", "status": "skip", "detail": "SKIP (missing)"},
            {"name": "t_proper_skip", "status": "fail", "detail": "exit 77"},
        ])
        cand = make_report([
            {"name": "t_fixed", "status": "pass", "detail": "ok"},
            {"name": "t_unskipped", "status": "pass", "detail": "ok"},
            {"name": "t_proper_skip", "status": "skip", "detail": "SKIP (unsupported: no Int64)"},
            {"name": "t_new_ok", "status": "pass", "detail": "ok"},
        ])
        res = cmp_gpu.compare_reports(base, cand)
        self.assertFalse(res["has_regressions"])
        self.assertEqual(res["exit_code"], 0)
        self.assertEqual(res["counts"]["fail_to_pass"], 1)
        self.assertEqual(res["counts"]["skip_to_pass"], 1)
        self.assertEqual(res["counts"]["fail_to_skip"], 1)
        self.assertEqual(res["counts"]["added_pass"], 1)

    def test_surfaces_incompatible_run_settings_gpu_and_driver(self):
        tests = [{"name": "cs_lds", "status": "pass", "detail": "ok"}]
        base = make_report(tests)
        cand = make_report(
            tests,
            gpu={"name": "NVIDIA GeForce RTX 5070 Ti", "vendor_id": 0x10DE, "device_id": 0x2C05},
            driver={"id": 4, "name": "NVIDIA", "info": "621.24", "version": 999},
            settings={
                "filter": "cs_",
                "wave64_emulate": True,
                "bary_fallback": True,
                "fma": "khr",
                "validate": False,
            },
        )
        res = cmp_gpu.compare_reports(base, cand)
        self.assertFalse(res["compatible"])
        self.assertFalse(res["has_regressions"])
        self.assertEqual(res["exit_code"], 1)

        fields = {(m["scope"], m["field"]) for m in res["incompatibilities"]}
        for expected in (
            ("gpu", "name"),
            ("gpu", "vendor_id"),
            ("gpu", "device_id"),
            ("driver", "id"),
            ("driver", "name"),
            ("driver", "info"),
            ("driver", "version"),
            ("settings", "filter"),
            ("settings", "wave64_emulate"),
            ("settings", "bary_fallback"),
            ("settings", "fma"),
            ("settings", "validate"),
        ):
            self.assertIn(expected, fields)

        text = cmp_gpu.format_comparison(res)
        self.assertIn("INCOMPATIBLE RUN SETTINGS / ENVIRONMENT:", text)
        self.assertIn("settings.wave64_emulate: False -> True", text)
        self.assertIn("gpu.name: 'Synthetic Radeon RX 7600 XT' -> 'NVIDIA GeForce RTX 5070 Ti'", text)

        # With --allow-incompatible, incompatibilities are still surfaced, but exit code is 0 if no regressions.
        allowed = cmp_gpu.compare_reports(base, cand, allow_incompatible=True)
        self.assertFalse(allowed["compatible"])
        self.assertEqual(len(allowed["incompatibilities"]), len(res["incompatibilities"]))
        self.assertEqual(allowed["exit_code"], 0)

    def test_non_comparable_settings_do_not_flag_incompatibility(self):
        tests = [{"name": "cs_lds", "status": "pass", "detail": "ok"}]
        base = make_report(tests, settings={"require_pass": False, "gpu_test_timeout": 120.0})
        cand = make_report(tests, settings={"require_pass": True, "gpu_test_timeout": 300.0})
        res = cmp_gpu.compare_reports(base, cand)
        self.assertTrue(res["compatible"])
        self.assertEqual(res["incompatibilities"], [])
        self.assertEqual(res["exit_code"], 0)

    def test_duplicate_test_names_return_nonzero_and_surface_ambiguous_cases(self):
        # Basic duplicate keeping worst status (fail > pass)
        base = make_report([
            {"name": "cs_dup", "status": "pass", "detail": "first"},
            {"name": "cs_dup", "status": "pass", "detail": "second"},
        ])
        cand = make_report([
            {"name": "cs_dup", "status": "fail", "detail": "broken row"},
            {"name": "cs_dup", "status": "pass", "detail": "later pass must not mask fail"},
        ])
        res = cmp_gpu.compare_reports(base, cand)
        self.assertEqual(res["exit_code"], 1)
        self.assertTrue(res["ambiguous"])
        self.assertEqual(res["duplicates"], {"baseline": ["cs_dup"], "candidate": ["cs_dup"]})
        self.assertEqual(res["counts"]["pass_to_fail"], 1)
        self.assertEqual(len(res["warnings"]), 2)

        # F3: Masked second-row regression ([fail, pass] -> [fail, fail]) must still exit 1 even with --allow-incompatible.
        base_masked = make_report([
            {"name": "cs_dup", "status": "fail", "detail": "subcase 1 pre-existing fail"},
            {"name": "cs_dup", "status": "pass", "detail": "subcase 2 ok"},
        ])
        cand_masked = make_report([
            {"name": "cs_dup", "status": "fail", "detail": "subcase 1 pre-existing fail"},
            {"name": "cs_dup", "status": "fail", "detail": "subcase 2 newly broken"},
        ])
        res_masked = cmp_gpu.compare_reports(base_masked, cand_masked, allow_incompatible=True)
        self.assertTrue(res_masked["ambiguous"])
        self.assertEqual(res_masked["exit_code"], 1)
        self.assertIn("AMBIGUOUS COMPARISON (DUPLICATE TEST NAMES):", cmp_gpu.format_comparison(res_masked))

        # F3: Dropped duplicate ([pass, pass] -> [pass]) must also exit 1 even with --allow-incompatible.
        base_drop = make_report([
            {"name": "cs_fmt_d16_xyz", "status": "pass", "detail": "80/80 dumped words exact"},
            {"name": "cs_fmt_d16_xyz", "status": "pass", "detail": "128/128 dumped words exact"},
        ])
        cand_drop = make_report([
            {"name": "cs_fmt_d16_xyz", "status": "pass", "detail": "80/80 dumped words exact"},
        ])
        res_drop = cmp_gpu.compare_reports(base_drop, cand_drop, allow_incompatible=True)
        self.assertTrue(res_drop["ambiguous"])
        self.assertEqual(res_drop["duplicates"], {"baseline": ["cs_fmt_d16_xyz"], "candidate": []})
        self.assertEqual(res_drop["exit_code"], 1)

    def test_run_level_error_on_either_side_fails_and_cannot_be_overridden(self):
        # F1: baseline error must be surfaced and exit 1 even with --allow-incompatible.
        base_err = make_report([], error="rdna2spirv.py failed with exit code 1")
        cand_ok = make_report([{"name": "cs_lds", "status": "pass", "detail": "ok"}])
        res_base = cmp_gpu.compare_reports(base_err, cand_ok, allow_incompatible=True)
        self.assertTrue(res_base["incomplete"])
        self.assertEqual(res_base["exit_code"], 1)
        text_base = cmp_gpu.format_comparison(res_base)
        self.assertIn("BASELINE ERROR: rdna2spirv.py failed with exit code 1", text_base)
        self.assertIn("INCOMPLETE COMPARISON:", text_base)

        # Candidate error must also exit 1 even with --allow-incompatible.
        base_ok = make_report([{"name": "cs_lds", "status": "pass", "detail": "ok"}])
        cand_err = make_report([], error="rdna2spirv.py failed with exit code 1")
        res_cand = cmp_gpu.compare_reports(base_ok, cand_err, allow_incompatible=True)
        self.assertTrue(res_cand["incomplete"])
        self.assertEqual(res_cand["exit_code"], 1)
        self.assertIn("CANDIDATE ERROR: rdna2spirv.py failed with exit code 1", cmp_gpu.format_comparison(res_cand))

    def test_empty_reports_and_all_skipped_and_summary_mismatch(self):
        # F2: Empty reports (no tests on either or both sides) cannot support a clean comparison.
        empty_base = make_report([])
        empty_cand = make_report([])
        res_empty = cmp_gpu.compare_reports(empty_base, empty_cand, allow_incompatible=True)
        self.assertTrue(res_empty["incomplete"])
        self.assertEqual(res_empty["exit_code"], 1)
        self.assertIn("baseline report contains no test rows", cmp_gpu.format_comparison(res_empty))
        self.assertIn("candidate report contains no test rows", cmp_gpu.format_comparison(res_empty))

        nonempty = make_report([{"name": "cs_lds", "status": "pass", "detail": "ok"}])
        res_empty_base = cmp_gpu.compare_reports(empty_base, nonempty, allow_incompatible=True)
        self.assertTrue(res_empty_base["incomplete"])
        self.assertEqual(res_empty_base["exit_code"], 1)

        # F2: All-skipped comparison must explicitly state that no tests ran.
        skip_rows = [{"name": "cs_wavemask", "status": "skip", "detail": "SKIP (no 64-wide)", "skip_reason": "no 64-wide"}]
        res_skip = cmp_gpu.compare_reports(make_report(skip_rows), make_report(skip_rows))
        self.assertTrue(res_skip["all_skipped"])
        self.assertEqual(res_skip["exit_code"], 0)
        text_skip = cmp_gpu.format_comparison(res_skip)
        self.assertIn("all tests were skipped (no tests ran)", text_skip)
        self.assertIn("all skipped (no tests ran)", text_skip)

        # F2: Summary counts that disagree with test rows must be rejected by load_report / CLI (exit 2).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good_path = write_json(root / "base.json", nonempty)
            bad_doc = make_report([{"name": "cs_lds", "status": "pass", "detail": "ok"}])
            bad_doc["summary"] = {"total": 100, "passed": 95, "failed": 5, "skipped": 0, "exit_code": 1}
            bad_path = write_json(root / "truncated.json", bad_doc)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cmp_gpu.main([good_path, bad_path])
            self.assertEqual(rc, 2)
            self.assertIn("summary counts", err.getvalue())
            self.assertIn("disagree with test rows", err.getvalue())

    def test_cli_json_output_and_collision_and_schema_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = write_json(root / "baseline.json", make_report([{"name": "cs_lds", "status": "pass", "detail": "ok"}]))
            cand_path = write_json(root / "candidate.json", make_report([{"name": "cs_lds", "status": "fail", "detail": "bad"}]))
            diff_path = root / "diff.json"

            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cmp_gpu.main([base_path, cand_path, "--json", str(diff_path)])
            self.assertEqual(rc, 1)
            self.assertEqual(err.getvalue(), "")
            raw_diff = diff_path.read_bytes()
            self.assertNotIn(b"\r\n", raw_diff)
            diff = json.loads(raw_diff.decode("utf-8"))
            self.assertEqual(diff["counts"]["pass_to_fail"], 1)

            # Refuses overwriting baseline or candidate (including via relative dot segment).
            orig_base = Path(base_path).read_bytes()
            alias_base = str(root / "." / "baseline.json")
            err2 = io.StringIO()
            with contextlib.redirect_stderr(err2):
                rc2 = cmp_gpu.main([base_path, cand_path, "--json", alias_base])
            self.assertEqual(rc2, 2)
            self.assertIn("must not overwrite", err2.getvalue())
            self.assertEqual(Path(base_path).read_bytes(), orig_base)

            # Invalid schema exits 2 without traceback.
            bad_path = write_json(root / "bad.json", {"schema_version": 99, "tests": []})
            err3 = io.StringIO()
            with contextlib.redirect_stderr(err3):
                rc3 = cmp_gpu.main([bad_path, cand_path])
            self.assertEqual(rc3, 2)
            self.assertIn("unsupported schema_version", err3.getvalue())


class EndToEndSyntheticComparisonTest(unittest.TestCase):
    """Run scripts/run_gpu_tests.py --json twice synthetically and compare with scripts/compare_gpu_reports.py."""

    def test_baseline_vs_pr_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disasm_dir = root / "disasm"
            disasm_dir.mkdir()
            for name in ("cs_lds", "cs_exec64", "cs_bvh"):
                (disasm_dir / f"{name}.s").write_text("; synthetic\n\ts_endpgm\n", encoding="utf-8", newline="\n")

            def run_once(report_file: Path, commit: str, outcomes: dict[str, tuple[int, str]], extra_env=None):
                work = root / f"work_{commit[:6]}"
                work.mkdir()
                args = SimpleNamespace(
                    filter="",
                    build=str(root / "build"),
                    llvm_mc="unused",
                    require_pass=False,
                    assembler="python",
                    no_validate=True,
                    disasm_dir=str(disasm_dir),
                    json_out=str(report_file),
                )

                def fake_run(cmd, env=None, gpu=True, cwd=None):
                    if not gpu:
                        out_dir = Path(cmd[cmd.index("--out") + 1])
                        in_dir = Path(cmd[3])
                        for s_file in in_dir.glob("*.s"):
                            (out_dir / (s_file.stem + ".spv")).touch()
                        return 0, "ok"
                    name = Path(cmd[1]).stem
                    return outcomes[name]

                env = {
                    "LEONIDA_W64_EMULATE": "0",
                    "LEONIDA_BARY_FALLBACK": "0",
                    "LEONIDA_F16_INTERP_FALLBACK": "0",
                    "LEONIDA_PC_TAIL_UBO": "0",
                    "LEONIDA_FMA": "glsl",
                    "LEONIDA_VK_DEVICE": "",
                    "LEONIDA_VK_VALIDATION": "0",
                }
                if extra_env:
                    env.update(extra_env)

                with (
                    patch.dict(rgt.os.environ, env, clear=False),
                    patch.object(rgt, "spirv_as"),
                    patch.object(rgt, "run_cmd", side_effect=fake_run),
                    patch.object(
                        rgt,
                        "probe_git_revision",
                        return_value={"commit": commit, "short_commit": commit[:12], "dirty": False},
                    ),
                    patch.object(rgt, "probe_device_info", return_value=FAKE_DEVICE_INFO),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    return rgt.run_suite(args, str(work))

            baseline_json = root / "baseline.json"
            pr_json = root / "pr.json"

            rc_base = run_once(
                baseline_json,
                "1111111111111111111111111111111111111111",
                {
                    "cs_lds": (0, "cs_lds: 64 lanes mirrored"),
                    "cs_exec64": (0, "cs_exec64: ok"),
                    "cs_bvh": (0, "cs_bvh: ok"),
                },
            )
            self.assertEqual(rc_base, 0)

            # PR run: cs_lds regresses to FAIL, cs_exec64 regresses to SKIP (exit 77), cs_bvh stays PASS.
            rc_pr = run_once(
                pr_json,
                "2222222222222222222222222222222222222222",
                {
                    "cs_lds": (1, "cs_lds: mismatch at lane 7"),
                    "cs_exec64": (77, "unsupported: shaderInt64 missing"),
                    "cs_bvh": (0, "cs_bvh: ok"),
                },
            )
            self.assertEqual(rc_pr, 1)

            diff_json = root / "diff.json"
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc_cmp = cmp_gpu.main([str(baseline_json), str(pr_json), "--json", str(diff_json)])
            self.assertEqual(rc_cmp, 1)
            self.assertEqual(stderr.getvalue(), "")

            diff = json.loads(diff_json.read_text(encoding="utf-8"))
            self.assertTrue(diff["compatible"])
            self.assertEqual([r["name"] for r in diff["regressions"]["pass_to_fail"]], ["cs_lds"])
            self.assertEqual([r["name"] for r in diff["regressions"]["pass_to_skip"]], ["cs_exec64"])
            self.assertEqual(diff["counts"]["unchanged_pass"], 1)
            self.assertIn("pass -> fail (1):", stdout.getvalue())
            self.assertIn("pass -> skip (1):", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Synthetic failure-path checks for the VS/GS subgroup probes."""
import contextlib
import importlib.util
import io
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runtime" / "test" / "shaders" / "w64_graphics.py"
spec = importlib.util.spec_from_file_location("w64_graphics", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProbeFailureTests(unittest.TestCase):
    def run_fixture(self, name, failed_width, diagnostic):
        stage = name[:2]
        count = runner.VS_VERTICES if stage == "vs" else runner.VS_VERTICES // 3
        records = [runner.FIXTURES[name][1](i) for i in range(count)]

        def draw_heap(tools, spirv, fixture, size, tmp):
            if fixture == runner.PROBES[stage]:
                if size == failed_width:
                    return None, diagnostic
                return [(i % 64, 0, 0, 0) for i in range(count)], None
            return records, None

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, clear=True), \
                patch.object(runner, "_draw_heap", side_effect=draw_heap), \
                patch.dict(runner._applied, {}, clear=True):
            spirv = Path(folder, "spirv")
            spirv.mkdir()
            (spirv / f"{name}.spv").touch()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = runner.main([folder, name])
        return status, output.getvalue()

    def test_probe_dispatch_failure_is_not_a_pass(self):
        for name, width in (("vs_w64_a_skip", 64), ("gs_w64_a_ring", 32)):
            with self.subTest(fixture=name, width=width):
                status, output = self.run_fixture(name, width, "exit 1: synthetic dispatch failure")
                self.assertEqual(status, 1, output)
                self.assertIn(f"{width}-wide {name[:2]} subgroup probe", output)
                self.assertIn("synthetic dispatch failure", output)

    def test_missing_probe_heap_is_not_a_pass(self):
        status, output = self.run_fixture("vs_w64_a_skip", 64, "exit 0: no heap dump")
        self.assertEqual(status, 1, output)
        self.assertIn("no heap dump", output)

    def test_unsupported_probe_keeps_default_draw(self):
        status, output = self.run_fixture("vs_w64_a_skip", 64, "unsupported: no geometry stage")
        self.assertEqual(status, 0, output)
        self.assertIn("  ok", output)

    def test_successful_probe_without_width_change_keeps_default_draw(self):
        status, output = self.run_fixture("vs_w64_a_skip", None, "")
        self.assertEqual(status, 0, output)
        self.assertIn("  ok", output)


class ProbeReadbackTests(unittest.TestCase):
    def run_pair(self, fixture, width, response):
        """Inject one harness response, then let the same fixture run again."""
        stage = fixture[:2]
        probe = runner.PROBES[stage]
        count = runner.VS_VERTICES // (3 if stage == "gs" else 1)
        injected = False

        def harness(cmd, *, env, **kwargs):
            nonlocal injected
            dest = Path(env["VKH_HEAP_DUMP"])
            name, size, _ = dest.name.split(".")
            fault = None
            if name == probe and size == str(width) and not injected:
                injected = True
                fault = response
            if fault == "silent":
                return subprocess.CompletedProcess(cmd, 77, "", "")
            data = bytearray(65536)
            for i in range(count):
                if name == probe:
                    # Partial waves with only 42 or 16 live lanes.
                    lane = i % (42 if size == "64" else 16)
                    if fault == "lane" and i == count - 1:
                        lane = 99
                    row = (lane, lane & 31, i, 0)
                else:
                    row = runner.FIXTURES[name][1](i)
                struct.pack_into("<4I", data, 4 * (runner.VS_RECORDS + 4 * i), *row)
            if fault == "short":
                data = data[:1]
            elif fault == "long":
                data.append(0)
            dest.write_bytes(data)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, clear=True), \
                patch.object(runner.subprocess, "run", side_effect=harness), \
                patch.dict(runner._applied, clear=True):
            spirv = Path(folder, "spirv")
            spirv.mkdir()
            (spirv / f"{fixture}.spv").touch()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = runner.main([folder, fixture, fixture])
            applied = dict(runner._applied)
        self.assertTrue(injected)
        return status, output.getvalue().splitlines(), applied

    def test_bad_readback_fails_fixture_then_retries_probe(self):
        for fixture in ("vs_w64_a_skip", "gs_w64_a_ring"):
            for width in (32, 64):
                for fault, reason in (("lane", "lane ID 99"), ("short", "1 bytes of heap"),
                                      ("long", "65537 bytes of heap")):
                    with self.subTest(fixture=fixture, width=width, fault=fault):
                        status, (first, second), applied = self.run_pair(fixture, width, fault)
                        self.assertEqual(status, 1)
                        self.assertIn(f"{width}-wide {fixture[:2]} subgroup probe", first)
                        self.assertIn(reason, first)
                        self.assertTrue(second.endswith("  ok"), second)
                        self.assertTrue(applied[fixture[:2], 32])

    def test_silent_unsupported_probe_keeps_fallback(self):
        for fixture in ("vs_w64_a_skip", "gs_w64_a_ring"):
            with self.subTest(fixture=fixture):
                status, lines, applied = self.run_pair(fixture, 64, "silent")
                self.assertEqual(status, 0, lines)
                self.assertTrue(all(line.endswith("  ok") for line in lines))
                self.assertFalse(applied[fixture[:2], 64])


if __name__ == "__main__":
    unittest.main()

"""Audio steps of make_game.py's assets stage: resuming a stopped build, ffmpeg failures, and awc_opus.py's writes.

Standard library only; no game data: the tools are mocked or fed hand-made inputs.
"""
import ast
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import awc_opus  # noqa: E402
import make_game  # noqa: E402
import winbatch  # noqa: E402


class WriteFileTest(unittest.TestCase):
    def test_writes_and_leaves_no_temporary(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.ogg")
            awc_opus.write_file(p, b"data")
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"data")
            self.assertEqual(os.listdir(d), ["a.ogg"])

    def test_failed_write_leaves_neither_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.ogg")
            with mock.patch.object(awc_opus.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    awc_opus.write_file(p, b"data")
            self.assertEqual(os.listdir(d), [])

    def test_interleaved_writes_use_private_staging_files(self):
        # Two writers to the same destination interleave: the first has staged its data and is paused at
        # os.replace while the second stages and publishes. With a shared ".part" sibling the second writer
        # truncates the first's staged data, so the first replace publishes the second writer's bytes (or
        # fails once the sibling is gone). A private staging file lets both writers publish their own bytes,
        # the last replace winning.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.ogg")
            real_replace = awc_opus.os.replace
            first_entered = threading.Event()
            allow_first = threading.Event()
            errors = []

            def delayed_replace(src, dst):
                if not first_entered.is_set():
                    first_entered.set()
                    if not allow_first.wait(5):
                        raise AssertionError("first writer was never released")
                real_replace(src, dst)

            def first():
                try:
                    awc_opus.write_file(p, b"A" * 100)
                except BaseException as e:  # pragma: no cover - recorded for the assertion below
                    errors.append(e)

            with mock.patch.object(awc_opus.os, "replace", side_effect=delayed_replace):
                t = threading.Thread(target=first)
                t.start()
                self.assertTrue(first_entered.wait(5), "first writer never reached os.replace")
                awc_opus.write_file(p, b"B" * 100)
                allow_first.set()
                t.join(5)
            self.assertFalse(t.is_alive())
            self.assertEqual(errors, [])
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"A" * 100)
            self.assertEqual(os.listdir(d), ["a.ogg"])


class AwcMainExitTest(unittest.TestCase):
    def run_main(self, error):
        with tempfile.TemporaryDirectory() as d:
            bank = os.path.join(d, "x.awc")
            open(bank, "wb").close()
            out, err = io.StringIO(), io.StringIO()
            argv = ["awc_opus.py", bank, os.path.join(d, "out"), "--jobs", "1"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(awc_opus, "export_bank", side_effect=error), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                return awc_opus.main(), out.getvalue(), err.getvalue()

    def test_write_error_fails_the_run(self):
        code, out, err = self.run_main(OSError(28, "No space left on device"))
        self.assertEqual(code, 1)
        self.assertIn("FAILED x.awc", out)
        self.assertIn("ERROR", err)

    def test_unparseable_bank_is_still_skipped(self):
        code, out, _ = self.run_main(ValueError("not an AWC"))
        self.assertEqual(code, 0)
        self.assertIn("skip x.awc", out)


class AudioStageResumeTest(unittest.TestCase):
    """Which audio steps assets() runs for a given work folder. rpf9dump, awc_opus.py and ffmpeg are mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = self.tmp.name
        self.game = os.path.join(self.work, "game")
        os.makedirs(self.game)
        self.audio = os.path.join(self.work, "assets", "audio")
        self.ran = []
        self.ffmpeg_ok = True
        # Model/texture manifests and the audio bank manifest present: only the steps under test can run.
        for rel in make_game.MODEL_ARCHIVES:
            name = os.path.splitext(os.path.basename(rel))[0]
            for mode in ("models", "textures"):
                self.touch(self.work, "assets", name, mode, "manifest.json")
                with open(os.path.join(self.work, "assets", name, mode, make_game.DECODER_RECORD), "w") as f:
                    f.write(make_game.decoder_digest())
        self.touch(self.audio, "audio_manifest.json")

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, *parts):
        p = os.path.join(*parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
        return p

    def fake_run(self, cmd, *a, **k):
        if cmd[1] == "xma":
            self.ran.append("rpf9dump xma")
            self.touch(self.audio, "xma", "bank", "s1.xma.wav")
            self.touch(self.audio, "xma", "bank", "s2.xma.wav")

    def fake_py(self, tool, *args):
        self.ran.append(tool)
        if tool == "awc_opus.py":
            os.makedirs(os.path.join(self.audio, "ogg"), exist_ok=True)

    def fake_subprocess(self, cmd, *a, **k):
        self.ran.append("ffmpeg")
        if self.ffmpeg_ok:
            open(cmd[-1], "w").close()
            return subprocess.CompletedProcess(cmd, 0)
        open(cmd[-1], "w").close()        # a failed ffmpeg can still leave part of its output
        return subprocess.CompletedProcess(cmd, 1)

    def run_stage(self, ffmpeg="ffmpeg"):
        self.ran = []
        with mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "run", self.fake_run), \
                mock.patch.object(make_game, "py", self.fake_py), \
                mock.patch.object(make_game.shutil, "which", lambda n: ffmpeg), \
                mock.patch.object(make_game.subprocess, "run", self.fake_subprocess), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.assets(self.game, self.work, False, 2)
        return self.ran

    def wavs(self):
        d = os.path.join(self.audio, "xma", "bank")
        return sorted(os.listdir(d))

    def test_fresh_then_nothing(self):
        self.assertEqual(self.run_stage(), ["awc_opus.py", "rpf9dump xma", "ffmpeg", "ffmpeg"])
        self.assertEqual(self.wavs(), ["s1.wav", "s1.xma.wav", "s2.wav", "s2.xma.wav"])
        self.assertEqual(self.run_stage(), [])

    def test_ogg_folder_from_a_stopped_run_does_not_count_as_done(self):
        os.makedirs(os.path.join(self.audio, "ogg"))
        self.assertIn("awc_opus.py", self.run_stage())

    def test_xma_folder_from_a_stopped_run_does_not_count_as_done(self):
        os.makedirs(os.path.join(self.audio, "xma"))
        self.assertIn("rpf9dump xma", self.run_stage())

    def test_a_truncated_or_foreign_xma_marker_exports_again(self):
        self.run_stage()
        for text in ("", "rpf9dump xm", "something else finished\n"):
            with self.subTest(text=text):
                with open(os.path.join(self.audio, "xma.done"), "w", encoding="utf-8") as f:
                    f.write(text)
                self.assertIn("rpf9dump xma", self.run_stage())
                with open(os.path.join(self.audio, "xma.done"), encoding="utf-8") as f:
                    self.assertEqual(f.read().strip(), "rpf9dump xma finished")

    def test_failed_decode_leaves_no_wav_and_is_retried(self):
        self.ffmpeg_ok = False
        self.run_stage()
        self.assertEqual(self.wavs(), ["s1.xma.wav", "s2.xma.wav"])     # no .wav, no .wav.part
        self.ffmpeg_ok = True
        self.assertEqual(self.run_stage(), ["ffmpeg", "ffmpeg"])
        self.assertEqual(self.wavs(), ["s1.wav", "s1.xma.wav", "s2.wav", "s2.xma.wav"])

    def test_ffmpeg_installed_later_decodes_on_the_next_run(self):
        self.assertNotIn("ffmpeg", self.run_stage(ffmpeg=None))
        self.assertEqual(self.run_stage(), ["ffmpeg", "ffmpeg"])

    def write_ogg_marker(self, text):
        os.makedirs(os.path.join(self.audio, "ogg"), exist_ok=True)
        with open(os.path.join(self.audio, "ogg.done"), "w", encoding="utf-8") as f:
            f.write(text)

    def test_marker_matches_reads_only_the_marker_length(self):
        p = os.path.join(self.audio, "probe.done")
        cases = (("tool finished\n", True), ("tool finished", True), ("tool finish", False),
                 ("tool finished" + "x" * (1 << 20), False), ("tool finished\n" + "x" * (1 << 20), False),
                 ("tool finished\n\nforeign\n", False), ("tool finished\n\n", False), (" tool finished\n", False))
        for text, want in cases:
            with self.subTest(size=len(text), want=want):
                with open(p, "w", encoding="utf-8") as f:
                    f.write(text)
                real_open = open
                reads = []

                def tracking_open(*a, **k):
                    f = real_open(*a, **k)
                    real_read = f.read
                    f.read = lambda n=-1: reads.append(n) or real_read(n)
                    return f
                with mock.patch("builtins.open", tracking_open):
                    self.assertEqual(make_game.marker_matches(p, "tool"), want)
                self.assertTrue(reads and all(0 <= n <= len("tool finished") + 2 for n in reads), reads)
        self.assertFalse(make_game.marker_matches(os.path.join(self.audio, "missing.done"), "tool"))

    def test_matching_awc_opus_fingerprint_skips_the_ogg_step(self):
        self.run_stage()
        self.write_ogg_marker(make_game.ogg_marker() + " finished\n")
        self.assertNotIn("awc_opus.py", self.run_stage())

    def test_old_or_other_awc_opus_marker_exports_again(self):
        for text in ("awc_opus.py finished\n", f"awc_opus.py {'0' * 64} finished\n"):
            with self.subTest(text=text):
                self.run_stage()
                self.write_ogg_marker(text)
                for folder in ("ogg", "wav"):
                    self.touch(self.audio, folder, "bank", "stale.wav")
                self.assertEqual(self.run_stage(), ["awc_opus.py"])
                self.assertEqual(os.listdir(os.path.join(self.audio, "ogg")), [])
                self.assertFalse(os.path.exists(os.path.join(self.audio, "wav")))
                with open(os.path.join(self.audio, "ogg.done"), encoding="utf-8") as f:
                    self.assertEqual(f.read(), make_game.ogg_marker() + " finished\n")
                self.assertEqual(self.run_stage(), [])

    def test_a_failed_re_export_then_the_previous_exporter_exports_again(self):
        self.run_stage()
        self.write_ogg_marker(make_game.ogg_marker() + " finished\n")
        self.touch(self.audio, "ogg", "bank", "kept.ogg")

        def failing_py(tool, *args):
            raise SystemExit(1)
        with mock.patch.object(make_game, "ogg_marker", lambda: "awc_opus.py " + "1" * 64), \
                mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "py", failing_py), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(SystemExit):
                make_game.assets(self.game, self.work, False, 2)
        self.assertFalse(os.path.exists(os.path.join(self.audio, "ogg.done")))
        self.assertIn("awc_opus.py", self.run_stage())

    def test_a_linked_audio_folder_stops_before_anything_is_removed(self):
        self.run_stage()
        self.write_ogg_marker("awc_opus.py finished\n")
        previous = self.touch(self.audio, "ogg", "previous.ogg")
        target = os.path.join(self.work, "elsewhere")
        kept = self.touch(target, "kept.wav")
        link = os.path.join(self.audio, "wav")
        try:
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(target, link)
            else:
                os.symlink(target, link, target_is_directory=True)
        except (OSError, ImportError, AttributeError) as e:
            self.skipTest(f"cannot make a link here: {e}")
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            self.run_stage()
        self.assertIn("link or junction", err.getvalue())
        self.assertNotIn("awc_opus.py", self.ran)
        self.assertTrue(os.path.isfile(previous))
        self.assertTrue(os.path.isfile(kept))

    def test_editing_awc_opus_or_its_imports_exports_again(self):
        tools = os.path.join(self.work, "tools")
        os.makedirs(tools)
        for name in make_game.AWC_OPUS_SOURCES:
            shutil.copy(os.path.join(make_game.TOOLS, name), tools)
        with mock.patch.object(make_game, "TOOLS", tools):
            self.run_stage()
            for name in make_game.AWC_OPUS_SOURCES:
                with open(os.path.join(tools, name), "ab") as f:
                    f.write(b"\n")
                self.assertEqual(self.run_stage(), ["awc_opus.py"], name)
                self.assertEqual(self.run_stage(), [], name)

    def test_the_list_covers_every_local_module_awc_opus_imports(self):
        seen, todo = set(), ["awc_opus"]
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            with open(os.path.join(make_game.TOOLS, name + ".py"), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module.split(".")[0]]
                else:
                    continue
                todo += [m for m in mods if os.path.isfile(os.path.join(make_game.TOOLS, m + ".py"))]
        self.assertEqual({n + ".py" for n in seen}, set(make_game.AWC_OPUS_SOURCES))

    def test_failed_ogg_step_leaves_no_marker(self):
        def failing_py(tool, *args):
            raise SystemExit(1)
        with mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "py", failing_py), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(SystemExit):
                make_game.assets(self.game, self.work, False, 2)
        self.assertFalse(os.path.exists(os.path.join(self.audio, "ogg.done")))


# Stands in for ffmpeg: writes what it could read from its stdin into the output file.
STDIN_FFMPEG = """import sys
with open(sys.argv[-1], "w", encoding="utf-8") as f:
    f.write(repr(sys.stdin.buffer.read() if sys.stdin else b""))
"""


class DecodeXmaStdinTest(unittest.TestCase):
    """ffmpeg reads stdin for key commands. Given the build's stdin, it stops a build in the background of a terminal
    ("Stopped (tty output)") at the first XMA2 stream and swallows input piped to make_game.py."""

    def test_ffmpeg_gets_no_stdin(self):
        with tempfile.TemporaryDirectory() as d:
            xma = os.path.join(d, "xma")
            os.makedirs(os.path.join(xma, "bank"))
            open(os.path.join(xma, "bank", "s1.xma.wav"), "wb").close()
            script = os.path.join(d, "fake_ffmpeg.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(STDIN_FFMPEG)
            if os.name == "nt":
                ffmpeg = winbatch.write_python_cmd(os.path.join(d, "ffmpeg.cmd"), sys.executable, script)
            else:
                ffmpeg = os.path.join(d, "ffmpeg")
                with open(ffmpeg, "w", encoding="utf-8") as f:
                    f.write(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
                os.chmod(ffmpeg, 0o755)
            # A child process, so that decode_xma() runs with a stdin this test controls.
            code = "import sys; sys.path.insert(0, sys.argv[1]); import make_game; make_game.decode_xma(*sys.argv[2:])"
            p = subprocess.run([sys.executable, "-I", "-c", code, ROOT, ffmpeg, xma],
                               input="tests/test_next.py\n", capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            with open(os.path.join(xma, "bank", "s1.wav"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "b''")


if __name__ == "__main__":
    unittest.main()

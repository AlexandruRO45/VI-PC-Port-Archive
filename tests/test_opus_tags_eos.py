"""Whole-page truncation checks using invented Ogg packets only; no game data."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import awc_opus  # noqa: E402
import opus_tags  # noqa: E402


def stream(folder, count=200, serial=1):
    path = folder / "fixture.ogg"
    packet = b"\xf8" + b"\x55" * 40
    awc_opus.write_ogg_opus(path, [packet] * count, 1, serial)
    return path.read_bytes()


class TestEndOfStream(unittest.TestCase):
    def test_whole_page_truncations_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            complete = stream(Path(tmp))
        spans = list(opus_tags.pages(complete))
        self.assertGreater(len(spans), 3)
        for start, _ in spans[2:]:
            with self.subTest(cut=start):
                truncated = complete[:start]
                # Every remaining page is complete and has a valid checksum.
                self.assertEqual(list(opus_tags.checked_pages(truncated)),
                                 list(opus_tags.pages(truncated)))
                with self.assertRaisesRegex(ValueError, "end-of-stream"):
                    opus_tags.retag(truncated, opus_tags.tags_packet([("TITLE", "synthetic")]))

    def test_complete_streams_keep_every_other_page(self):
        for count in (0, 1, 200):
            with self.subTest(packets=count), tempfile.TemporaryDirectory() as tmp:
                complete = stream(Path(tmp), count)
                packet = opus_tags.tags_packet([("TITLE", "synthetic")])
                result = opus_tags.retag(complete, packet)
                old_pages = list(opus_tags.pages(complete))
                new_pages = list(opus_tags.checked_pages(result))
                self.assertEqual(len(old_pages), len(new_pages))
                self.assertEqual(result[:new_pages[1][0]], complete[:old_pages[1][0]])
                self.assertEqual(result[new_pages[2][0]:], complete[old_pages[2][0]:])
                self.assertEqual(opus_tags.retag(result, packet), result)

    def test_complete_chained_stream_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = stream(Path(tmp), 1, serial=1)
            second = stream(Path(tmp), 1, serial=2)
        result = opus_tags.retag(first + second, opus_tags.tags_packet([("TITLE", "synthetic")]))
        self.assertTrue(result.endswith(second))
        self.assertEqual(len(list(opus_tags.checked_pages(result))), 6)

    def test_earlier_eos_does_not_hide_a_truncated_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = stream(Path(tmp), 1, serial=1)
            second = stream(Path(tmp), 200, serial=2)
        final_start, _ = list(opus_tags.pages(second))[-1]
        with self.assertRaisesRegex(ValueError, "end-of-stream"):
            opus_tags.retag(first + second[:final_start], opus_tags.tags_packet([]))

    def test_cli_skips_incomplete_stream_and_continues(self):
        for dry_run in (False, True):
            for keep_tags_only in (False, True):
                with self.subTest(dry_run=dry_run, tags_only=keep_tags_only):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        complete = stream(root)
                        spans = list(opus_tags.pages(complete))
                        cut = spans[2 if keep_tags_only else -1][0]
                        truncated = complete[:cut]
                        bank = root / "opus" / "bank"
                        bank.mkdir(parents=True)
                        damaged = bank / "00000001.ogg"
                        good = bank / "00000002.ogg"
                        damaged.write_bytes(truncated)
                        good.write_bytes(complete)
                        os.utime(damaged, ns=(1_600_000_000_000_000_000,) * 2)
                        before_mtime = damaged.stat().st_mtime_ns
                        names = root / "names.csv"
                        names.write_text("bank,stream_id,name\nbank,1,synthetic_one\n"
                                         "bank,2,synthetic_two\n", encoding="utf-8", newline="\n")
                        args = [sys.executable, "-I", str(ROOT / "tools" / "opus_tags.py"),
                                str(root / "opus"), "--names", str(names)]
                        if dry_run:
                            args.append("--dry-run")
                        result = subprocess.run(args, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stderr, "")
                        counts = dict(line.rsplit(None, 1) for line in result.stdout.splitlines())
                        self.assertEqual(counts.get("tagged"), "1")
                        self.assertEqual(counts.get("ogg files"), "2")
                        skipped = {k: v for k, v in counts.items() if k.startswith("skipped:")}
                        self.assertEqual(len(skipped), 1, result.stdout)
                        reason, value = next(iter(skipped.items()))
                        self.assertIn("end-of-stream", reason)
                        self.assertEqual(value, "1")
                        self.assertEqual(damaged.read_bytes(), truncated)
                        self.assertEqual(damaged.stat().st_mtime_ns, before_mtime)
                        if dry_run:
                            self.assertEqual(good.read_bytes(), complete)
                        else:
                            self.assertNotEqual(good.read_bytes(), complete)


if __name__ == "__main__":
    unittest.main()

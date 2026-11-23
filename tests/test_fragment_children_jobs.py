"""fragment_children keeps its --jobs below the Windows pool limit.

    python -I tests/test_fragment_children_jobs.py

`tools/fragment_children.py` split the range over `--jobs` slices and opened `multiprocessing.Pool(len(work))` as
it was. Above 61 workers that pool does not return on Windows (the measurement in tools/pool_workers.py:4-7), and a
large archive reaches it through the CLI alone: `--jobs 62` on 496 KiB of zeros already splits into 62 slices.
Every other tool that takes --jobs for a process pool goes through `pool_workers.process_workers`; this one must
too. The archive here is a file of zeros written by the test, and the pool is replaced by one that records its size
and runs the work in this process, so the cap is checked on every platform.
"""
import multiprocessing
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

import fragment_children  # noqa: E402
import pool_workers  # noqa: E402

WINDOWS_LIMIT = 61
SLICES = 62 * 8192          # --jobs 62 splits this into 62 page-aligned slices, one worker each
SIZES = []                  # the size each recording pool was created with


class RecordingPool:
    """Stands in for multiprocessing.Pool: notes its size and runs each job at once, in this process."""

    def __init__(self, processes=None):
        SIZES.append(processes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, jobs, chunksize=1):
        return [fn(job) for job in jobs]


class PoolLimitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = os.path.join(self.tmp.name, "zeros.rpf")
        with open(self.archive, "wb") as f:
            f.write(bytes(SLICES))
        SIZES.clear()
        patcher = mock.patch.object(pool_workers, "MAX_PROCESS_WORKERS", WINDOWS_LIMIT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_jobs_above_the_limit_still_scans(self):
        one = fragment_children.scan_archive(self.archive, jobs=1)
        self.assertEqual((one["fragments"], one["unreadable"]), (0, 0))   # nothing to find in zeros
        self.assertEqual(SIZES, [])                                       # one worker opens no pool
        with mock.patch.object(multiprocessing, "Pool", RecordingPool):
            self.assertEqual(fragment_children.scan_archive(self.archive, jobs=62), one)
            self.assertLessEqual(SIZES.pop(), WINDOWS_LIMIT)
            self.assertEqual(fragment_children.scan_archive(self.archive, jobs=200), one)
            self.assertLessEqual(SIZES.pop(), WINDOWS_LIMIT)


if __name__ == "__main__":
    unittest.main()

"""`support.translate_listing` reuses one scratch directory; a translation must never see another listing's text.

Two ways the reuse could hand back the wrong shader, both pinned here:

* the same listing name from two threads: the file was written by one call and translated after the other
  call overwrote it (the write and the translation are now one locked step);
* two names that are one file on a case-insensitive volume (`ps_case` and `ps_CASE`): the written-bytes
  cache had two keys for one file, so the third call skipped the write (the file name, and so the key, goes through `os.path.normcase`).
  The Linux run emulates the volume: `normcase` folds case and the translator opens the existing file whose
  name matches ignoring case, as NTFS does.

Standard library only, no game data, no GPU.
"""
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support as s


def _value(name, value):
    lines = [f"v_mov_b32_e32 v0, {value}", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return s.mrt0_words(s.translate_lines(name, lines))[0]


def _open_ignoring_case(real):
    def translate(path):
        d, base = os.path.split(path)
        for entry in os.listdir(d):
            if entry.lower() == base.lower():
                return real(os.path.join(d, entry))
        return real(path)
    return translate


class ScratchCache(unittest.TestCase):
    def test_names_differing_only_in_case(self):
        real = s.rdna2spirv.translate
        with patch.object(os.path, "normcase", lambda p: p.lower()), \
                patch.object(s.rdna2spirv, "translate", _open_ignoring_case(real)):
            got = [_value("ps_case", 1), _value("ps_CASE", 2), _value("ps_case", 1)]
        self.assertEqual(got, [1, 2, 1])

    def test_same_name_from_two_threads(self):
        first_ready, release = threading.Event(), threading.Event()
        real = s.rdna2spirv.translate

        def ordered(path):
            if threading.current_thread().name.startswith("worker"):
                first_ready.set()
                release.wait(2)         # held open: the other call runs meanwhile if nothing serialises them
            return real(path)

        with patch.object(s.rdna2spirv, "translate", ordered), ThreadPoolExecutor(1, thread_name_prefix="worker") as pool:
            a = pool.submit(_value, "ps_shared", 1)
            self.assertTrue(first_ready.wait(10))
            b = threading.Thread(target=lambda: results.append(_value("ps_shared", 2)))
            results = []
            b.start()
            b.join(0.5)                 # serialised: still waiting for the lock; unserialised: already done
            release.set()
            b.join(10)
            self.assertEqual((a.result(10), results), (1, [2]))


if __name__ == "__main__":
    unittest.main()

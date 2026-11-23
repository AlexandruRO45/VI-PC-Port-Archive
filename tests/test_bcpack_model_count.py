"""Tests for tools/bcpack_model_count.py with hand-built BCPack streams (no game data).

The streams come from the synthetic encoder in tests/test_bcpack.py. The Rust decoder is replaced by a stand-in that
follows the DOCUMENTED behaviour of third_party/bcpack-rs (the strict model with no virtual zero bytes); it checks the
classification and counting logic, not the real library, which needs the Rust toolchain.

    python -I tests/test_bcpack_model_count.py
"""
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import bcpack  # noqa: E402
import bcpack_model_count as bmc  # noqa: E402
import test_bcpack as tb  # noqa: E402

CODEC = 1                 # BC1: 12 lanes, widths 8 8 5 6 5 5 6 5 8 8 8 8
WIDTHS = bcpack.LANE_WIDTHS[CODEC]
NSYM = 6                  # lane elements per chunk (8 bytes each)
SIZE = NSYM * bcpack.ELEMENT_BYTES


def random_lanes(rng):
    return [[rng.randrange(1 << w) for _ in range(NSYM)] for w in WIDTHS]


def build(lanes, strip):
    """A whole chunk from per-lane symbols; ``strip`` omits trailing zero bytes of each flushed fragment (up to 8)."""
    streams = [tb._encode_fragment(syms, w, w, 0, flush=True, strip_zeros=strip)[0] for syms, w in zip(lanes, WIDTHS)]
    return tb.interleave_lanes(streams, WIDTHS, NSYM)


def find_stripped_chunk():
    """(lanes, unstripped chunk, stripped chunk): a q = 0 stream whose final zero byte is omitted."""
    rng = random.Random(519)
    for _ in range(20000):
        lanes = random_lanes(rng)
        plain = build(lanes, False)
        stripped = build(lanes, True)
        if len(stripped) < len(plain):
            return lanes, plain, stripped
    raise AssertionError("no stream with an omitted trailing zero found")


class StandInRust:
    """Documented behaviour of third_party/bcpack-rs: the strict model without virtual zero bytes."""

    def decode(self, raw, codec, size, want_symbols=True):
        status, lanes, used, _ = bmc.python_decode(raw, codec, size, virtual_zeros=False)
        if status != "ok":
            return "error", None, -1
        return ("full" if used == len(raw) else "short"), lanes, used


class FixedRust:
    def __init__(self, status, lanes=None):
        self.status, self.lanes = status, lanes

    def decode(self, raw, codec, size, want_symbols=True):
        return self.status, self.lanes, len(raw)


class Classification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lanes, cls.plain, cls.stripped = find_stripped_chunk()

    def tally(self, raw, rs, **kw):
        counts = bmc.Counter()
        result = bmc.classify(raw, CODEC, SIZE, rs, **kw)
        bmc.tally(counts, result)
        return counts, result

    def test_the_strict_model_reads_the_synthetic_chunks_back(self):
        for raw in (self.plain, self.stripped):
            status, lanes, used, counter = bmc.python_decode(raw, CODEC, SIZE)
            self.assertEqual((status, lanes, used), ("ok", self.lanes, len(raw)))
            self.assertEqual(counter.fragments, len(WIDTHS))        # one fragment per lane

    def test_accepted_by_both_without_virtual_zero_bytes(self):
        counts, result = self.tally(self.plain, StandInRust())
        self.assertEqual(counts["accepted_by_both"], 1)
        self.assertEqual(counts["python_accepts"], 1)
        self.assertEqual(counts["fragments"], len(WIDTHS))
        self.assertEqual(result["agree"], True)
        self.assertEqual(counts["symbol_disagreements"], 0)
        self.assertNotIn("accepted_only_thanks_to_virtual_zero_bytes", counts)
        self.assertEqual(result["vz_needed"], False)

    def test_omitted_trailing_zero_is_accepted_only_by_python_thanks_to_virtual_zero_bytes(self):
        counts, result = self.tally(self.stripped, StandInRust())
        self.assertEqual(counts["only_python"], 1)
        self.assertEqual(counts["only_python_rust_error"], 1)
        self.assertEqual(counts["accepted_only_thanks_to_virtual_zero_bytes"], 1)
        self.assertEqual(counts["only_python_and_needs_virtual_zero_bytes"], 1)
        self.assertNotIn("accepted_by_both", counts)
        self.assertEqual(bmc.python_decode(self.stripped, CODEC, SIZE, virtual_zeros=False)[:2], ("error", "fragment not at home state"))

    def test_virtual_zero_bytes_setting_is_restored(self):
        before = bcpack.MAX_VIRTUAL_ZEROS
        bmc.python_decode(self.stripped, CODEC, SIZE, virtual_zeros=False)
        self.assertEqual(bcpack.MAX_VIRTUAL_ZEROS, before)
        bmc.python_decode(b"", CODEC, SIZE)
        self.assertEqual(bcpack.MAX_VIRTUAL_ZEROS, before)

    def test_accepted_only_by_rust_and_by_neither(self):
        truncated = self.plain[:len(self.plain) // 2]
        counts, _ = self.tally(truncated, FixedRust("full", self.lanes))
        self.assertEqual(counts["only_rust"], 1)
        self.assertEqual(counts["python_error[input ran out]"] + counts["python_error[fragment not at home state]"], 1)
        counts, _ = self.tally(truncated, StandInRust())
        self.assertEqual(counts["neither"], 1)
        self.assertEqual(sum(v for k, v in counts.items() if k.startswith("neither[python: ") and "rust: error]" in k), 1)

    def test_rust_short_consumption_counts_as_only_python(self):
        counts, _ = self.tally(self.plain, FixedRust("short"))
        self.assertEqual(counts["only_python"], 1)
        self.assertEqual(counts["only_python_rust_short"], 1)
        self.assertEqual(counts["rust_short"], 1)

    def test_symbol_disagreement_where_both_accept(self):
        wrong = [list(lane) for lane in self.lanes]
        wrong[3][2] ^= 1
        counts, result = self.tally(self.plain, FixedRust("full", wrong))
        self.assertEqual(counts["accepted_by_both"], 1)
        self.assertEqual(counts["symbol_disagreements"], 1)
        self.assertEqual(result["agree"], False)

    def test_python_only_and_python_stride(self):
        counts, result = self.tally(self.stripped, None)
        self.assertEqual((counts["python_accepts"], counts["accepted_only_thanks_to_virtual_zero_bytes"]), (1, 1))
        self.assertNotIn("accepted_by_both", counts)
        counts, result = self.tally(self.plain, StandInRust(), run_python=False)       # Rust only: the stride skipped Python
        self.assertEqual((counts["rust_full"], counts["python_run"], counts["chunks_classified"]), (1, 0, 1))
        counts, result = self.tally(self.plain, None, vz_check=False)
        self.assertIsNone(result["vz_needed"])

    def test_width_9_fragments_are_counted_by_width_and_q(self):
        with bmc.ModelCounter() as counter:
            bcpack._get_model(9, 8)
            bcpack._get_model(9, 3)
            bcpack._get_model(8, 8)
        self.assertEqual((counter.fragments, counter.width9, counter.width9_divergent), (3, 2, 1))
        for q in bmc.DIVERGENT_W9_Q:
            with bmc.ModelCounter() as counter:
                bcpack._get_model(9, q)
            self.assertEqual(counter.width9_divergent, 1)
        self.assertEqual(max(max(w) for w in bcpack.LANE_WIDTHS.values()), 8)           # no codec lane is width 9 by default


@unittest.skipUnless(os.path.isfile(bmc.default_rs_library()), "the bcpack_calibration library is not built")
class RealRustLibrary(unittest.TestCase):
    """The compiled third_party/bcpack-rs decoder on the synthetic chunks: the finding behind ASSET-519."""

    def test_rust_refuses_an_omitted_trailing_zero_that_python_accepts(self):
        lanes, plain, stripped = find_stripped_chunk()
        rs = bmc.RsDecoder(bmc.default_rs_library())
        both = bmc.classify(plain, CODEC, SIZE, rs)
        self.assertEqual((both["rs"], both["py"], both["agree"]), ("full", "ok", True))
        only_python = bmc.classify(stripped, CODEC, SIZE, rs)
        self.assertEqual((only_python["rs"], only_python["py"], only_python["vz_needed"]), ("error", "ok", True))


class FakePool:
    """Stands in for ttpf_mips.Pool: chunks are (virt, size, start, packed, codec, pad) over one buffer."""
    instances = {}

    def __init__(self, path):
        if path not in self.instances:
            raise ValueError("not an RDS5 container")
        self.chunks, self.data = self.instances[path]

    def close(self):
        pass


class ScanBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lanes, cls.plain, cls.stripped = find_stripped_chunk()

    def pool(self, *pieces):
        data, chunks, at = b"", [], 0
        for raw, codec in pieces:
            chunks.append((0, SIZE, at, len(raw), codec, 0))
            data += raw
            at += len(raw)
        return chunks, data

    def run_scan(self, pools, **kw):
        FakePool.instances = pools
        with mock.patch("ttpf_mips.find_pools", return_value=list(pools) + ["unreadable"]), mock.patch("ttpf_mips.Pool", FakePool):
            return bmc.scan("unused", StandInRust(), **kw)


class Scan(ScanBase):
    def test_scan_counts_chunks_by_class_and_never_names_a_pool(self):
        pools = {"p1": self.pool((self.plain, CODEC), (self.stripped, CODEC), (b"\0" * SIZE, 0)),
                 "p2": self.pool((self.plain[:10], CODEC), (self.plain, 8))}
        counts, seconds = self.run_scan(pools)
        self.assertEqual((counts["pools_found"], counts["pools_read"], counts["pool_not_opened[ValueError]"]), (3, 2, 1))
        self.assertEqual((counts["chunks_total"], counts["chunks_not_bcpack"], counts["chunks_bcpack"]), (5, 2, 3))
        self.assertEqual((counts["accepted_by_both"], counts["only_python"], counts["neither"]), (1, 1, 1))
        self.assertEqual(counts["accepted_only_thanks_to_virtual_zero_bytes"], 1)
        text = bmc.render(counts, seconds, "stand-in")
        for name in ("p1", "p2", "unreadable", "unused"):
            self.assertNotIn(name, text.replace("python", "").replace("pools", ""))

    def test_stride_and_limit(self):
        pools = {"p1": self.pool(*[(self.plain, CODEC)] * 6)}
        counts, _ = self.run_scan(pools, stride=3)
        self.assertEqual((counts["chunks_classified"], counts["python_run"], counts["rust_full"]), (6, 2, 6))
        counts, _ = self.run_scan(pools, limit=2)
        self.assertEqual((counts["chunks_bcpack"], counts["chunks_classified"]), (6, 2))


class ScanPolicy(ScanBase):
    def test_python_runs_on_every_rust_refusal_and_on_a_stride_of_the_rest(self):
        order = [self.plain, self.stripped, self.plain, self.plain, self.stripped, self.plain, self.plain]
        pools = {"p1": self.pool(*[(raw, CODEC) for raw in order])}
        counts, _ = self.run_scan(pools, stride=4)
        self.assertEqual((counts["rust_full"], counts["rust_error"]), (5, 2))
        self.assertEqual(counts["only_python"], 2)                       # both refusals were checked by Python
        self.assertEqual(counts["python_run"], 3)                        # the 2 refusals and chunk 0 of the stride
        self.assertEqual(counts["accepted_by_both"], 1)
        counts, _ = self.run_scan(pools, run_python=False)               # --rust-only
        self.assertEqual((counts["python_run"], counts["rust_full"], counts["rust_error"]), (0, 5, 2))
        self.assertNotIn("only_python", counts)

    def test_python_only_honours_the_stride(self):
        pools = {"p1": self.pool(*[(self.plain, CODEC)] * 6)}
        FakePool.instances = pools
        with mock.patch("ttpf_mips.find_pools", return_value=list(pools)), mock.patch("ttpf_mips.Pool", FakePool):
            counts, _ = bmc.scan("unused", None, stride=2)
        self.assertEqual((counts["python_run"], counts["chunks_classified"], counts["python_accepts"]), (3, 6, 3))

    def test_both_accept_although_python_shifts_virtual_zeros_is_counted_apart(self):
        counts = bmc.Counter()
        bmc.tally(counts, {"rs": "full", "py": "ok", "fragments": 1, "w9": 0, "w9_divergent": 0, "vz_needed": True, "agree": True})
        bmc.tally(counts, {"rs": "error", "py": "ok", "fragments": 1, "w9": 0, "w9_divergent": 0, "vz_needed": True, "agree": None})
        self.assertEqual(counts["accepted_only_thanks_to_virtual_zero_bytes"], 2)
        self.assertEqual(counts["accepted_by_both_although_python_shifts_virtual_zeros"], 1)
        self.assertEqual(counts["only_python_and_needs_virtual_zero_bytes"], 1)


class Cli(unittest.TestCase):
    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = bmc.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_missing_rust_library_is_exit_77_and_python_only_does_not_need_it(self):
        with tempfile.TemporaryDirectory() as game, tempfile.TemporaryDirectory() as work:
            os.makedirs(os.path.join(game, "data", "ttdb"))
            with open(os.path.join(game, "data", "ttdb", "secret_pool_name.ttpf"), "wb") as handle:
                handle.write(b"not a pool")
            code, out, err = self.run_main("--game-dir", game, "--work", work, "--rs-lib", os.path.join(work, "missing.dll"))
            self.assertEqual(code, 77, err)
            self.assertEqual(out, "")
            code, out, err = self.run_main("--game-dir", game, "--work", work, "--python-only")
            self.assertEqual(code, 0, err)
            self.assertIn("pool_not_opened[ValueError]: 1", out)
            self.assertNotIn("secret_pool_name", out + err)
            self.assertNotIn(game, out + err)
            with open(os.path.join(work, "bcpack_model_count.json"), encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["counts"]["pools_found"], 1)
            self.assertNotIn("secret_pool_name", json.dumps(saved))

    def test_argument_guards(self):
        with tempfile.TemporaryDirectory() as game:
            self.assertEqual(self.run_main("--game-dir", game, "--work", os.path.join(game, "out"), "--python-only")[0], 2)
            self.assertEqual(self.run_main("--game-dir", game, "--work", game + "-work", "--python-only")[0], 2)   # no pool
            self.assertEqual(self.run_main("--game-dir", game + "-nope", "--work", game + "-work", "--python-only")[0], 2)
            self.assertEqual(self.run_main("--game-dir", game, "--work", game + "-work", "--python-stride", "0")[0], 2)
        self.assertFalse(os.path.exists(game + "-work"))

    def test_symbol_disagreement_makes_exit_1(self):
        counts = bmc.Counter({"pools_found": 1, "symbol_disagreements": 2})
        with tempfile.TemporaryDirectory() as game, tempfile.TemporaryDirectory() as work, \
                mock.patch.object(bmc, "scan", return_value=(counts, 0.0)), \
                mock.patch.object(bmc, "RsDecoder", return_value=object()):
            self.assertEqual(self.run_main("--game-dir", game, "--work", work)[0], 1)

    def test_error_classes_carry_nothing_of_the_stream(self):
        for text, want in (("lane 3 (width 8, q 0) at input byte 41: input ran out", "input ran out"),
                           ("lane 1 (width 5, q 2) at input byte 9: fragment did not end at its home state",
                            "fragment not at home state"),
                           ("something new", "other: BcpackError")):
            self.assertEqual(bmc.error_class(bcpack.BcpackError(text)), want)

    def test_default_library_path_is_the_documented_build_output(self):
        path = bmc.default_rs_library().replace("\\", "/")
        self.assertIn("tools/bcpack_calibration/target/release/", path)


if __name__ == "__main__":
    unittest.main()

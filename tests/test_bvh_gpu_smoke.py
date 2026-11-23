"""tools/bvh_gpu_smoke.py without a GPU or game files: the heap image, the word count and the judging of runs, with
a fake vk_compute_test in place of the real one."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bvh_gpu_smoke as smoke  # noqa: E402


def fake_runner(behaviour):
    """A runner that writes the heap image plus per-program changes to the dump, as vk_compute_test's VKC_DUMP does.
    behaviour: name -> ("ok", writes) | ("exit 1",) | ("validation",) | ("timeout",) | ("flaky", writes)."""
    calls = {}

    def run(vkc, spv, init, dump, heap_words, validation, timeout, subgroup=None):
        name = Path(spv).stem
        calls[name] = calls.get(name, 0) + 1
        kind = behaviour.get(name, ("ok", {}))
        if kind[0] not in ("ok", "flaky"):
            return kind[0], "boom"
        heap = bytearray(Path(init).read_bytes())
        assert len(heap) == heap_words * 4
        for offset, word in kind[1].items():
            if kind[0] == "flaky" and calls[name] == 2:
                word ^= 1
            struct.pack_into("<I", heap, offset, word)
        Path(dump).write_bytes(bytes(heap))
        return "ok", "out[0..7] = ..."
    return run, calls


class HeapImage(unittest.TestCase):
    def test_table_then_zeros(self):
        image = smoke.heap_image(0x2000, 8192)
        self.assertEqual(len(image), 8192)
        self.assertEqual(struct.unpack_from("<Q", image, 0)[0], 0x2000)
        self.assertEqual(struct.unpack_from("<Q", image, 4088)[0], 0x2000)
        self.assertEqual(image[4096:], bytes(4096))

    def test_too_small(self):
        with self.assertRaises(ValueError):
            smoke.heap_image(0, 1024)

    def test_changed_words(self):
        a = bytes(16)
        b = struct.pack("<4I", 0, 7, 0, 9)
        self.assertEqual(smoke.changed_words(a, b), 2)


class Smoke(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.spirv = Path(self.dir.name)
        for name in ("Alpha", "Beta", "Gamma", "Delta", "Echo", "LeafRehydration"):
            (self.spirv / (name + ".spv")).write_bytes(b"\x03\x02\x23\x07")

    def tearDown(self):
        self.dir.cleanup()

    def run_smoke(self, behaviour, **kw):
        run, calls = fake_runner(behaviour)
        args = dict(table_qword=0x2000, heap_bytes=8192, validation=True, timeout=5, include_rehydration=False)
        args.update(kw)
        return smoke.smoke(self.spirv, "vkc", runner=run, **args), calls

    def test_statuses(self):
        results, calls = self.run_smoke({
            "Alpha": ("ok", {0x1000: 5, 0x1004: 6}),
            "Beta": ("exit 1",),
            "Gamma": ("validation",),
            "Delta": ("timeout",),
            "Echo": ("flaky", {0x1800: 1}),
        })
        got = {r["name"]: (r["status"], r["words_changed"]) for r in results}
        self.assertEqual(got, {"Alpha": ("pass", 2), "Beta": ("exit 1", None), "Gamma": ("validation", None),
                               "Delta": ("timeout", None), "Echo": ("nondeterministic", 1)})
        self.assertEqual(calls["Alpha"], 2, "a passing program runs twice")
        self.assertEqual(calls["Beta"], 1, "a failing first run is not repeated")
        self.assertEqual([r["name"] for r in results], sorted(got), "name order")

    def test_rehydration_is_left_to_test_376(self):
        results, calls = self.run_smoke({})
        self.assertNotIn("LeafRehydration", calls)
        results, calls = self.run_smoke({}, include_rehydration=True)
        self.assertIn("LeafRehydration", [r["name"] for r in results])


class Checked(unittest.TestCase):
    """--checked: the inputs each program gets and how its output is judged."""

    def test_heap_layout(self):
        heap = smoke.checked_heap([(0x8, 8, smoke.DST), (0x14, 4, 7)], 1 << 16)
        self.assertEqual(struct.unpack_from("<Q", heap, 0x8)[0], smoke.DST)
        self.assertEqual(struct.unpack_from("<I", heap, 0x14)[0], 7)
        self.assertEqual(heap[0x18:0x20], bytes(8), "the rest of the table is zero")
        self.assertEqual(heap[smoke.DST:smoke.DST + 4], bytes([smoke.FILL]) * 4)
        self.assertEqual(heap[smoke.SRC:smoke.SRC + 16], smoke._source_pattern(0))

    def test_expect_entries(self):
        heap = bytearray([smoke.FILL]) * 0x400
        entry = lambda i: struct.pack("<I", i)
        self.assertTrue(smoke._expect_entries(heap, 0, 4, 3, entry), "nothing written yet")
        for i in range(3):
            heap[4 * i:4 * i + 4] = entry(i)
        self.assertEqual(smoke._expect_entries(heap, 0, 4, 3, entry), [])
        heap[12] = 0
        self.assertIn("bytes past entry 2 changed", smoke._expect_entries(heap, 0, 4, 3, entry))

    def test_every_case_passes_with_the_output_it_describes(self):
        """A fake runner that does what each case says the program does: all pass; one wrong word: mismatch."""
        def ideal(spv, heap):
            name = Path(spv).stem
            fields = {off: val for off, _, val in smoke.CHECKED[name][0]} if name in smoke.CHECKED else {}
            n = smoke.COUNT
            q = lambda at: struct.unpack_from("<Q", heap, at)[0]
            d = lambda at: struct.unpack_from("<I", heap, at)[0]
            if name == "InitHeaders":     # read from the table, as the program does
                for i in range(d(0x10)):
                    rec = q(0) + 40 * i
                    f = struct.unpack_from("<6I", heap, rec + 16)
                    struct.pack_into("<16I", heap, q(rec), f[0], 0, f[1], f[2], *[smoke.FMAX] * 3, *[smoke.NFMAX] * 3,
                                     0, f[3], 0xFFFFFFFF, f[4], f[5], 0x7F7F7F)
                    if q(rec + 8):
                        struct.pack_into("<6I", heap, q(rec + 8), *[smoke.FMAX] * 3, *[smoke.NFMAX] * 3)
            elif name == "GetBVHCompactedSize":
                mode, hp = d(0x20), q(0x18)
                h0, h4, h12 = struct.unpack_from("<i", heap, hp)[0], d(hp + 4), d(hp + 12)
                if mode == 1:
                    struct.pack_into("<Q", heap, q(0), (h4 + 1) << 6)
                elif mode == 2:
                    k = h12 if h0 >= 0 else 0
                    struct.pack_into("<QQ", heap, q(0), ((h4 + 1) << 6) + ((8 * k + 0x137) & 0xFFFFFF00), k)
                elif mode == 3:
                    struct.pack_into("<I", heap, q(8), (h4 + 1) << 2)
                    struct.pack_into("<I", heap, q(0x10), h4 + 1)
            elif name == "RadixReset":
                for i in range(n):
                    heap[fields[0] + 16 * i: fields[0] + 16 * (i + 1)] = bytes(16)
            elif name == "ResetKDOP":
                for i in range(n):
                    struct.pack_into("<2I", heap, fields[0] + 8 * i, 0xFF7FFFFF, 0x7F7FFFFF)
            elif name == "ResetToIndexBuffer":
                for i in range(n):
                    struct.pack_into("<4I", heap, fields[8] + 16 * i, *(fields[0x18] + 4 * i + k for k in range(4)))
            elif name == "CopyCounters":
                heap[fields[8]:fields[8] + 16 * n] = heap[fields[0]:fields[0] + 16 * n]
            elif name == "ResetCounterValues":
                for i in range(n):
                    struct.pack_into("<4I", heap, fields[8] + 16 * i, *[fields[0x18]] * 4)
            elif name.startswith("PrefixSum"):
                size = int(name[len("PrefixSum"):])
                values = struct.unpack_from(f"<{size}I", heap, fields[0])
                run = 0
                for i, x in enumerate(values):
                    struct.pack_into("<I", heap, fields[0] + 4 * i, run)
                    run = (run + x) & 0xFFFFFFFF
                struct.pack_into("<I", heap, fields[8], run)

        def runner_for(corrupt):
            def run(vkc, spv, init, dump, heap_words, validation, timeout, subgroup=None):
                heap = bytearray(Path(init).read_bytes())
                ideal(spv, heap)
                if corrupt:   # one wrong byte in a region each case checks
                    stem = Path(spv).stem
                    at = smoke.PREFIX_DATA if stem.startswith("PrefixSum") else smoke.HDR if stem == "InitHeaders" \
                        else smoke.DST
                    heap[at + 4] ^= 1
                Path(dump).write_bytes(bytes(heap))
                return "ok", ""
            return run

        with tempfile.TemporaryDirectory() as d:
            for program in {name.split(":")[0] for name in smoke.CHECKED}:   # one file per program
                (Path(d) / (program + ".spv")).write_bytes(b"\x03\x02\x23\x07")
            good = smoke.run_checked(d, "vkc", heap_bytes=1 << 18, validation=False, timeout=5, runner=runner_for(False))
            self.assertEqual({r["name"]: r["status"] for r in good}, {n: "pass" for n in smoke.CHECKED})
            bad = smoke.run_checked(d, "vkc", heap_bytes=1 << 18, validation=False, timeout=5, runner=runner_for(True))
            self.assertEqual({r["status"] for r in bad}, {"mismatch"})

    def test_missing_program(self):
        with tempfile.TemporaryDirectory() as d:
            results = smoke.run_checked(d, "vkc", heap_bytes=1 << 16, validation=False, timeout=5,
                                        runner=lambda *a: ("ok", ""))
            self.assertEqual({r["status"] for r in results}, {"missing"})


class Cli(unittest.TestCase):
    def test_missing_vk_compute_test(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as e:
                smoke.main(["--spirv", d, "--build", d])
            self.assertIn("vk_compute_test", str(e.exception))

    def test_bad_heap_size(self):
        with self.assertRaises(SystemExit):
            smoke.main(["--spirv", ".", "--heap-bytes", "100"])

    def test_missing_dll(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as e:
                smoke.translate(d, d, 32768)
            self.assertIn("XboxBvhBuilder.dll", str(e.exception))


if __name__ == "__main__":
    unittest.main()

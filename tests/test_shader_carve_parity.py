"""Parity of the fast shader_carve paths with plain reference implementations built inline.

    python -I tests/test_shader_carve_parity.py

No game files and no hard-coded hashes: every expectation is computed here, from the same synthetic input, by the
straightforward code the fast paths replaced (`json.dump(..., indent=1)` of a list of dicts, a `with open` write
per shader, a Counter over the index rows).
- index_entry() writes exactly the text json.dump(indent=1) writes for the same row
- WriteBehind writes the same bytes as a plain write, bounds the queued bytes and raises a failed write
- a carve that goes past the inline writes (threads) gives the same files, index.json and report as the reference
"""
import collections
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import shader_carve  # noqa: E402
from test_shader_carve import make_dxbc, make_dxil_payload, make_rpf_entry  # noqa: E402


def reference_row(name, fname, offset, kind, stage, sm, size, p):
    return {"name": name, "file": fname, "offset": offset, "kind": kind, "stage": stage, "sm": sm, "size": size,
            "parts": {k: v[1] for k, v in p.items()}}


class IndexEntryParity(unittest.TestCase):
    def check(self, rows):
        entries = [shader_carve.index_entry(*r) for r in rows]
        expected = json.dumps([reference_row(*r) for r in rows], indent=1)
        self.assertEqual("[\n" + ",\n".join(entries) + "\n]", expected)
        self.assertEqual(json.loads("[\n" + ",\n".join(entries) + "\n]"), [reference_row(*r) for r in rows])

    def test_rows_match_json_dump_indent(self):
        self.check([
            ("vs_main", "vs_main.dxbc", 12345, 1, "vs", "6.5", 4096, {"DXIL": (100, 20), "PRIV": (200, 7)}),
            ("no_dxil", "no_dxil.dxbc", 0, 0, None, None, 33, {"PRIV": (64, 1)}),
            ("empty_parts", "empty_parts.dxbc", 7, 255, "ps", "6.0", 64, {}),
            ("big~1", "big~1.dxbc", 2 ** 40, 5, "cs", "6.6", 2 ** 31, {"A": (1, 0), "B": (2, 4294967295)}),
            ('quote"back\\slash', 'q.dxbc', 1, 2, "9", "0.15", 40, {"xé": (1, 2)}),
            ("oddÿ\x7f", "o.dxbc", 1, 2, "ms", "6.5", 40, {"\x00\x01\x02\x03": (1, 2)}),
        ])

    def test_single_and_many_rows(self):
        row = ("a", "a.dxbc", 1, 1, "ps", "6.0", 40, {"DXIL": (1, 2)})
        self.check([row])
        self.check([row[:1] + (f"a{i}.dxbc", i) + row[3:] for i in range(50)])


class WriteBehindParity(unittest.TestCase):
    def payloads(self, n):
        return [(f"{i}.dxbc", bytes([i % 251]) * (1 + (i * 37) % 900)) for i in range(n)]

    def test_files_match_plain_writes(self):
        for inline in (0, 3, 1000):
            with self.subTest(inline=inline), tempfile.TemporaryDirectory() as d:
                items = self.payloads(40)
                with shader_carve.WriteBehind(workers=3, inline=inline) as w:
                    for name, data in items:
                        w.throttle(0 if inline == 0 else 100)
                        w.submit(os.path.join(d, name), data)
                    w.drain()
                    self.assertEqual((len(w.queue), w.held), (0, 0))
                self.assertEqual(sorted(os.listdir(d)), sorted(n for n, _ in items))
                for name, data in items:
                    with open(os.path.join(d, name), "rb") as f:
                        self.assertEqual(f.read(), data)

    def test_throttle_bounds_the_queued_bytes(self):
        with tempfile.TemporaryDirectory() as d, shader_carve.WriteBehind(workers=2, inline=0) as w:
            for i in range(60):
                w.throttle(1500)
                self.assertLessEqual(w.held, 1500)
                w.submit(os.path.join(d, f"{i}"), bytes(700))
                self.assertLessEqual(w.held, 1500 + 700)
            w.drain()
            self.assertEqual(w.held, 0)

    def test_a_failed_write_is_raised_and_the_pool_is_joined(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "no_such_folder", "x.dxbc")
            with self.assertRaises(OSError):
                with shader_carve.WriteBehind(workers=2, inline=0) as w:
                    w.submit(missing, b"data")
                    w.drain()
            with self.assertRaises(OSError):  # an inline write raises at once
                with shader_carve.WriteBehind(workers=2, inline=1) as w:
                    w.submit(missing, b"data")

    def test_no_threads_before_the_inline_writes_are_used(self):
        with tempfile.TemporaryDirectory() as d, shader_carve.WriteBehind(inline=2) as w:
            w.submit(os.path.join(d, "a"), b"1")
            w.submit(os.path.join(d, "b"), b"2")
            self.assertIsNone(w.pool)
            w.submit(os.path.join(d, "c"), b"3")
            self.assertIsNotNone(w.pool)
            w.drain()


class CarveParity(unittest.TestCase):
    def build(self, n):
        """n shaders of varying size and stage, a rejected candidate after every fifth, and junk between some."""
        buf = bytearray()
        raws = []
        for i in range(n):
            stage_kind = (0, 1, 5, 99)[i % 4]
            chunks = [(b"PRIV", bytes((i * 7 + j) % 256 for j in range(50 + (i * 131) % 3000)))]
            if i % 9:
                chunks.insert(0, (b"DXIL", make_dxil_payload(0x60 + i % 6, stage_kind)))
            if i % 6 == 0:
                chunks.append((b"HASH", bytes(20)))
            raw = make_dxbc(chunks)
            raws.append(raw)
            offset = len(buf)
            buf += make_rpf_entry(f"shader_{i % 31}" if i % 3 else f"Shader_{i % 31}", i % 7, raw, entry_id=i)
            raws[-1] = (offset, raw)
            if i % 5 == 4:
                buf += make_rpf_entry(f"rejected_{i}", 1, b"JUNK" + bytes((j * 37) % 251 for j in range(300)), entry_id=500 + i)
            if i % 11 == 0:
                buf += bytes(range(256)) * 3
        return bytes(buf), raws

    def reference(self, buf, raws, out):
        """What the carve did before: one dict per shader, json.dump(indent=1), a `with open` write each."""
        index, seen, used = [], {}, set()
        for offset, raw in raws:
            hdr_end = buf.index(b"\x00", offset)
            name = buf[offset:hdr_end].decode("latin-1")
            kind = buf[hdr_end + 1]
            p = shader_carve.parts(raw)
            stage, sm = shader_carve.shader_model(raw, p)
            fname = shader_carve.file_name(name, seen, used)
            with open(os.path.join(out, fname), "wb") as f:
                f.write(raw)
            index.append(reference_row(name, fname, offset, kind, stage, sm, len(raw), p))
        report = [f"{len(index)} shaders carved, {{bad}} candidates rejected",
                  "stages: " + repr(dict(collections.Counter(e["stage"] for e in index))),
                  "shader models: " + repr(dict(collections.Counter(e["sm"] for e in index))),
                  "parts: " + repr(dict(collections.Counter(k for e in index for k in e["parts"])))]
        return json.dumps(index, indent=1), report

    def run_carve(self, buf, out):
        archive = os.path.join(os.path.dirname(out), "a.rpf")
        with open(archive, "wb") as f:
            f.write(buf)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(shader_carve.main([archive, out]), 0)
        return stdout.getvalue().splitlines()

    def test_matches_the_reference_with_and_without_threads(self):
        buf, raws = self.build(150)
        for inline in (64, 0, 100000):
            with self.subTest(inline=inline), tempfile.TemporaryDirectory() as tmp:
                out, ref = os.path.join(tmp, "out"), os.path.join(tmp, "ref")
                os.mkdir(ref)
                real = shader_carve.WriteBehind
                with unittest.mock.patch.object(shader_carve, "WriteBehind",
                                                lambda: real(workers=3, inline=inline)):
                    lines = self.run_carve(buf, out)
                index_text, report = self.reference(buf, raws, ref)
                rejected = int(lines[0].split(", ")[1].split()[0])
                self.assertEqual(rejected, 150 // 5)
                self.assertEqual(lines, [r.replace("{bad}", str(rejected)) for r in report])
                self.assertEqual(sorted(os.listdir(out)), sorted(os.listdir(ref) + ["index.json"]))
                with open(os.path.join(out, "index.json"), "rb") as f:
                    self.assertEqual(f.read(), index_text.encode("utf-8"))
                for name in os.listdir(ref):
                    with open(os.path.join(ref, name), "rb") as a, open(os.path.join(out, name), "rb") as b:
                        self.assertEqual(a.read(), b.read(), name)

    def test_a_failed_write_after_the_inline_ones_is_not_swallowed(self):
        buf, _ = self.build(80)
        with tempfile.TemporaryDirectory() as tmp:
            real_write = shader_carve.write_dxbc
            calls = []

            def flaky(path, data):
                calls.append(path)
                if len(calls) == 70:
                    raise OSError("disk full")
                real_write(path, data)

            with unittest.mock.patch.object(shader_carve, "write_dxbc", flaky), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(OSError):
                    self.run_carve(buf, os.path.join(tmp, "out"))
            self.assertFalse(os.path.exists(os.path.join(tmp, "out", "index.json")))


if __name__ == "__main__":
    unittest.main()

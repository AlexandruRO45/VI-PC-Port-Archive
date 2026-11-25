import hashlib
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from project_tuxbox.runtime_shaders import DXIL_STORE_ID, compile_request, dxil_store, snapshot_compiler_cache


def program(body):
    bitcode = b"BC\xc0\xde" + body
    bitcode += bytes((-len(bitcode)) % 4)
    header = struct.pack("<II4sIII", 0x50060, (24 + len(bitcode)) // 4, b"DXIL", 0x100, 16, len(bitcode))
    chunk = struct.pack("<4sI", b"DXIL", len(header) + len(bitcode)) + header + bitcode
    return b"DXBC" + bytes(16) + struct.pack("<IIII", 1, 36 + len(chunk), 1, 36) + chunk


class SharedDxilStore(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = self.root / "session"
        self.session.mkdir()
        self.release = self.root / "release-xshader"
        self.release.write_bytes(b"translator build 1")
        self.dxc = self.root / "dxc"
        self.dxc.write_text("#!/bin/sh\necho dxc version 1\n")
        self.dxc.chmod(0o700)
        self.dxv = self.root / "dxv"
        self.dxv.write_bytes(b"validator build 1")
        self.hlsl = "void main() {}"
        self.calls = []

    def snapshot(self):
        with patch("project_tuxbox.runtime_shaders.compiler_tools",
                   return_value=(self.release, self.dxc, self.dxv)):
            return snapshot_compiler_cache(self.root / "pinned-xshader", self.root)

    def request(self, data=b"Xbox compute shader"):
        source = self.root / "request.dxbc"
        source.write_bytes(data)
        return source, hashlib.sha256(data).hexdigest()

    def run_tool(self, args, *, timeout):
        name = Path(args[0]).name
        self.calls.append(name)
        if name == "pinned-xshader":
            args[args.index("--hlsl-out") + 1].write_text(self.hlsl)
            args[args.index("--profile-out") + 1].write_text("cs_6_0")
        elif name == "dxc":
            args[args.index("-Fo") + 1].write_bytes(program(args[-1].read_bytes()))

    def compile(self, tools, cache, source, key):
        with patch("project_tuxbox.runtime_shaders.run_tool", side_effect=self.run_tool):
            return compile_request(source, key, tools, cache, self.session)

    def test_translator_rebuild_reuses_dxil_for_unchanged_hlsl(self):
        tools, cache = self.snapshot()
        source, key = self.request()
        self.assertTrue(self.compile(tools, cache, source, key))
        self.assertEqual(self.calls, ["pinned-xshader", "dxc", "dxv"])
        first = (cache / (key + ".cso")).read_bytes()

        self.release.write_bytes(b"translator build 2")
        tools, rebuilt = self.snapshot()
        self.assertNotEqual(cache, rebuilt)
        self.assertEqual(dxil_store(cache), dxil_store(rebuilt))
        self.calls.clear()
        self.assertTrue(self.compile(tools, rebuilt, source, key))
        # The new translator still runs; DXC and DXV are skipped.
        self.assertEqual(self.calls, ["pinned-xshader"])
        self.assertEqual((rebuilt / (key + ".cso")).read_bytes(), first)
        self.assertEqual((rebuilt / (key + ".sha256")).read_text(),
                         hashlib.sha256(first).hexdigest())
        self.assertEqual((self.session / (key + ".cso")).read_bytes(), first)

    def test_changed_hlsl_or_backend_compiles_again(self):
        tools, cache = self.snapshot()
        source, key = self.request()
        self.compile(tools, cache, source, key)

        self.release.write_bytes(b"translator build 2")
        self.hlsl = "void main() { /* changed */ }"
        tools, rebuilt = self.snapshot()
        self.calls.clear()
        self.compile(tools, rebuilt, source, key)
        self.assertEqual(self.calls, ["pinned-xshader", "dxc", "dxv"])

        self.dxv.write_bytes(b"validator build 2")
        tools, revalidated = self.snapshot()
        self.assertNotEqual(dxil_store(rebuilt), dxil_store(revalidated))
        self.calls.clear()
        self.compile(tools, revalidated, source, key)
        self.assertEqual(self.calls, ["pinned-xshader", "dxc", "dxv"])

    def test_damaged_store_entry_is_rebuilt(self):
        tools, cache = self.snapshot()
        source, key = self.request()
        self.compile(tools, cache, source, key)
        store = dxil_store(cache)
        (entry,) = store.glob("*.cso")
        entry.write_bytes(entry.read_bytes()[:-1])

        self.release.write_bytes(b"translator build 2")
        tools, rebuilt = self.snapshot()
        self.calls.clear()
        self.compile(tools, rebuilt, source, key)
        self.assertEqual(self.calls, ["pinned-xshader", "dxc", "dxv"])
        self.assertEqual(entry.read_bytes(), (rebuilt / (key + ".cso")).read_bytes())
        self.assertEqual(entry.with_suffix(".sha256").read_text(), hashlib.sha256(entry.read_bytes()).hexdigest())

    def test_cache_without_store_marker_keeps_previous_behaviour(self):
        cache = self.root / "runtime" / ("0" * 64)
        cache.mkdir(parents=True)
        tools = (self.root / "pinned-xshader", self.dxc, self.dxv)
        source, key = self.request()
        for _ in range(2):
            (cache / (key + ".cso")).unlink(missing_ok=True)
            self.compile(tools, cache, source, key)
        self.assertEqual(self.calls, ["pinned-xshader", "dxc", "dxv"] * 2)
        self.assertFalse((cache.parent / "dxil").exists())
        self.assertIsNone(dxil_store(cache))

    def test_invalid_marker_disables_store(self):
        tools, cache = self.snapshot()
        (cache / DXIL_STORE_ID).write_text("../../elsewhere", encoding="ascii")
        self.assertIsNone(dxil_store(cache))


if __name__ == "__main__":
    unittest.main()

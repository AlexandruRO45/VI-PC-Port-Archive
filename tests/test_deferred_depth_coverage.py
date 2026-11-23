"""CPU-only signature, inventory and sweep evidence joins."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fssm_unpack import make_dxbc, make_fssm, make_xbch, make_xdat
from deferred_depth_coverage import coverage, declarations


class CoverageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.translation = self.root / "translated"
        self.translation.mkdir()
        self.results = self.root / "tag"
        self.results.mkdir()
        blob = make_fssm()
        signature = struct.pack("<II6IBB2xI", 1, 8, 0, 0, 0, 65, 3, 0xffffffff, 1, 0, 0)
        container = make_dxbc([(b"OSG1", signature), (b"DXIL", make_xdat({13: make_xbch(blob)}))])
        (self.root / "original.dxbc").write_bytes(container)
        self.write(self.root / "index.json", [dict(stage="ps", name="Deferred", file="original.dxbc")])
        self.payloads, counts = declarations(self.root)
        self.assertEqual(counts, dict(deferred_entries=1, declared_depth=1))
        self.meta = dict(stage="ps", bindings=[1], fssm_sha256=hashlib.sha256(blob).hexdigest())
        self.index = self.translation / "index.json"
        self.write(self.index, {"alias": self.meta})
        words = [0x07230203, 0x10000, 0, 2, 0, 4 << 16 | 71, 1, 11, 22]
        (self.translation / "alias.spv").write_bytes(struct.pack("<9I", *words))
        self.write(self.root / "sample.json", [{"name": "alias"}])
        self.write(self.results / "results_0.jsonl", dict(i=0, depth=True, code=0, samples_passed=0))

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8", newline="\n")

    def test_renamed_payload_and_missing_readback(self):
        counts = coverage(self.payloads, self.translation, self.results)
        self.assertEqual(counts["frag_depth"], 1)
        self.assertEqual(counts["sample_without_depth_measurement"], 1)
        (self.results / "0.raw.depth").write_bytes(bytes(4096))
        self.assertEqual(coverage(self.payloads, self.translation, self.results)["depth_measured"], 1)

    def test_receipts_survive_reordered_sample_renamed_or_absent_inventory(self):
        log = self.results / "results_0.jsonl"
        digest, = self.payloads
        self.write(log, dict(i=1, depth=True, code=0, source_identity={"fssm_sha256": digest}))
        (self.results / "1.raw.depth").write_bytes(bytes(4096))
        self.write(self.root / "sample.json", [{"name": "renamed"}, {"name": "other"}])
        self.write(self.index, {"other": dict(self.meta, fssm_sha256="b" * 64), "renamed": self.meta})
        self.assertEqual(coverage(self.payloads, self.translation, self.results)["depth_measured"], 1)
        (self.root / "sample.json").unlink()
        self.index.unlink()
        self.assertEqual(coverage(self.payloads, results=self.results)["depth_measured"], 1)
        self.write(self.results / "results_2.jsonl", dict(i=2, code=3, source_identity={"fssm_sha256": digest}))
        self.assertEqual(coverage(self.payloads, results=self.results)["ambiguous_sample"], 1)

    def test_modern_missing_hash_never_falls_back_to_sample_name_or_listing(self):
        self.write(self.results / "results_0.jsonl", dict(i=0, code=0, depth=True, source_identity={
            "src_sha256": self.payloads[0], "fssm_sha256": "invalid"}))
        (self.results / "0.raw.depth").write_bytes(bytes(4096))
        counts = coverage(self.payloads, self.translation, self.results)
        self.assertEqual(counts["unknown_sample_identity"], 1)
        self.assertEqual(counts["depth_measured"], 0)

    def test_duplicate_and_missing_identity(self):
        self.write(self.index, {"first": self.meta, "second": self.meta})
        self.assertEqual(coverage(self.payloads, self.translation), {"ambiguous_identity": 1})
        self.write(self.index, {"original": dict(stage="ps", fssm_sha256="0" * 64)})
        self.assertEqual(coverage(self.payloads, self.translation), {"absent_from_index": 1})
        self.write(self.index, {"original": dict(stage="ps")})
        self.assertEqual(coverage(self.payloads, self.translation), {"unresolved_identity": 1})


if __name__ == "__main__":
    unittest.main()

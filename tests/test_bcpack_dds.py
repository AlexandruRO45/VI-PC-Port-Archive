"""Compare the Rust TTPF batch CLI with the reference on generated pools."""

from contextlib import closing
import hashlib
import json
from itertools import chain
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]

from bcpack_tables import BC1_HISTORY_CONFIRMED
import ttpf_assets
import ttpf_mips as ttpf
import test_ttpf_mips as fixtures
from test_bcpack import encode_bc7_mode6_blocks, encode_lanes
from test_calibrated_texture import synthetic_profile


@unittest.skipUnless(os.environ.get("LEONIDA_BCPACK_DDS"), "set LEONIDA_BCPACK_DDS to the built CLI")
class RustBatchExport(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "dds"
        self.source = self.root / "pool.ttpf"

    def export(self, *args):
        command = [os.environ["LEONIDA_BCPACK_DDS"], self.source, self.out, "--jobs", "2", *args]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((self.out / "manifest.json").read_bytes())

    def check_reference(self):
        with closing(ttpf.Pool(self.source)) as pool:
            expected = ttpf.texture_dds(pool, ttpf.analyze(pool, 0))
        (row,) = self.export()
        self.assertEqual(row.get("mode"), "strict", row)
        self.assertEqual((self.out / row["file"]).read_bytes(), expected)
        self.assertEqual(row["sha256"], hashlib.sha256(expected).hexdigest())

    def test_strict_bc1_with_confirmed_initial_history(self):
        # Exercise every confirmed slot before the chunk has enough blocks
        # to satisfy the corresponding predictor from its own history.
        slots = sorted(BC1_HISTORY_CONFIRMED)
        distances = [256 + i - slot for i, slot in enumerate(slots)]
        fixtures.build_bc1_bcpack_pool(
            self.source, ref_dists=distances + [0] * (1024 - len(distances)),
        )
        self.check_reference()

    def test_strict_multimip_and_fitted_retry(self):
        fixtures.build_bc1_bcpack_pool(self.source)
        self.check_reference()
        old = (self.out / "tex_00000.dds").read_bytes()
        fixtures.build_bc1_bcpack_pool(self.source, ref_dists=[250] + [0] * 1023)
        (failed,) = self.export()
        self.assertEqual(failed["status"], "failed")
        self.assertIn("missing predictor slot 6", failed["reason"])
        self.assertNotIn("file", failed)
        self.assertNotIn("sha256", failed)
        self.assertEqual((self.out / "tex_00000.dds").read_bytes(), old)
        profile = self.root / "profile.json"
        config = synthetic_profile()
        config["bc1"]["endpoints"][250] = [5, 9, 13, 1, 10, 19]
        config["bc1"]["selectors"][0] = [value ^ 1 for value in config["bc1"]["selectors"][0]]
        profile.write_text(json.dumps(config), encoding="utf-8", newline="\n")
        (fitted,) = self.export("--calibration", profile)
        self.assertEqual(fitted.get("mode"), "fitted", fitted)
        data = (self.out / fitted["file"]).read_bytes()
        self.assertNotEqual(data, old)
        self.assertEqual(fitted["sha256"], hashlib.sha256(data).hexdigest())
        try:
            from calibrated_texture import LocalDecoder, native
            native()
        except (ImportError, OSError):
            return
        with closing(ttpf.Pool(self.source)) as pool:
            decoder = LocalDecoder(pool, config)
            self.addCleanup(decoder.read_chunk.cache_clear)
            self.assertEqual(data, decoder.dds(0))

    def test_strict_bc1_rejects_nonzero_reserved_lane(self):
        lanes = [bytes(512) for _ in range(12)]
        lanes[1] = bytes([1]) + bytes(511)
        self.write_compressed_tile(71, 1, encode_lanes(lanes, 1))
        (row,) = self.export()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["reason"], "BC1 block 0 has a nonzero reserved lane")
        self.assertFalse((self.out / "tex_00000.dds").exists())

    def test_mixed_publication_report(self):
        records = [fixtures.pack_etpf_record(4, 4, 0, 1, 71, 1, 1, 4, i * 4096)
                   for i in range(1, 4)]
        fixtures.write_rds5_pool(self.source, records, [bytes(4096)] * 3)
        pool = bytearray(self.source.read_bytes()[:4352])
        lanes = [bytes(512) for _ in range(12)]
        good = encode_lanes(lanes, 1)
        for i, data in enumerate((good, good, good[:1])):
            pad = -len(data) % 256
            entry = len(pool) // 256 | (1 << 26) | ((1 + i) << 30) | (pad << 52)
            struct.pack_into("<Q", pool, 24 + 8 * i, entry)
            pool.extend(data + bytes(pad))
        struct.pack_into("<Q", pool, 48, len(pool) // 256 | (4 << 30))
        self.source.write_bytes(pool)

        def run(directory, report):
            directory.mkdir()
            manifest = {"Textures": [{"Index": i, "DxgiFormat": 71, "File": None} for i in range(3)]}
            result = ttpf_assets.complete_pool(self.source, directory, manifest, ("v1-strict", None),
                                               os.environ["LEONIDA_BCPACK_DDS"], report=report)
            self.assertEqual(result, (2, 1), manifest)
            return {p.name: p.read_bytes() for p in directory.iterdir()}

        invoke = subprocess.run

        def refuse_second(command, **kwargs):
            result = invoke(command, **kwargs)
            result.check_returncode()
            catalog = Path(command[2], "manifest.json")
            rows = json.loads(catalog.read_bytes())
            rows[1] = {"index": 1, "status": "failed", "reason": "controlled batch refusal"}
            catalog.write_text(json.dumps(rows), encoding="utf-8", newline="\n")
            return result

        with mock.patch.object(subprocess, "run", side_effect=refuse_second):
            ordinary = run(self.root / "ordinary", False)
            outputs = run(self.out, True)
        entries = json.loads(outputs.pop("bcpack-export-report.json"))["entries"]
        self.assertEqual(outputs, ordinary)
        self.assertEqual([e["index"] for e in entries], [0, 1, 2])
        self.assertEqual([(e["publisher"], e["outcome"]) for e in entries],
                         [("rust", "published"), ("python", "published"), (None, "blocked")])
        self.assertEqual(entries[1]["rust"]["reason"], "controlled batch refusal")
        self.assertEqual(entries[1]["decoder"], "v1-strict")
        self.assertTrue(entries[2]["rust"]["reason"])
        self.assertTrue(entries[2]["reason"])

    def test_fitted_bc7_signature(self):
        block = bytes.fromhex("40317204a56819be023456789abcdef0")
        data = encode_bc7_mode6_blocks(block * 256)
        # Change the signature in the synthetic stream, so only the explicitly supplied
        # profile can identify the encoded blocks as mode 6.
        import bcpack
        lanes, _ = bcpack.decode_lanes(data, 7, 4096)
        lanes[15] = bytes([21, 0]) * 256
        self.write_compressed_tile(98, 7, encode_lanes(lanes, 7))
        (failed,) = self.export()
        self.assertEqual(failed["status"], "failed")
        labels = [[-1, -1, -1, -1] for _ in range(1024)]
        labels[32 * 21] = [6, 0, 0, 0]
        profile = self.root / "profile.json"
        profile.write_text(json.dumps({"version": 1, "bc7": {
            "labels": labels, "anchors": [1] * 128, "seeds": {"base": [[[0, 0]] * 56] * 256},
        }}), encoding="utf-8", newline="\n")
        (fitted,) = self.export("--calibration", profile)
        self.assertEqual(fitted.get("mode"), "fitted", fitted)
        self.assertEqual((self.out / fitted["file"]).read_bytes()[148:], block)

    def test_fitted_bc7_visible_blocks_and_refinement(self):
        import bcpack

        block = bytes.fromhex("40317204a56819be023456789abcdef0")
        lanes, _ = bcpack.decode_lanes(encode_bc7_mode6_blocks(block * 256), 7, 4096)
        labels = [[-1, -1, -1, -1] for _ in range(1024)]
        labels[32 * 20] = [6, 0, 0, 0]
        seeds = {name: [[[0, 0]] * 56] * 256 for name in ("base", "secondary", "refined")}
        profile = {"version": 1, "entries": {self.source.name: {"0": "refined"}},
                   "bc7": {"labels": labels, "anchors": [1] * 128, "seeds": seeds}}
        config = self.root / "profile.json"

        # An unknown signature outside the 4x4 visible region is immaterial.
        lanes[15] = bytes(lanes[15][:2]) + bytes([21, 0]) + bytes(lanes[15][4:])
        self.write_compressed_tile(98, 7, encode_lanes(lanes, 7))
        config.write_text(json.dumps(profile), encoding="utf-8", newline="\n")
        (row,) = self.export("--calibration", config)
        self.assertEqual(row.get("mode"), "fitted", row)
        self.assertEqual((self.out / row["file"]).read_bytes()[148:], block)

        # The visible block now needs an age-one endpoint predictor.
        lanes[0] = bytes([0, 1]) + bytes(lanes[0][2:])
        self.write_compressed_tile(98, 7, encode_lanes(lanes, 7))
        (row,) = self.export("--calibration", config)
        self.assertEqual(row["status"], "failed")
        key = hashlib.sha256(bytes(chain.from_iterable(zip(*lanes))) + bytes(8)).hexdigest()
        profile["bc7"]["adjustments"] = {key: [[1, field, 0] for field in range(8)]}
        config.write_text(json.dumps(profile), encoding="utf-8", newline="\n")
        (row,) = self.export("--calibration", config)
        self.assertEqual(row.get("mode"), "fitted", row)
        self.assertEqual((self.out / row["file"]).read_bytes()[148:], block)

    def test_fitted_bc5_and_bc6h_history(self):
        profile = self.root / "profile.json"
        for fmt, codec in ((83, 5), (95, 6)):
            with self.subTest(codec=codec):
                lanes = [bytes(512) for _ in range(16 if codec == 6 else 12)]
                # BC5 zigzag and BC6H zigzag relative to the index history
                # both select a distance of one using the upper nibble lane.
                lanes[0] = bytes([0, 1]) + bytes(510)
                settings = {"version": 1}
                if codec == 5:
                    settings["bc5"] = {"255": [13, 17, 19, 23]}
                else:
                    lanes[11:15] = [bytes([0x3f]) * 512 for _ in range(4)]
                    settings["bc6h"] = {
                        "endpointHistory": {"255": [[0x8000, 0x200, 0x20, 0x20]] * 3},
                        "endpointClass": [0] * 255 + [1],
                    }
                self.write_compressed_tile(fmt, codec, encode_lanes(lanes, codec))
                (row,) = self.export()
                self.assertEqual(row["status"], "failed")
                profile.write_text(json.dumps(settings), encoding="utf-8", newline="\n")
                (row,) = self.export("--calibration", profile)
                self.assertEqual(row.get("mode"), "fitted", row)
                payload = (self.out / row["file"]).read_bytes()[148:]
                if codec == 5:
                    self.assertEqual(payload[:2] + payload[8:10], bytes([13, 17, 19, 23]))
                else:
                    lanes[0] = bytes(512)
                    self.write_compressed_tile(fmt, codec, encode_lanes(lanes, codec))
                    (strict,) = self.export()
                    self.assertEqual(strict.get("mode"), "strict", strict)
                    self.assertEqual((self.out / strict["file"]).read_bytes()[148:], payload)

    def write_compressed_tile(self, fmt, codec, data):
        record = fixtures.pack_etpf_record(4, 4, 0, 1, fmt, 1, 1, 4, 4096)
        fixtures.write_rds5_pool(self.source, [record], [bytes(4096)])
        pool = bytearray(self.source.read_bytes())
        start = 4352
        pad = (-len(data)) % 256
        pool[start:] = data + bytes(pad)
        struct.pack_into("<QQ", pool, 24, start // 256 | (codec << 26) | (1 << 30) | (pad << 52),
                         len(pool) // 256 | (2 << 30))
        self.source.write_bytes(pool)

    def test_self_contained_codecs_and_truncation(self):
        for fmt, codec in ((77, 3), (80, 4), (83, 5), (95, 6)):
            with self.subTest(codec=codec):
                lanes = [bytes(512) for _ in range(16 if codec == 6 else 12)]
                if codec == 6:
                    # Mode 1's reset predictor gives selector 4. Apply -4 to each
                    # selector so both anchor pixels fit their two-bit fields.
                    lanes[11:15] = [bytes([0x3f]) * 512 for _ in range(4)]
                data = encode_lanes(lanes, codec)
                self.write_compressed_tile(fmt, codec, data)
                self.check_reference()
                old = (self.out / "tex_00000.dds").read_bytes()
                self.write_compressed_tile(fmt, codec, data[:1])
                (failed,) = self.export()
                self.assertEqual(failed["status"], "failed")
                self.assertNotIn("file", failed)
                self.assertNotIn("sha256", failed)
                self.assertEqual((self.out / "tex_00000.dds").read_bytes(), old)

    def test_array_slice_order(self):
        fixtures.SyntheticArrayPoolTest()._build_array_pool(self.source)
        self.check_reference()

    def test_streaming_cubemap(self):
        fixtures.SyntheticCubemapPoolTest()._build_cubemap_pool(self.source, non_tail=1)
        self.check_reference()

    def test_float_volumes(self):
        for fmt in (10, 54):
            with self.subTest(fmt=fmt):
                fixtures.SyntheticVolumePoolTest()._build_volume_pool(self.source, 17, 13, 9, fmt)
                self.check_reference()


if __name__ == "__main__":
    unittest.main()

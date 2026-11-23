"""scripts/run_gpu_tests.py reads the work folder's spirv/index.json with or without a UTF-8 BOM (no device needed)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gpu_tests as rgt

INDEX = {"cs_fma_fused": {"subgroup_size": 32, "features": ["shaderFmaFloat64"]}}


class SpirvIndexEncoding(unittest.TestCase):
    def read(self, data: bytes) -> dict:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.json"
            path.write_bytes(data)
            return rgt.read_spirv_index(str(path))

    def test_plain_utf8_index(self):
        self.assertEqual(self.read(json.dumps(INDEX).encode("utf-8")), INDEX)

    def test_index_saved_with_a_utf8_bom(self):
        self.assertEqual(self.read(b"\xef\xbb\xbf" + json.dumps(INDEX).encode("utf-8")), INDEX)


if __name__ == "__main__":
    unittest.main()

"""first_light.translate on a spirv/index.json saved with a UTF-8 BOM: the pair is still found.

    python -I tests/test_first_light_bom_index.py

Fully synthetic: the three external tools are stubbed out, so this needs no package, no LLVM and no GPU.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import first_light  # noqa: E402


class BomIndex(unittest.TestCase):
    def test_translate_reads_an_index_with_a_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            dxbc = os.path.join(tmp, "shaders", "dxbc")
            os.makedirs(dxbc)
            open(os.path.join(dxbc, "a_vs.dxbc"), "wb").close()
            open(os.path.join(dxbc, "b_ps.dxbc"), "wb").close()

            def fake_run(cmd):
                if any("rdna2spirv" in str(part) for part in cmd):
                    spirv = os.path.join(tmp, "first_light", "spirv")
                    os.makedirs(spirv, exist_ok=True)
                    with open(os.path.join(spirv, "index.json"), "wb") as f:
                        f.write(b"\xef\xbb\xbf" + json.dumps(
                            {"a_vs": {"stage": "vs"}, "b_ps": {"stage": "ps"}}).encode("utf-8"))

            real_run = first_light.run
            first_light.run = fake_run
            try:
                result = first_light.translate(tmp, [{"name": "a", "file": "a_vs.dxbc", "stage": "vs"},
                                                     {"name": "b", "file": "b_ps.dxbc", "stage": "ps"}])
            finally:
                first_light.run = real_run
            self.assertEqual({k: v["stage"] for k, v in result.items()}, {"vs": "vs", "ps": "ps"})


if __name__ == "__main__":
    unittest.main()

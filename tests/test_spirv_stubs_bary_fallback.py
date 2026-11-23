"""Barycentric fallback PS partners match the actual translated vec3 input.

All shaders are hand-written. No GPU is required; spirv-val validates both
partners when available. The pipeline checker only compiles these stubs.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "semantic"))
import support  # noqa: E402
import rdna2spirv  # noqa: E402
import shader_link_check  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_interface  # noqa: E402
import spirv_stubs  # noqa: E402


def translate(fallback, ordinary_location=0):
    with mock.patch.object(rdna2spirv, "BARY_FALLBACK", fallback):
        if ordinary_location is None:
            text = support.translate_lines("ps_bary_empty", [
                "v_mov_b32 v2, v0",
                "v_mov_b32 v3, v1",
                "exp mrt0 v2, v3, v2, v3 done vm",
                "s_endpgm",
            ])
        else:
            listing = Path(support.FIXTURE_DIR, "ps_bary.s").read_bytes()
            listing = listing.replace(b"attr0.", f"attr{ordinary_location}.".encode("ascii"))
            text = support.translate_listing("ps_bary", listing)
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, text)
    return spirv_asm.assemble(text), meta


class BaryFallbackStubTests(unittest.TestCase):
    def validate(self, data):
        validator = shutil.which("spirv-val")
        if validator:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder, "shader.spv")
                path.write_bytes(data)
                result = subprocess.run([validator, "--target-env", "vulkan1.2", str(path)],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_link(self, vertex, pixel):
        self.validate(vertex)
        self.validate(pixel)
        producer = spirv_interface.reflect_module(vertex, stage="vs")
        consumer = spirv_interface.reflect_module(pixel, stage="ps")
        self.assertEqual(shader_link_check.compare_stages(producer, consumer), [])
        return {varying.address.location: varying.type.describe()
                for declaration in producer.outputs for varying in declaration.varyings}

    def test_native_and_fallback_with_the_same_attributes_get_distinct_partners(self):
        native, native_meta = translate(False)
        fallback, fallback_meta = translate(True)
        self.assertEqual(native_meta["inputs"], [0])
        self.assertEqual(fallback_meta["inputs"], [0])
        self.assertEqual(fallback_meta["bary_fallback_input"], {"location": 1})
        with tempfile.TemporaryDirectory() as folder:
            names = spirv_stubs.generate_for_index(folder, {
                "native": native_meta, "fallback": fallback_meta,
            })
            self.assertEqual(names, ["vs_0.spv", "vs_0_bary1.spv"])
            ordinary = Path(folder, "stubs", "vs_0.spv").read_bytes()
            partner = Path(folder, "stubs", "vs_0_bary1.spv").read_bytes()
            self.assertEqual(ordinary, spirv_stubs.build_vs_stub([0]))
            self.assertEqual(self.assert_link(ordinary, native), {0: "vec4<f32>"})
            self.assertEqual(self.assert_link(partner, fallback), {0: "vec4<f32>", 1: "vec3<f32>"})

    def test_empty_ordinary_inputs_still_need_location_zero_for_fallback(self):
        native, native_meta = translate(False, None)
        fallback, fallback_meta = translate(True, None)
        self.assertEqual(fallback_meta.get("inputs", []), [])
        self.assertEqual(fallback_meta["bary_fallback_input"], {"location": 0})
        with tempfile.TemporaryDirectory() as folder:
            names = spirv_stubs.generate_for_index(folder, {
                "native": native_meta, "fallback": fallback_meta,
            })
            self.assertEqual(names, ["vs_empty.spv", "vs_empty_bary0.spv"])
            self.assertEqual(self.assert_link(Path(folder, "stubs", "vs_empty.spv").read_bytes(), native), {})
            partner = Path(folder, "stubs", "vs_empty_bary0.spv").read_bytes()
            self.assertEqual(self.assert_link(partner, fallback), {0: "vec3<f32>"})

    def test_sparse_inputs_preserve_the_lowest_free_barycentric_location(self):
        pixel, meta = translate(True, 30)
        self.assertEqual(meta["inputs"], [30])
        self.assertEqual(meta["bary_fallback_input"], {"location": 0})
        with tempfile.TemporaryDirectory() as folder:
            names = spirv_stubs.generate_for_index(folder, {"sparse": meta})
            self.assertEqual(names, ["vs_30_bary0.spv"])
            partner = Path(folder, "stubs", names[0]).read_bytes()
            self.assertEqual(self.assert_link(partner, pixel), {0: "vec3<f32>", 30: "vec4<f32>"})

    def test_barycentric_location_is_part_of_the_deduplication_key(self):
        index = {
            "ordinary": {"stage": "ps", "inputs": [3, 1]},
            "null": {"stage": "ps", "inputs": [1, 3], "bary_fallback_input": None},
            "bary0": {"stage": "ps", "inputs": [1, 3], "bary_fallback_input": {"location": 0}},
            "same": {"stage": "ps", "inputs": [3, 1, 3], "bary_fallback_input": {"location": 0}},
            "bary2": {"stage": "ps", "inputs": [1, 3], "bary_fallback_input": {"location": 2}},
            "vertex": {"stage": "vs", "outputs": [3, 1]},
        }
        with tempfile.TemporaryDirectory() as folder:
            names = spirv_stubs.generate_for_index(folder, index)
            self.assertEqual(names, [
                "ps_1_3.spv", "vs_1_3.spv", "vs_1_3_bary0.spv", "vs_1_3_bary2.spv",
            ])
            for name in names:
                self.validate(Path(folder, "stubs", name).read_bytes())
        self.assertEqual(spirv_stubs.stub_name("vs", [3, 1, 3], bary_location=2), "vs_1_3_bary2.spv")
        self.assertEqual(spirv_stubs.stub_name("vs", [], bary_location=0), "vs_empty_bary0.spv")

    def test_bad_fallback_metadata_is_rejected_before_any_stub_is_written(self):
        bad_values = [
            True, 0, "0", [], {}, {"location": None}, {"location": True},
            {"location": "0"}, {"location": -1}, {"location": 1.5},
            {"location": 1 << 32}, {"location": 1},  # collides with the vec4 input
        ]
        for value in bad_values:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                index = {
                    "ordinary": {"stage": "ps", "inputs": []},
                    "bad": {"stage": "ps", "inputs": [1], "bary_fallback_input": value},
                }
                with self.assertRaisesRegex(ValueError, "bad:.*bary"):
                    spirv_stubs.generate_for_index(folder, index)
                self.assertFalse(Path(folder, "stubs").exists())


if __name__ == "__main__":
    unittest.main()

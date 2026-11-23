"""The GPU fixture pair for partial vertex exports (docs/design.md "Partial pixel color and vertex exports").

runtime/test/shaders/vs_partial_exports.asm writes pos0 as x,y and then z,w, and param0 in three partial exports.
ps_partial_exports_vs.asm reads attr0 with v_interp_mov and exports it to mrt0. On a GPU, check.py and
scripts/run_gpu_tests.py draw the pair and compare mrt0 with the words in the PS header.

These tests run the same pair in the SPIR-V interpreter: the translated VS at its three vertices, then the
translated PS with the VS's param0 as its flat attr0. They also keep the header words, check.py's PIXEL_BITS and
run_gpu_tests.py's table in step, so a GPU run checks the same words as this file.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")
sys.path.insert(0, HERE)
sys.path.insert(0, SHADERS)

import support  # noqa: E402
import check    # noqa: E402  (runtime/test/shaders/check.py; importing it runs nothing)
from spirv_mini_interp import run_fragment  # noqa: E402

VS = "vs_partial_exports"
PS = "ps_partial_exports_vs"
ONE = 0x3F800000
MINUS_ONE = 0xBF800000
THREE = 0x40400000
# (2.0, -0.25, 0.375, 0.625): x from the third export, y from the first, z and w from the second.
PARAM0 = (0x40000000, 0xBE800000, 0x3EC00000, 0x3F200000)
# The vs_ids full-screen triangle: x = 2 * ((id << 1) & 2) - 1, y = 2 * (id & 2) - 1, z = 0, w = 1.
POSITIONS = {0: (MINUS_ONE, MINUS_ONE, 0, ONE), 1: (THREE, MINUS_ONE, 0, ONE), 2: (MINUS_ONE, THREE, 0, ONE)}


def source(name):
    with open(os.path.join(SHADERS, name + ".asm"), encoding="utf-8") as f:
        return f.read()


def header_words(name):
    m = re.search(r"Expected mrt0 bits: (.*)", source(name))
    return tuple(int(x, 16) for x in m.group(1).split(","))


def runner_words(name):
    """The tuple scripts/run_gpu_tests.py expects for `name` (its table is local to run_suite, so read the source)."""
    with open(os.path.join(ROOT, "scripts", "run_gpu_tests.py"), encoding="utf-8") as f:
        m = re.search(r'"%s": \(([^)]*)\)' % re.escape(name), f.read())
    return tuple(int(x, 16) for x in m.group(1).split(",")) if m else None


def run_vertex(vertex_id):
    inputs = {"%vertex_index": vertex_id, "%base_vertex": 0, "%instance_index": 0, "%base_instance": 0}
    return run_fragment(support.translate_fixture(VS), inputs=inputs)


class PartialVertexExportFixtures(unittest.TestCase):
    def test_vertex_shader_keeps_earlier_channels(self):
        for vertex_id, position in POSITIONS.items():
            with self.subTest(vertex=vertex_id):
                out = run_vertex(vertex_id)
                self.assertEqual([f"{w:#010x}" for w in out["%out_param0"]], [f"{w:#010x}" for w in PARAM0])
                self.assertEqual([f"{w:#010x}" for w in out["%out_pos"]], [f"{w:#010x}" for w in position])

    def test_pixel_shader_exports_the_vertex_shader_param(self):
        spv = support.translate_fixture(PS)
        self.assertIn("OpDecorate %in_attr0 Flat", spv)   # read only with v_interp_mov: no interpolation
        words = support.mrt0_words(spv, inputs={"%in_attr0": tuple(run_vertex(0)["%out_param0"])})
        self.assertEqual([f"{w:#010x}" for w in words], [f"{w:#010x}" for w in header_words(PS)])

    def test_header_words_are_the_vertex_shader_param(self):
        self.assertEqual(header_words(PS), PARAM0)

    def test_both_gpu_runners_draw_the_pair_and_expect_the_header_words(self):
        self.assertEqual(check.PIXEL_VERTEX.get(PS), VS)
        self.assertIn(f"vk_harness {VS}.spv {PS}.spv", source(PS))
        self.assertEqual(check.PIXEL_BITS.get(PS), header_words(PS))
        self.assertEqual(runner_words(PS), header_words(PS))


if __name__ == "__main__":
    unittest.main()

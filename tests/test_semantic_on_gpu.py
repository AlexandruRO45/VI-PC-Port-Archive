"""scripts/semantic_on_gpu.py without a GPU: which runs it keeps, how it compares words, what it calls approximate.

    python -I tests/test_semantic_on_gpu.py

The draws need vk_harness and a GPU, so they are not run here; collect() runs a small synthetic test folder on the
CPU interpreter. No game files.
"""
import contextlib
import io
import os
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.dont_write_bytecode = True

import semantic_on_gpu as sog  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))
import support  # noqa: E402

PS = ["v_mov_b32 v0, 1.0", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]

FIXTURE = textwrap.dedent('''\
    import unittest

    import spirv_mini_interp as si
    import support

    PS = ["v_mov_b32 v0, 1.0", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    PS2 = ["v_mov_b32 v0, 2.0", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]


    class Fixture(unittest.TestCase):
        def test_repeated(self):
            spv = support.translate_lines("ps_sog_a", PS)
            for _ in range(3):
                self.assertEqual(support.mrt0_words(spv), [0x3F800000] * 4)

        def test_swapped_interpreter(self):
            spv = support.translate_lines("ps_sog_b", PS2)
            saved = si.fma_f32
            si.fma_f32 = lambda x, y, z: 0.0
            try:
                support.mrt0_words(spv)
            finally:
                si.fma_f32 = saved

        def test_many(self):
            for k in range(4):
                spv = support.translate_lines("ps_sog_c", [f"v_mov_b32 v0, {k}", *PS[1:]])
                si.run_fragment(spv, push_constants=[k])
    ''')


class Words(unittest.TestCase):
    def test_nan_against_nan_is_equal(self):
        self.assertTrue(sog.same_word(0x7FFFFFFF, 0x7FC00000))
        self.assertTrue(sog.same_word(0xFFC00000, 0x7FA00000))
        self.assertTrue(sog.same_word(0xA55A7FFF, 0xA55A7E00))      # f16 NaN in the low half, same high half
        self.assertFalse(sog.same_word(0xA55B7FFF, 0xA55A7E00))     # the high half differs and is no NaN
        self.assertFalse(sog.same_word(0x7FC00000, 0x7F800000))     # NaN against inf
        self.assertFalse(sog.same_word(0xFF800000, 0xFFC00000))
        self.assertTrue(sog.same_word(0x7FFF7FFF, 0x7E007E00))      # f16 NaN in both halves
        self.assertTrue(sog.same_word(0x7FFFBEEF, 0x7E00BEEF))      # in the high half only
        self.assertFalse(sog.same_word(0x80000000, 0x00000000))     # -0 against +0

    def test_ulps(self):
        self.assertEqual(sog.ulps(0x3F800000, 0x3F800001), 1)
        self.assertEqual(sog.ulps(0x3F800001, 0x3F800000), 1)
        self.assertEqual(sog.ulps(0x00000000, 0x80000000), 1)
        self.assertEqual(sog.ulps(0x00000001, 0x80000001), 3)
        self.assertGreater(sog.ulps(0x3F800000, 0xBF800000), 1 << 30)

    def test_approximate_only_within_two_ulp_of_an_approximate_instruction(self):
        rsq = "%t2 = OpExtInst %float %glsl InverseSqrt %t1\n"
        self.assertEqual(sog.classify(rsq, [("mrt0[0]", 0x4413C063, 0x4413C062)]), "approx")
        self.assertEqual(sog.classify(rsq, [("mrt0[0]", 0x4413C065, 0x4413C062)]), "diff")
        self.assertEqual(sog.classify("%t2 = OpFDiv %float %a %b\n", [("mrt0[0]", 2, 1)]), "approx")
        fma = "%t35 = OpExtInst %float %glsl Fma %t25 %t28 %t31\n"
        self.assertEqual(sog.classify(fma, [("mrt0[0]", 0x65C00004, 0x65C00005)]), "diff")


class Replayable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spv = support.translate_lines("ps_sog", PS)

    def test_a_translated_module_with_nothing_else(self):
        self.assertIsNone(sog.replayable(self.spv))
        self.assertIsNone(sog.replayable(self.spv, push_constants=[1] * 32, memory={0x1000: 5, 0xFFFC: 1}))

    def test_what_a_draw_cannot_repeat(self):
        self.assertEqual(sog.replayable(self.spv, inputs={"%bary": (0.5, 0.5)}), "inputs")
        self.assertEqual(sog.replayable(self.spv, push_constants=[0] * 33), "push constants past s31")
        self.assertEqual(sog.replayable(self.spv, memory={0x10000: 1}), "memory past 64 KiB")
        self.assertEqual(sog.replayable(self.spv, memory={0x1002: 1}), "memory past 64 KiB")
        self.assertEqual(sog.replayable(self.spv.replace("OpMemberDecorate %UserData 1 Offset 128", "")),
                         "not a translated module")
        with_input = self.spv.replace("OpEntryPoint Fragment %main",
                                      "%in = OpVariable %_ptr_In_float Input\nOpEntryPoint Fragment %main")
        self.assertEqual(sog.replayable(with_input), "inputs")


class Compare(unittest.TestCase):
    CASE = {"mrt0": [1, 2, 3, 4], "heap": [(0x1000, 7)], "mem_after": [(0x1000, 7), (0x1004, 9)]}

    def heap(self, words):
        data = bytearray(0x2000)
        for a, w in words.items():
            data[a:a + 4] = w.to_bytes(4, "little")
        return bytes(data)

    def test_equal(self):
        self.assertEqual(sog.compare(self.CASE, (1, 2, 3, 4), self.heap({0x1000: 7, 0x1004: 9})), [])

    def test_each_kind_of_difference(self):
        got = sog.compare(self.CASE, (1, 2, 3, 5), self.heap({0x1000: 7, 0x1004: 8, 0x1800: 1}))
        self.assertEqual(got, [("mrt0[3]", 5, 4), ("mem[0x1004]", 8, 9), ("mem[0x1800]", 1, 0)])


class Collect(unittest.TestCase):
    def test_a_synthetic_test_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "test_sog_fixture.py"), "w", encoding="utf-8") as f:
                f.write(FIXTURE)
            cases, left_out = sog.collect(start=tmp, per_test=3)
        tests = [c["test"] for c in cases]
        self.assertEqual(tests.count("test_sog_fixture.py::test_repeated"), 1)
        self.assertEqual(tests.count("test_sog_fixture.py::test_many"), 3)
        self.assertNotIn("test_sog_fixture.py::test_swapped_interpreter", tests)
        self.assertEqual(left_out, {"repeated": 2, "a swapped interpreter function": 1, "over --per-test": 1})
        first = next(c for c in cases if c["test"].endswith("test_repeated"))
        self.assertEqual(first["mrt0"], [0x3F800000] * 4)
        self.assertEqual([c["sgprs"] for c in cases if c["test"].endswith("test_many")], [[0], [1], [2]])


class Usage(unittest.TestCase):
    def test_no_vk_harness_is_a_usage_error(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(err):
            self.assertEqual(sog.main(["--build", tmp]), 2)
        self.assertIn("no vk_harness", err.getvalue())


if __name__ == "__main__":
    unittest.main()

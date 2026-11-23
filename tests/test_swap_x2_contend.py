"""The checker of cs_w32_swap_x2_contend (TEST-561), without a GPU: what the heap of a contended
global_atomic_swap_x2 must look like, what counts as a torn pair, and how the dispatch is launched."""
import random
import re
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SHADERS = Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"
sys.path.insert(0, str(SHADERS))
import check  # noqa: E402

N = check.SWAP_X2_INVOCATIONS


def heap(low_order=None, high_order=None):
    """The heap when the invocations' low dwords were exchanged in low_order and their high dwords in high_order
    (invocation order by default): each invocation gets back what the one before it put in."""
    words = [0] * (2 * N + 8)
    for half, order in ((0, low_order or range(N)), (1, high_order or range(N))):
        held = check.SWAP_X2_FILL
        for i in order:
            words[2 * i + half], held = held, check.swap_x2_replacement(i)[half]
        words[2 * N + half] = held
    return words


class Words(unittest.TestCase):
    def test_whole_swaps_in_any_order_pass_with_no_torn_pair(self):
        for seed in (None, 1, 2):
            order = list(range(N))
            if seed is not None:
                random.Random(seed).shuffle(order)
            with self.subTest(seed=seed):
                self.assertEqual(check.swap_x2_contend_words(heap(order, order)), ("", 0))

    def test_dwords_exchanged_in_different_orders_pass_and_are_counted_as_torn(self):
        low, high = list(range(N)), list(range(N))
        random.Random(7).shuffle(high)
        problem, torn = check.swap_x2_contend_words(heap(low, high))
        self.assertEqual(problem, "")
        self.assertGreater(torn, N // 2)

    def test_two_invocations_swapping_at_once_tear_two_pairs(self):
        # A and B both exchange their low dword before either exchanges its high one: A.lo, B.lo, B.hi, A.hi. B gets
        # A's low with the element's old high, and the element keeps B's low with A's high.
        low = list(range(N))
        high = [1, 0] + list(range(2, N))
        problem, torn = check.swap_x2_contend_words(heap(low, high))
        self.assertEqual((problem, torn), ("", 3))   # invocations 0, 1 and 2 hold mixed halves

    def test_a_low_dword_handed_out_twice_fails(self):
        words = heap()
        words[2 * 5] = words[2 * 6]            # invocation 5 got the value invocation 6 got: one lost, one doubled
        problem, _ = check.swap_x2_contend_words(words)
        self.assertRegex(problem, r"^low dwords: 1 never handed out, 1 handed out twice or unknown \(e\.g\. 0x")
        self.assertNotIn("high dwords", problem)

    def test_an_unknown_high_dword_fails(self):
        words = heap()
        words[2 * N + 1] = 0x12345678          # the element's final high dword is nobody's
        problem, _ = check.swap_x2_contend_words(words)
        self.assertIn("high dwords: 1 never handed out, 1 handed out twice or unknown (e.g. 0x12345678)", problem)

    def test_an_untouched_heap_fails_both_halves(self):
        problem, _ = check.swap_x2_contend_words([check.SWAP_X2_FILL] * (2 * N + 8))
        self.assertIn("low dwords: %d never handed out" % N, problem)
        self.assertIn("high dwords: %d never handed out" % N, problem)

    def test_a_short_dump_is_reported(self):
        self.assertEqual(check.swap_x2_contend_words([0] * 64), ("the dump has 64 words, expected at least 4098", 0))

    def test_replacement_pairs_are_unique_and_belong_together(self):
        pairs = [check.swap_x2_replacement(i) for i in range(N)]
        self.assertEqual(len({low for low, _ in pairs}), N)
        self.assertTrue(all(high == low ^ 0xFFFFFFFF and low != check.SWAP_X2_FILL for low, high in pairs))


class Launch(unittest.TestCase):
    def run_test(self, code, data, output=""):
        seen = {}

        def harness(tool, shader, *args, env=None):
            seen.update(tool=tool, shader=shader, args=args, env=env)
            if data is not None:
                Path(env["VKC_DUMP"]).write_bytes(data)
            return code, output

        with patch.object(check, "run", harness):
            return check.swap_x2_contend_test("build", "spirv", {"VK_DRIVER_FILES": "driver.json"}), seen

    def test_dispatches_64_groups_with_a_heap_that_holds_the_element(self):
        (problem, torn), seen = self.run_test(0, struct.pack("<4104I", *heap()))
        self.assertEqual((problem, torn), ("", 0))
        self.assertEqual(seen["args"], ("64", "dump"))
        self.assertEqual(Path(seen["shader"]).name, "cs_w32_swap_x2_contend.spv")
        self.assertEqual((seen["env"]["VKC_OUT_WORDS"], seen["env"]["VK_DRIVER_FILES"]), ("4104", "driver.json"))
        self.assertFalse(Path(seen["env"]["VKC_DUMP"]).exists())   # the temporary folder is gone

    def test_exit_77_is_a_skip(self):
        (problem, _), _ = self.run_test(77, None, "unsupported: no bufferDeviceAddress\n")
        self.assertIsInstance(problem, check.Skip)
        self.assertEqual(problem, "SKIP (unsupported: no bufferDeviceAddress)")

    def test_a_failed_or_dumpless_run_fails(self):
        (problem, _), _ = self.run_test(1, None, "vkCreateDevice failed\n")
        self.assertEqual(problem, "exit 1: vkCreateDevice failed")
        (problem, _), _ = self.run_test(0, None)
        self.assertEqual(problem, "no VKC_DUMP file written")


class Fixture(unittest.TestCase):
    def test_the_listing_uses_the_layout_the_checker_reads(self):
        text = (SHADERS / "cs_w32_swap_x2_contend.asm").read_text(encoding="utf-8")
        code = [line for line in text.splitlines() if line and not line.startswith(";")]
        self.assertIn("; group: %d,1,1" % check.SWAP_X2_LOCAL, text)
        self.assertIn("v_lshl_add_u32 v1, s4, 5, v0", code)                     # i = s4 * 32 + v0
        self.assertEqual(1 << 5, check.SWAP_X2_LOCAL)
        self.assertIn("v_or_b32_e32 v30, 0x51000000, v1", code)
        self.assertEqual(check.swap_x2_replacement(0)[0], 0x51000000)
        self.assertIn("v_mov_b32_e32 v10, %#x" % (8 * N), code)                 # the element follows the pairs
        self.assertEqual(sum(line.startswith("global_atomic_swap_x2 ") and line.endswith(" glc") for line in code), 1)
        self.assertTrue(re.search(r"cs_w32_swap_x2_contend\.spv %d dump" % check.SWAP_X2_GROUPS, text))


if __name__ == "__main__":
    unittest.main()

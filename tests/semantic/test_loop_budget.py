"""--loop-budget: a translated loop that never exits finishes once its per-invocation budget is spent.

Pure Python stdlib only (no llvm-mc, spirv-val, GPU, or game data). Translates synthetic RDNA2 listings with
tools/rdna2spirv.py and executes the emitted SPIR-V with the workgroup interpreter of test_loop_ballot_barrier.py.

  - Off (the default) a loop whose exit never comes runs until the interpreter's step limit: the hang the budget is for.
  - On, the loop leaves after N trips and the code after it runs: a wrong result, but a finished shader.
  - The loops of one invocation share the budget.
  - A budget far above what a correct loop needs changes nothing in its result.
  - The same holds in the dispatch-loop form (--dispatch-loop).

    python -I tests/semantic/test_loop_budget.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
from test_loop_ballot_barrier import VSHARP, _translate_cs, run_compute_workgroup  # noqa: E402

THREADS = 32
START = ["v_lshlrev_b32_e32 v2, 2, v0"]     # v2: this lane's word in the output buffer
END = ["s_endpgm"]
# Each trip stores the loop's counter, so the output shows how far a loop got even if what follows it is skipped.


def counting_loop(counter, step, exit_at, label):
    """`counter` goes up by `step` each trip and is stored; the loop exits when it equals `exit_at`."""
    return [f"s_mov_b32 {counter}, 0", f"{label}:", f"s_add_u32 {counter}, {counter}, {step}",
            f"v_mov_b32_e32 v1, {counter}", "buffer_store_dword v1, v2, s[0:3], 0 offen",
            f"s_cmp_lg_u32 {counter}, {exit_at}", f"s_cbranch_scc1 {label}"]


def finite_loop(trips):
    return counting_loop("s10", 1, trips, "loop")


def runaway_loop(counter="s10", label="spin"):
    """The counter steps by 2 and the exit wants it to equal 1: it never does."""
    return counting_loop(counter, 2, 1, label)


def run(program, budget, structured=True, max_steps=200000):
    """The last word the shader stored for each of its lanes, translated with --loop-budget `budget`."""
    old = getattr(rdna2spirv, "LOOP_BUDGET", 0)
    rdna2spirv.LOOP_BUDGET = budget
    try:
        spv, _gen = _translate_cs("cs_loop_budget", START + program + END, threads=THREADS, structured=structured)
    finally:
        rdna2spirv.LOOP_BUDGET = old
    mem = run_compute_workgroup(spv, THREADS, 32, push_constants=VSHARP, max_steps=max_steps)
    return [mem.get(4 * t) for t in range(THREADS)]


class LoopBudgetTests(unittest.TestCase):
    def test_without_a_budget_a_loop_with_no_exit_does_not_finish(self):
        with self.assertRaises(TimeoutError):
            run(runaway_loop(), 0)

    def test_a_spent_budget_ends_the_loop_and_the_shader_finishes(self):
        # one trip per unit: the counter steps by 2 for 100 trips
        self.assertEqual(run(runaway_loop(), 100), [200] * THREADS)

    def test_the_loops_of_one_invocation_share_the_budget(self):
        # 10 trips round the first loop take 9 units, so the runaway loop gets 91 trips
        program = finite_loop(10) + runaway_loop("s11", "spin2")
        self.assertEqual(run(program, 100), [2 * 91] * THREADS)

    def test_a_large_budget_leaves_a_loop_that_exits_alone(self):
        self.assertEqual(run(finite_loop(10), 65536), [10] * THREADS)
        self.assertEqual(run(finite_loop(10), 0), [10] * THREADS)

    def test_dispatch_loop_form(self):
        self.assertEqual(run(runaway_loop(), 100, structured=False), [200] * THREADS)
        self.assertEqual(run(finite_loop(10), 65536, structured=False), [10] * THREADS)


if __name__ == "__main__":
    unittest.main()

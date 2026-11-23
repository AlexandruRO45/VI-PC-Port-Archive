"""Dispatch cs_w32_swap_x2_contend many times and count what global_atomic_swap_x2 with glc handed out (TEST-561).

    python -I runtime/test/shaders/build.py <out dir> cs_w32_swap_x2_contend
    python -I runtime/test/shaders/swap_x2_contend.py <out dir> <folder with vk_compute_test> [--runs N] [--strict]

check.py and scripts/run_gpu_tests.py dispatch the shader once; this runs it --runs times (default 20) and adds the
counts up, since how often a pair tears depends on the device and on timing. Every invocation of 64 workgroups of 32
swaps its own pair (low = 0x51000000 | i, high = ~low) into one shared 64-bit element and stores the pair it got back
(the .asm has the layout). Whatever order the invocations ran in:

  - per dword: the returned low dwords plus the element's final low dword are exactly the 0xEEEEEEEE fill plus every
    replacement low, each once, and they are one order: going back from the final value, each invocation's returned
    low names the invocation before it, through all of them to the fill. The same for the high dwords. A value handed
    out twice or never, or an order no run can have had, fails the run.
  - per pair: a returned pair (or the final element) whose halves don't belong together is a torn element. The
    translator swaps the two dwords with one 32-bit atomic each (docs/design.md, global atomics), so this can happen
    where the console swaps the element whole. Torn pairs are counted, and fail the run only with --strict.

LEONIDA_VK_DEVICE picks the GPU, as for the other test programs. Exit 0: every run kept both dword sets (and, with
--strict, no pair was torn); 1: a run did not; 77: vk_compute_test can't run the shader on this device.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # this script is also run with python -I
from check import SWAP_X2_INVOCATIONS, SWAP_X2_PAIRS, Skip, swap_x2_contend_test  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", help="build.py's output folder (it holds spirv/cs_w32_swap_x2_contend.spv)")
    parser.add_argument("build", help="the folder with vk_compute_test")
    parser.add_argument("--runs", type=int, default=20, help="dispatches to check (default 20)")
    parser.add_argument("--strict", action="store_true", help="a torn pair fails the run too")
    args = parser.parse_args()
    spirv = os.path.join(args.out, "spirv")
    if not os.path.isfile(os.path.join(spirv, "cs_w32_swap_x2_contend.spv")):
        print(f"{spirv}: no cs_w32_swap_x2_contend.spv (run build.py first)", file=sys.stderr)
        return 2
    bad_runs = torn_runs = torn_total = 0
    for run in range(args.runs):
        problem, torn = swap_x2_contend_test(args.build, spirv)
        if isinstance(problem, Skip):
            print(problem)
            return 77
        if problem:
            print(f"FAIL  run {run}: {problem}")
        bad_runs += bool(problem)
        torn_runs += bool(torn)
        torn_total += torn
    print(f"{args.runs} runs of {SWAP_X2_INVOCATIONS} invocations: {args.runs - bad_runs} kept both dword orders; "
          f"{torn_runs} runs had torn pairs ({torn_total} of {SWAP_X2_PAIRS * args.runs} pairs)")
    return 1 if bad_runs or (args.strict and torn_total) else 0


if __name__ == "__main__":
    sys.exit(main())

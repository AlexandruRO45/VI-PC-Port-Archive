"""Replay the pixel-shader runs of the semantic tests on a GPU and compare them with the CPU interpreter.

tests/semantic checks the translator's SPIR-V on spirv_mini_interp, a CPU model of what Vulkan does. This runs that
suite with `run_fragment` wrapped and keeps each run that a one-pixel draw can repeat, up to --per-test distinct runs
per test. A run qualifies when:
- the translator wrote the module (its %UserData push-constant block holds the heap base, which turns a guest address
  into the device address vk_harness passes);
- it has no varyings and no Input variable;
- its push constants fit in s0..s31;
- its memory is below 64 KiB.
A run made while a test had swapped a function of the interpreter (test_fma_exact splits Fma, for example) is left
out, because it models another driver.

Each kept run is drawn by vk_harness into one pixel. VKH_SGPRS holds the push constants, VKH_HEAP the memory before
the run, and VKH_HEAP_DUMP the memory after it. The four MRT0 words and every heap word are compared with the
interpreter's:

    python -I scripts/semantic_on_gpu.py --build runtime/build/<dir> [--per-test 25] [--jobs 4]
        [--files 'test_alu_*.py'] [--json report.json]

A word that is NaN on both sides (f32, or either f16 half) counts as equal: SPIR-V leaves NaN bits to the driver
(NVIDIA writes 0x7fffffff where RDNA2 keeps 0x7fc00000). Some other differences are the driver's right, too: a
module with an approximate instruction (OpFDiv, InverseSqrt, Sqrt, Exp, Log, Sin, Cos, Tan, Pow) may be 1 or 2 ulp
away. These are reported as "approx", and a module spirv-as rejects (a test that breaks one on purpose) as
"invalid". Everything else is a "diff": the GPU and the model disagree, so the translator
leans on behaviour the model has and the driver doesn't, or the model is wrong. Exit 1 on a diff, 77 when vk_harness
cannot draw on this machine, 2 on a usage error, 0 otherwise.

Needs spirv-as and vk_harness (built in --build). No game files.
"""
import argparse
import collections
import concurrent.futures
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
HEAP_BYTES = 0x10000          # vk_harness's VKH_HEAP: Xbox VA 0..64 KiB
UNSUPPORTED = 77
TRANSLATED = re.compile(r"^\s*OpMemberDecorate %UserData 1 Offset 128\s*$", re.M)
INPUT_VAR = re.compile(r"OpVariable %\S+ Input\b")
APPROX = re.compile(r"\bOpFDiv\b|OpExtInst %\S+ %\S+ (InverseSqrt|Sqrt|Exp2|Exp|Log2|Log|Sin|Cos|Tan|Pow)\b")


def replayable(text, inputs=None, push_constants=None, memory=None):
    """Why one run_fragment call can't be drawn by vk_harness as it ran on the CPU, or None if it can."""
    if not TRANSLATED.search(text):
        return "not a translated module"
    if inputs or INPUT_VAR.search(text):
        return "inputs"
    if len(push_constants or ()) > 32:
        return "push constants past s31"
    if any(not 0 <= a <= HEAP_BYTES - 4 or a % 4 for a in (memory or {})):
        return "memory past 64 KiB"
    return None


def _nan32(w):
    return w & 0x7F800000 == 0x7F800000 and w & 0x7FFFFF != 0


def _nan16(h):
    return h & 0x7C00 == 0x7C00 and h & 0x3FF != 0


def same_word(gpu, cpu):
    """Equal, or NaN on both sides: as f32, or in each f16 half that differs. An f32 infinity is never read as a
    pair of halves (its high half 0x7f80 looks like an f16 NaN), so inf against NaN stays a difference."""
    if gpu == cpu or (_nan32(gpu) and _nan32(cpu)):
        return True
    if gpu & 0x7FFFFFFF == 0x7F800000 or cpu & 0x7FFFFFFF == 0x7F800000:
        return False
    for shift in (0, 16):
        a, b = gpu >> shift & 0xFFFF, cpu >> shift & 0xFFFF
        if a != b and not (_nan16(a) and _nan16(b)):
            return False
    return True


def ulps(a, b):
    """Distance between two binary32 words in units in the last place (+0 and -0 are 1 apart)."""
    def ordered(w):
        return w if w < 0x80000000 else -(w & 0x7FFFFFFF) - 1
    return abs(ordered(a) - ordered(b))


def classify(text, mismatches):
    """'approx' when every mismatching word is within 2 ulp and the module has an approximate instruction."""
    if APPROX.search(text) and all(ulps(g, c) <= 2 for _, g, c in mismatches):
        return "approx"
    return "diff"


def _test_name():
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name.startswith("test"):
            return f"{os.path.basename(frame.f_code.co_filename)}::{frame.f_code.co_name}"
        frame = frame.f_back
    return "?"


def collect(start=SEMANTIC, files="test*.py", per_test=25):
    """Run the semantic tests in `start` with run_fragment wrapped; return (cases, counts of what was left out)."""
    for path in (TOOLS, SEMANTIC):
        if path not in sys.path:
            sys.path.insert(0, path)
    import spirv_mini_interp as si
    import support
    original = si.run_fragment
    pristine = {k: v for k, v in vars(si).items() if callable(v) and k != "run_fragment"}
    cases, seen, per = [], set(), collections.Counter()
    left_out = collections.Counter()

    def wrapped(text, inputs=None, push_constants=None, memory=None):
        before = sorted(memory.items()) if memory is not None else None
        patched = any(getattr(si, k, None) is not v for k, v in pristine.items())
        out = original(text, inputs=inputs, push_constants=push_constants, memory=memory)
        why = "a swapped interpreter function" if patched else replayable(text, inputs, push_constants, memory)
        test = _test_name()
        if why is None and per[test] >= per_test:
            why = "over --per-test"
        if why is None:
            key = (text, tuple(push_constants or ()), tuple(before or ()))
            if key in seen:
                why = "repeated"
            else:
                seen.add(key)
                per[test] += 1
                cases.append({"test": test, "text": text, "sgprs": list(push_constants or ()), "heap": before or [],
                              "mem_after": sorted(memory.items()) if memory is not None else None,
                              "mrt0": list(out["%out_mrt0"]) if "%out_mrt0" in out else None})
        if why:
            left_out[why] += 1
        return out

    si.run_fragment = wrapped
    support.run_fragment = wrapped
    try:
        suite = unittest.defaultTestLoader.discover(start, pattern=files, top_level_dir=start)
        unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    finally:
        si.run_fragment = original
        support.run_fragment = original
    return cases, left_out


def compare(case, px, heap):
    """The (what, gpu word, cpu word) that differ: MRT0's four words, then every heap word either side wrote."""
    out = []
    if case["mrt0"] is not None and len(case["mrt0"]) == 4:
        for i, (g, c) in enumerate(zip(px, case["mrt0"])):
            if not same_word(g, c):
                out.append((f"mrt0[{i}]", g, c))
    if case["mem_after"] is not None:
        cpu, init = dict(case["mem_after"]), dict(case["heap"])
        words = struct.unpack(f"<{len(heap) // 4}I", heap)
        for a in sorted(set(cpu) | {4 * i for i, w in enumerate(words) if w}):
            if a < len(heap):
                g, c = words[a // 4], cpu.get(a, init.get(a, 0))
                if not same_word(g, c):
                    out.append((f"mem[{a:#x}]", g, c))
    return out


def draw(case, harness, vs, spirv_as, work):
    """Draw one case; return ('ok' | 'approx' | 'diff', mismatches) or ('invalid' | 'unsupported' | 'error', text)."""
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        spv, raw, dump = (os.path.join(tmp, n) for n in ("ps.spv", "out.raw", "heap.bin"))
        try:
            spirv_as(case["text"], spv)
        except RuntimeError as e:   # a negative control that breaks its module on purpose
            return "invalid", str(e).strip().splitlines()[-1]
        env = dict(os.environ, VKH_HEAP=",".join(f"{a:#x}={w:#x}" for a, w in case["heap"]) or "0=0",
                   VKH_HEAP_DUMP=dump)
        env.pop("VKH_SGPRS", None)
        if case["sgprs"]:
            env["VKH_SGPRS"] = ",".join(f"{w:#x}" for w in case["sgprs"])
        try:
            p = subprocess.run([harness, vs, spv, raw, "1", "1"], env=env, capture_output=True, text=True,
                               timeout=120)
        except subprocess.TimeoutExpired:
            return "error", "vk_harness timed out"
        if p.returncode == UNSUPPORTED:
            return "unsupported", (p.stderr or p.stdout).strip()[-200:]
        if p.returncode or not os.path.isfile(raw) or not os.path.isfile(dump):
            return "error", f"vk_harness exit {p.returncode}: {(p.stderr or p.stdout).strip()[-200:]}"
        with open(raw, "rb") as f:
            px = struct.unpack("<4I", f.read(16))
        with open(dump, "rb") as f:
            heap = f.read()
    bad = compare(case, px, heap)
    return ("ok" if not bad else classify(case["text"], bad)), bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="the folder vk_harness was built in")
    ap.add_argument("--per-test", type=int, default=25, help="distinct runs kept per test (default 25)")
    ap.add_argument("--jobs", type=int, default=4, help="draws at a time (default 4)")
    ap.add_argument("--files", default="test*.py", help="which tests/semantic files to run (a glob, default test*.py)")
    ap.add_argument("--json", help="also write every case's result here")
    a = ap.parse_args(argv)
    if a.per_test < 1 or a.jobs < 1:
        ap.error("--per-test and --jobs must be at least 1")
    harness = os.path.join(a.build, "vk_harness" + (".exe" if os.name == "nt" else ""))
    if not os.path.isfile(harness):
        print(f"ERROR: no vk_harness in {a.build}: build the runtime test programs first", file=sys.stderr)
        return 2
    sys.path.insert(0, HERE)
    import run_gpu_tests
    cases, left_out = collect(files=a.files, per_test=a.per_test)
    print(f"{len(cases)} runs to draw; left out: "
          + (", ".join(f"{n} {why}" for why, n in sorted(left_out.items())) or "none"), flush=True)
    if not cases:
        return 0
    with tempfile.TemporaryDirectory() as work:
        vs = os.path.join(work, "vs_tri.spv")
        run_gpu_tests.spirv_as(run_gpu_tests.VS_TRI_SPVASM, vs)
        first = draw(cases[0], harness, vs, run_gpu_tests.spirv_as, work)
        if first[0] == "unsupported":
            print(f"SKIP: vk_harness cannot draw here: {first[1]}")
            return UNSUPPORTED
        results = [first]
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
            results += pool.map(lambda c: draw(c, harness, vs, run_gpu_tests.spirv_as, work), cases[1:])
    counts = collections.Counter(status for status, _ in results)
    by_test = collections.defaultdict(list)
    for case, (status, detail) in zip(cases, results):
        if status != "ok":
            by_test[(status, case["test"])].append(detail)
    for (status, test), details in sorted(by_test.items()):
        print(f"{status:6} {test}: {len(details)} run{'s' if len(details) != 1 else ''}")
        for detail in details[:3]:
            if isinstance(detail, str):
                print(f"         {detail}")
            else:
                print("         " + "; ".join(f"{w} gpu={g:#010x} cpu={c:#010x}" for w, g, c in detail[:4]))
    print(", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump([{"test": c["test"], "status": s,
                        "detail": d if isinstance(d, str) else [[w, g, cpu] for w, g, cpu in d]}
                       for c, (s, d) in zip(cases, results)], f, indent=1)
    if counts["error"]:
        return 2
    return 1 if counts["diff"] else 0


if __name__ == "__main__":
    sys.exit(main())

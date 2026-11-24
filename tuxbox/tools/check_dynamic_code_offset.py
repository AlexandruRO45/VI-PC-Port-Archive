#!/usr/bin/env python3
"""GPU differential check of the bounded dynamic embedded-table load.

Input and output instrumentation only; the translator's address and load
instructions are unchanged. No title files are used.
"""
import argparse
import json
import random
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build import find_dxc
from frosty_gpu_guest import run_gpu_guest
from project_tuxbox.inspect import digest


def instrument(hlsl):
    start = "    // 0x0000 s_getpc_b64"
    end = "    // 0x0014 s_endpgm"
    if hlsl.count(start) != 1 or hlsl.count(end) != 1 or "tb_code[t1 + 0u]" not in hlsl:
        raise ValueError("unexpected translated fixture")
    hlsl = hlsl.replace(start, "    s12 = output[0];\n" + start)
    hlsl = hlsl.replace(end, "    output[64u + gtid.x] = s16;\n" + end)
    return "RWBuffer<uint> output : register(u0);\n" + hlsl


def cases():
    rng = random.Random(0x53434152)
    values = [0, 1, 2, 3, 4, 7, 0xffffffff, 0x80000002]
    values += [rng.getrandbits(32) for _ in range(56)]
    for i, value in enumerate(values):
        seed = [0x76540000 + n for n in range(384)]
        seed[96] = value
        expected = seed.copy()
        expected[160:224] = [11 * ((value & 3) + 1)] * 64
        yield f"offset_{i:02}", struct.pack("<384I", *seed), struct.pack("<384I", *expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hlsl", "provider", "proton", "prefix", "compiler-dir", "work", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    dxc = Path(find_dxc(None)).resolve()
    with TemporaryDirectory(dir=work) as tmp:
        trial = Path(tmp)
        source = trial / "table.hlsl"
        source.write_text(instrument(args.hlsl.read_text()))
        blob = trial / "table.cso"
        subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", blob, source], check=True, timeout=30)
        subprocess.run([dxc.with_name("dxv"), blob], check=True, timeout=30)
        expected = {}
        for name, seed, reference in cases():
            (trial / f"{name}.seed").write_bytes(seed)
            (trial / f"{name}.cso").hardlink_to(blob)
            expected[name] = reference
        (trial / "cases.txt").write_text("".join(f"{name} 0 uint 1\n" for name in expected))
        exe, status, fields = run_gpu_guest(
            ROOT / "tools/reference/independent-shaders/compute.cpp", "dynamic-code-offset.exe",
            trial, work, args.proton.resolve(), args.prefix, args.provider, args.compiler_dir)
        mismatches = [name for name, ref in expected.items()
                      if not (trial / f"{name}.bin").is_file() or (trial / f"{name}.bin").read_bytes() != ref]
        report = dict(passed=status == 0 and fields.get("adapter") == args.adapter and not mismatches,
                      scope="Synthetic runtime table indexing, input/output instrumentation only",
                      adapter=fields.get("adapter"), cases=len(expected), compared_words=len(expected)*384,
                      result_words=len(expected)*64, mismatches=mismatches, guest_exit=status,
                      emitter_hlsl_sha256=digest(args.hlsl), dxil_sha256=digest(blob),
                      dxc_sha256=digest(dxc), guest_sha256=digest(exe),
                      provider_sha256=digest(args.provider))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("GPU differential check failed")
    print(f"PASS: {report['cases']} cases, {report['compared_words']} words")

if __name__ == "__main__":
    main()

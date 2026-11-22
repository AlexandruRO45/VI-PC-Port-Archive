r"""v_fract_f16 with an output modifier: the rule measured on RDNA2, and a check of it. Standard library only.

docs/design.md ("Float min/max/med3 with NaN, `v_fract`") left open in which order v_fract_f16 rounds and applies
`mul:2`/`mul:4`/`div:2` (SHADER-338). Measured on gfx1036 (RDNA2) over all 65,536 f16 inputs, the eight modifier
forms below and four MODEs (`rule()` is the result):

    1. A NaN input passes through, an sNaN quieted under IEEE 1 (payload kept); +-inf gives 0xfe00. With
       f16 denormals flushed (MODE.denorm_mode_16_64 = 0) a denormal input reads as zero.
    2. fract(x) = x - floor(x) is rounded to f16 first; a result that rounds to 1.0 becomes 0x3bff (fract(-2^-14)).
    3. The output modifier then scales that f16 word, but only under IEEE 0 with f16 denormals flushed; in every other
       MODE it is ignored (SHADER-135, SHADER-297). The product is exact except below 2^-14, where it flushes to +0
       before rounding (0x07ff div:2 is +0, not the smallest normal 0x0400). So 0x8400 (-2^-14) mul:2 gives 0x3fff
       (2 * 0x3bff), not 0x4000 (2 * the exact 1 - 2^-15, rounded once).
    4. `clamp` comes last: [0, 1], and a NaN gives +0 (DX10_CLAMP 1).

Sides:

    rule        the hand-worked rows in ROWS against rule(). Always runs.
    translator  tools/rdna2spirv.py -> tests/semantic/spirv_mini_interp.py on the CPU, in the MODE the translator
                assumes (IEEE 0, f16 denormals kept, so the modifier is ignored): the inputs of SAMPLE (or all 65,536
                with --all), each modifier form, against rule(). A NaN result only has to be a NaN, since SPIR-V
                doesn't fix NaN bits. Under `clamp` the NaN and infinity inputs are left out: the interpreter refuses
                FClamp on a NaN, which GLSL.std.450 leaves undefined (the hardware gives +0).
    hw         the native instruction on an AMD GPU (tools/hw_oracle), all 65,536 inputs, each modifier form, in the
                four MODEs IEEE 0/1 x f16 denormals flushed/kept (DX10_CLAMP 1), wave32, and one block of 1,024
                inputs per run in wave64; every word against rule(), bit for bit.

    python -I tools/fract_f16_probe.py                  # rule and translator
    python -I tools/fract_f16_probe.py --oracle hw      # + the AMD GPU in this machine

Exit code: 0 when every side that ran follows rule(), 1 when one does not, 2 on a harness error, 77 when the hardware
oracle is unavailable. Nothing here reads game files.
"""
import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))

MODS = ["", "mul:2", "mul:4", "div:2", "clamp", "clamp mul:2", "clamp mul:4", "clamp div:2"]
SCALE = {"mul:2": 2.0, "mul:4": 4.0, "div:2": 0.5}
HW_MODES = [dict(ieee=i, dx10_clamp=1, denorm32=0, denorm16_64=d) for i in (0, 1) for d in (0, 3)]
TRANSLATOR_MODE = dict(ieee=0, dx10_clamp=1, denorm32=0, denorm16_64=3)
BLOCK = 1024                    # inputs per kernel / workgroup: lane i reads base + i
HIGH = 0x5A5A0000               # the destination's high half, which v_fract_f16 keeps
MIN_NORMAL = 2.0 ** -14


def f16(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f16_word(x):
    """x (a non-negative float exactly representable in f16, or the f16 product of one by 2, 4 or 1/2) as f16 bits."""
    return struct.unpack("<H", struct.pack("<e", x))[0]


def is_nan(h):
    return (h & 0x7C00) == 0x7C00 and (h & 0x3FF) != 0


def rule(x, mod="", ieee=0, flush=False):
    """The measured low half of v_fract_f16_e64 d, x <mod> on RDNA2. flush: MODE.denorm_mode_16_64 = 0."""
    clamp = "clamp" in mod
    scale = SCALE.get(mod.replace("clamp", "").strip())
    if (x & 0x7C00) == 0x7C00:
        if not is_nan(x):
            h = 0xFE00                                      # fract(+-inf)
        else:
            h = x | 0x0200 if ieee and not x & 0x0200 else x
        return 0 if clamp else h
    if flush and (x & 0x7C00) == 0:
        x = 0
    v = f16(x)
    h = f16_word(v - math.floor(v) + 0.0)                   # exact in a double (+ 0.0: fract(-0) = +0), rounded
    if h == 0x3C00:
        h = 0x3BFF
    if scale and not ieee and flush:
        p = f16(h) * scale                                  # exact: a power-of-two multiple of an f16 value
        h = 0 if p < MIN_NORMAL else f16_word(p)
    if clamp and h > 0x3C00:
        h = 0x3C00
    return h


# (input, modifier, ieee, flush, expected): worked by hand from the f16 encoding.
ROWS = [
    (0x3E00, "", 0, False, 0x3800),             # fract(1.5) = 0.5
    (0xBD00, "", 0, False, 0x3A00),             # fract(-1.25) = -1.25 - (-2) = 0.75
    (0x8400, "", 0, False, 0x3BFF),             # 1 - 2^-14 rounds to 1.0 in f16 -> 0x3bff
    (0x8400, "mul:2", 0, True, 0x3FFF),         # 2 * 0x3bff: rounded fract first, then the modifier
    (0x8400, "mul:4", 0, True, 0x43FF),
    (0x8400, "div:2", 0, True, 0x37FF),
    (0x8400, "mul:2", 0, False, 0x3BFF),        # f16 denormals kept: the modifier is ignored
    (0x8400, "mul:2", 1, True, 0x3BFF),         # IEEE 1: ignored
    (0x8400, "clamp mul:2", 0, True, 0x3C00),   # clamp after the modifier
    (0x07FF, "div:2", 0, True, 0x0000),         # 0x07ff / 2 < 2^-14: flushed before rounding (not 0x0400)
    (0x0800, "div:2", 0, True, 0x0400),
    (0x8001, "", 0, False, 0x3BFF),             # a kept negative denormal
    (0x8001, "", 0, True, 0x0000),              # a flushed one reads as -0, fract(-0) = +0
    (0x8000, "", 0, False, 0x0000),
    (0x7C00, "", 0, False, 0xFE00),
    (0xFC00, "mul:2", 0, True, 0xFE00),
    (0x7C01, "", 0, False, 0x7C01),             # sNaN kept under IEEE 0
    (0x7C01, "", 1, False, 0x7E01),             # quieted under IEEE 1, payload kept
    (0xFE05, "mul:4", 0, True, 0xFE05),
    (0x7E00, "clamp", 0, False, 0x0000),        # NaN under clamp: +0 (DX10_CLAMP 1)
    (0x3B00, "clamp mul:4", 0, True, 0x3C00),
]

# Translator side: every 64th input plus each range where the rule has an edge (tiny negatives that round to 1.0,
# denormals, the smallest normals, NaN and infinity).
SAMPLE = sorted(set(range(0, 0x10000, 64)) | set(range(0x8000, 0x8000 + BLOCK)) | set(range(0x8400, 0x8400 + BLOCK))
                | set(range(0x0000, 0x0000 + 64)) | set(range(0x07C0, 0x0840)) | set(range(0x7C00, 0x7C40))
                | set(range(0xFC00, 0xFC40)))
NAN_BLOCKS = (0x7C00, 0xFC00)                   # the blocks of +-inf and the NaNs
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]          # s[0:3]: a buffer at 0 covering the whole heap


def cs_lines(base, mod, store):
    return ["v_lshlrev_b32_e32 v5, 2, v0", f"v_add_nc_u32_e32 v1, 0x{base:x}, v0", f"v_mov_b32 v8, 0x{HIGH:x}",
            f"v_fract_f16_e64 v8, v1 {mod}".rstrip(), store]


def run_translator(inputs, mods=MODS):
    """{mod: {x: word}} from the translated compute shader, one 1,024-lane workgroup per block of inputs."""
    import rdna2spirv
    import spirv_mini_interp as si
    blocks = sorted({x - x % BLOCK for x in inputs})
    fssm = bytearray(0x148)
    struct.pack_into("<4I", fssm, 0x138, 0x0088, BLOCK, 1, 1)   # 4 user SGPRs (the V#), TGID_X, v0 = local id.x
    out = {}
    with tempfile.TemporaryDirectory(prefix="fract_f16_probe_") as work:
        os.makedirs(os.path.join(work, "disasm"))
        os.makedirs(os.path.join(work, "fssm"))
        with open(os.path.join(work, "cs_fract_f16_probe.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(work, "fssm", "cs_fract_f16_probe.fssm"), "wb") as f:
            f.write(bytes(fssm))
        src = os.path.join(work, "disasm", "cs_fract_f16_probe.s")
        for mod in mods:
            words = out[mod] = {}
            for base in blocks:
                if "clamp" in mod and base in NAN_BLOCKS:
                    continue            # the interpreter refuses FClamp on a NaN, which GLSL.std.450 leaves undefined
                lines = cs_lines(base, mod, "buffer_store_dword v8, v5, s[0:3], 0 offen") + ["s_endpgm"]
                with open(src, "w", encoding="utf-8", newline="\n") as f:
                    f.write("; fract_f16_probe\n" + "".join(f"\t{ln}\n" for ln in lines))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    spv = rdna2spirv.translate(src)
                mem = {}
                si.run_workgroup(spv, BLOCK, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
                                 push_constants=VSHARP + [0] * 28, memory=mem)
                for i in range(BLOCK):
                    if base + i in inputs:
                        words[base + i] = mem.get(4 * i)
    return out


def check_translator(words):
    bad = []
    for mod, got in words.items():
        for x, w in got.items():
            want = HIGH | rule(x, mod, TRANSLATOR_MODE["ieee"], False)
            if w != want and not (w is not None and w >> 16 == HIGH >> 16 and is_nan(w) and is_nan(want)):
                bad.append((x, mod, w, want))
    return bad


def hw_kernel(base, mod, mode, wave):
    from hw_oracle import asm
    return asm.Kernel(cs_lines(base, mod, "global_store_dword v5, v8, s[2:3]") + ["s_endpgm"], wave=wave,
                      threads=BLOCK, words=BLOCK, raw=True, mode=mode)


def run_hw(oracle):
    """[(mode, mod, wave, x, word, want)] for every mismatch, and the number of words compared."""
    bad, n = [], 0
    for mode in HW_MODES:
        flush = mode["denorm16_64"] == 0
        for mod in MODS:
            for wave, bases in ((32, range(0, 0x10000, BLOCK)), (64, [0x8000])):
                got = oracle.run([hw_kernel(b, mod, mode, wave) for b in bases], words=BLOCK)
                for base, words in zip(bases, got):
                    if isinstance(words, str):
                        raise RuntimeError(words)
                    for i, w in enumerate(words):
                        x = base + i
                        want = HIGH | rule(x, mod, mode["ieee"], flush)
                        n += 1
                        if w != want:
                            bad.append((mode, mod, wave, x, w, want))
    return bad, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--oracle", default="", help="hw: also run the native instruction on this machine's AMD GPU")
    ap.add_argument("--all", action="store_true", help="translator side: all 65,536 inputs (about 15 minutes)")
    ap.add_argument("--hw-arch", help="GPU architecture to assemble for (default: detected; LEONIDA_HW_ARCH)")
    a = ap.parse_args(argv)
    sides = {s.strip() for s in a.oracle.split(",") if s.strip()} - {"translator", "rule"}
    if sides - {"hw"}:
        ap.error(f"unknown --oracle side(s): {', '.join(sorted(sides - {'hw'}))}")
    rows_bad = [r for r in ROWS if rule(*r[:4]) != r[4]]
    for x, mod, ieee, flush, want in rows_bad:
        print(f"rule: {x:04x} {mod or '(none)'} ieee{ieee} flush={flush}: {rule(x, mod, ieee, flush):04x}, "
              f"worked by hand {want:04x}")
    print(f"rule: {len(ROWS) - len(rows_bad)}/{len(ROWS)} hand-worked rows")
    if rows_bad:
        return 2
    failed = False
    try:
        inputs = set(range(0x10000)) if a.all else set(SAMPLE)
        words = run_translator(inputs)
        bad = check_translator(words)
        n = sum(len(w) for w in words.values())
        print(f"translator (IEEE 0, f16 denormals kept): {n - len(bad)}/{n} words follow the rule (NaN and infinity "
              "inputs left out under clamp)")
        for x, mod, w, want in bad[:10]:
            print(f"  {x:04x} {mod or '(none)'}: {w:08x}, rule {want:08x}")
        failed |= bool(bad)
        if "hw" in sides:
            from hw_oracle import runner
            try:
                oracle = runner.open_oracle(arch=a.hw_arch)
            except runner.Unavailable as e:
                print(f"hardware oracle unavailable: {e}")
                return 77
            try:
                bad, n = run_hw(oracle)
            finally:
                oracle.close()
            print(f"hw (native, {oracle.arch}): {n - len(bad)}/{n} words follow the rule")
            for mode, mod, wave, x, w, want in bad[:10]:
                print(f"  ieee{mode['ieee']} denorm16={mode['denorm16_64']} wave{wave} {x:04x} {mod or '(none)'}: "
                      f"{w:08x}, rule {want:08x}")
            failed |= bool(bad)
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f"harness error: {e}")
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())

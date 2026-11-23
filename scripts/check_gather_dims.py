"""Opt-in synthetic GPU readback test for explicit-LOD array/cube gathers (#7097).

Uses the existing vk_texture_test harness and Python SPIR-V assembler. No game files,
LLVM or shader compiler needed. Optional --spirv-val validates every generated module.
This tests the translator's documented array/cube contract, not native Xbox semantics.
"""
import argparse
import itertools
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "semantic"))
import support
import spirv_asm


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def half_pair(a, b):
    return struct.unpack("<I", struct.pack("<ee", a, b))[0]


def shader(dim, layer, mip, a16, offset, channel):
    # All coordinates are exactly representable in f16/f32; sample away from seams.
    u, v = (1.375, 1.625) if dim == "CUBE" else (0.375, 0.625)
    values = [half_pair(u, v), half_pair(layer, mip)] if a16 else [bits(x) for x in (u, v, layer, mip)]
    if offset:
        values.insert(0, 1 | (63 << 8))  # signed six-bit x=+1, y=-1, z=0
    lines = ["s_load_dwordx8 s[8:15], s[0:1], 0x0", "s_waitcnt lgkmcnt(0)"]
    lines += [f"s_mov_b32 s{16+i}, 0x{value:x}" for i, value in enumerate((0x92, 0xc00000, 0x14500000, 0))]
    lines += [f"v_mov_b32_e32 v{4+i}, 0x{value:08x}" for i, value in enumerate(values)]
    op = "image_gather4_l_o" if offset else "image_gather4_l"
    lines += [f"{op} v[0:3], v[4:{3+len(values)}], s[8:15], s[16:19] dmask:0x{1<<channel:x} dim:SQ_RSRC_IMG_{dim}" + (" a16" if a16 else ""),
              "s_waitcnt vmcnt(0)", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    return lines


def expected(dim, layer, mip, offset, channel):
    # Contract: layer coordinates round to nearest-even then clamp. Cube coordinates
    # encode 8*slice+face, but the array has six faces per slice. Valid face IDs only.
    if dim == "CUBE":
        layer = 6 * (int(layer) // 8) + int(layer) % 8
    layer = min(11, max(0, round(layer)))
    side = 8 >> mip
    x = math.floor(0.375 * side - 0.5) + (1 if offset else 0)
    y = math.floor(0.625 * side - 0.5) - (1 if offset else 0)
    # Gather's four taps: lower-left, lower-right, upper-right, upper-left.
    taps = ((x, y+1), (x+1, y+1), (x+1, y), (x, y))
    result = []
    for tx, ty in taps:
        tx, ty = min(side-1, max(0, tx)), min(side-1, max(0, ty))
        rgba = (tx*32 + mip*7, ty*32 + mip*7, layer*20 + mip*7, 255)
        result.append(rgba[channel] / 255)
    return result


def check_pixels(raw, wanted):
    if len(wanted) != 4:
        raise AssertionError("expected values must specify all four gather taps")
    if len(raw) != 64*64*16:
        raise AssertionError(f"readback size {len(raw)}, expected {64*64*16}")
    for pixel, rgba in enumerate(struct.iter_unpack("<4f", raw)):
        for tap, (got, want) in enumerate(zip(rgba, wanted)):
            if not math.isfinite(got) or abs(got-want) > 1e-5:
                raise AssertionError(f"pixel {pixel}, tap {tap}: {got} != {want}")


def write_shader(path, name, lines, validator):
    assembly = support.translate_lines(name, lines)
    path.write_bytes(spirv_asm.assemble(assembly, target_env="vulkan1.2"))
    if validator:
        subprocess.run([validator, "--target-env", "vulkan1.2", str(path)], check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, type=Path, help="built vk_texture_test executable")
    parser.add_argument("--spirv-val", help="optional SPIRV-Tools validator executable")
    parser.add_argument("--limit", type=int, help="run only the first N cases while diagnosing; reported as partial")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    harness = str(args.harness.resolve(strict=True))
    cases = []
    for dim, layers in (("2D_ARRAY", (-1, 2.5, 3.5, 12)), ("CUBE", (0, 5, 8, 13))):
        cases.extend((dim, *values) for values in itertools.product(layers, (0, 1), (False, True), (False, True), (0, 1, 2)))
    if args.limit:
        cases = cases[:args.limit]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="leonida_gather_dims_") as temp:
        work = Path(temp)
        vs, ps, readback = work/"vs.spv", work/"ps.spv", work/"out.raw"
        lines = (ROOT/"runtime/test/shaders/vs_ids.asm").read_text(encoding="utf-8").splitlines()
        write_shader(vs, "vs_gather_dims", [line for line in lines if line and not line.startswith(";")], args.spirv_val)
        for case in cases:
            write_shader(ps, "ps_gather_dims", shader(*case), args.spirv_val)
            readback.unlink(missing_ok=True)  # stale output can never satisfy a failed run
            result = subprocess.run([harness, str(vs), str(ps), str(readback), "0", "92", "c00000", "14500000", "0", "--array-mips"],
                                    cwd=work, capture_output=True, text=True, timeout=30)
            if result.returncode == 77:
                if passed:
                    raise AssertionError(f"device became unavailable after {passed} passes: {result.stderr}")
                print(f"SKIP: {result.stderr.strip()}")
                return 77
            if result.returncode:
                raise AssertionError(f"{case}: harness exit {result.returncode}: {result.stderr}")
            if passed == 0:
                for line in (result.stdout + result.stderr).splitlines():
                    if line.startswith("device:"):
                        print(line, flush=True)
            dim, layer, mip, a16, offset, channel = case
            try:
                check_pixels(readback.read_bytes(), expected(dim, layer, mip, offset, channel))
            except AssertionError as error:
                raise AssertionError(f"{case}: {error}") from error
            passed += 1
            if passed % 24 == 0:
                print(f"checked {passed}/{len(cases)}", flush=True)
    print(f"PASS: {passed} draws, {passed*4096} pixels ({'partial' if args.limit else 'full matrix'}); "
          f"SPIR-V validation {'enabled' if args.spirv_val else 'not run'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, OSError, subprocess.SubprocessError, ValueError, support.rdna2spirv.Unsupported) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)

"""Verify image_sample_c_lz (depth-comparison sampling at LOD zero) in pixel_aot on D3D12.

Modes:
  (none)                                CPU controls only
  binary <llvm-mc> <llvm-objcopy> <lib> controls, then assemble each fixture and translate the decoded words
  draw <dxc> <offscreen_check> <adapter> controls, then draw every case and the negative controls
"""
from fractions import Fraction
from pathlib import Path
import math
import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pixel_aot_check import envelope, must_reject, write_record, lower as recorded_lower
import pixel_aot
from pixel_aot import disassemble, native_program

# The sampler fields below are written from the SQ_IMG_SAMP_WORD0..2 layout (Mesa gfx10-rsrc.json)
# independently of pixel_aot.comparison_sampler_state, so a wrong decoder bit or enum value changes the draw.
CLAMP_LAST_TEXEL = 2
TRUNC_COORD = 1 << 27  # word 0: set with point filtering, clear with bilinear (the pairs Mesa programs on gfx10)
COMPARE = {"NEVER": 0, "LESS": 1, "EQUAL": 2, "LESS_EQUAL": 3,
           "GREATER": 4, "NOT_EQUAL": 5, "GREATER_EQUAL": 6, "ALWAYS": 7}
XY_FILTER = {"point": 0, "bilinear": 1}
D3D12_FILTER = {"point": 0x80, "bilinear": 0x94}  # D3D12_FILTER_COMPARISON_MIN_MAG(_LINEAR)_MIP_POINT
TEXTURE = 8
TARGET = 16


def sampler_words(compare, filter_name, *, w0=0, w1=0, w2=0, w3=0, trunc_coord=None):
    xy = XY_FILTER[filter_name]
    if trunc_coord is None:
        trunc_coord = filter_name == "point"
    word0 = CLAMP_LAST_TEXEL | CLAMP_LAST_TEXEL << 3 | COMPARE[compare] << 12 | (TRUNC_COORD if trunc_coord else 0)
    word2 = xy << 20 | xy << 22
    return [word0 | w0, w1, word2 | w2, w3]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def depth(x, y):
    """Distinct (4k+1)/64 values, never equal to a pixel-centre u of the 16-pixel target, (2j+1)/32."""
    return f32(((x * 3 + y * 5) % 16 * 4 + 1) / 64)


DEPTHS = [depth(x, y) for y in range(TEXTURE) for x in range(TEXTURE)]
CONSTANT_REFERENCE = depth(5, 2)  # equal to some texels, so EQUAL and the LESS/LESS_EQUAL split are exercised


def passes(compare, reference, texel):
    return {"NEVER": False, "LESS": reference < texel, "EQUAL": reference == texel,
            "LESS_EQUAL": reference <= texel, "GREATER": reference > texel,
            "NOT_EQUAL": reference != texel, "GREATER_EQUAL": reference >= texel, "ALWAYS": True}[compare]


def endpoint_depth(value):
    """A second texture with texels at exactly 0 and 1, for references at the ends of [0, 1]."""
    return 0.0 if value < 0.25 else 1.0 if value > 0.75 else value


ENDPOINT_DEPTHS = [endpoint_depth(value) for value in DEPTHS]


def texel(x, y, depths=DEPTHS):
    x = min(max(x, 0), TEXTURE - 1)
    y = min(max(y, 0), TEXTURE - 1)
    return depths[y * TEXTURE + x]


def coordinates(px, py, offset):
    """Texel-space (u * 8, v * 8) of a pixel centre, after adding the same constant offset to u and v."""
    return (TEXTURE * (Fraction(2 * px + 1, 2 * TARGET) + offset),
            TEXTURE * (Fraction(2 * py + 1, 2 * TARGET) + offset))


def point_index(scaled, rule):
    """floor: D3D point addressing and TRUNC_COORD 1. round6: TRUNC_COORD 0 on gfx10, rounding to nearest even
    at six fraction bits before dropping the fraction. other-side: the texel below an exact edge."""
    if rule == "round6":
        return math.floor(Fraction(round(scaled * 64), 64))  # Python round() is round half to even
    if rule == "other-side" and scaled == math.floor(scaled):
        return math.floor(scaled) - 1
    return math.floor(scaled)


def expected_compare(compare, filter_name, reference, px, py, *, offset=0, depths=DEPTHS, rule="floor"):
    """D3D comparison filtering: compare each texel first, then filter the 0/1 results (PCF)."""
    su, sv = coordinates(px, py, offset)
    if filter_name == "point":
        return float(passes(compare, reference, texel(point_index(su, rule), point_index(sv, rule), depths)))
    x, y = su - Fraction(1, 2), sv - Fraction(1, 2)
    assert (x * 64).denominator == 1 and (y * 64).denominator == 1, "bilinear positions must be multiples of 1/64"
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    total = Fraction(0)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            total += wx * wy * passes(compare, reference, texel(x0 + dx, y0 + dy, depths))
    return float(total)  # multiples of 1/16 or 1/4 here: exact in binary32


def expected_filtered_depth(compare, filter_name, reference, px, py, *, offset=0, depths=DEPTHS, rule=None):
    """The wrong order for a negative control: filter the depths bilinearly, then compare once."""
    su, sv = coordinates(px, py, offset)
    x, y = su - Fraction(1, 2), sv - Fraction(1, 2)
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    filtered = sum(wx * wy * Fraction(texel(x0 + dx, y0 + dy, depths))
                   for dx, wx in ((0, 1 - fx), (1, fx)) for dy, wy in ((0, 1 - fy), (1, fy)))
    return float(passes(compare, Fraction(reference), filtered))


def prefix():
    return [
        "s_mov_b32 m0, s24",
        "v_interp_p1_f32_e32 v8, v0, attr0.x",
        "v_interp_p1_f32_e32 v9, v0, attr0.y",
        "v_interp_p2_f32_e32 v9, v1, attr0.y",
        "v_interp_p2_f32_e32 v8, v1, attr0.x",
        "s_mov_b32 s21, 0",
        "s_load_dwordx4 s[44:47], s[20:21], 0x40",
        "s_load_dwordx8 s[28:35], s[20:21], null",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[60:63], s[44:47], null",
        "s_waitcnt lgkmcnt(0)",
    ]


def sampler_moves(words):
    return [f"s_mov_b32 s{40 + k}, 0x{word:x}" for k, word in enumerate(words)]


def tail(result="v12", reference=None):
    copy = f"v_add_f32_e32 v13, 0, {reference}" if reference else "v_mov_b32_e32 v13, 0.5"
    return [
        "s_waitcnt vmcnt(0)",
        copy,
        "v_mov_b32_e32 v14, 0.5",
        "v_mov_b32_e32 v15, 1.0",
        f"exp mrt0 {result}, v13, v14, v15 done vm",
        "s_endpgm",
    ]


def constant_program(words):
    """Reference from constant word 0, contiguous address v[7:9] = (reference, u, v)."""
    return prefix() + sampler_moves(words) + [
        "v_mov_b32_e32 v7, s60",
        "image_sample_c_lz v12, v[7:9], s[28:35], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    ] + tail(reference="v7")


def attribute_program(words):
    """Reference interpolated from attr0.x (the u coordinate itself), contiguous v[7:9]."""
    program = prefix() + sampler_moves(words)
    program[1:1] = ["v_interp_p1_f32_e32 v7, v0, attr0.x", "v_interp_p2_f32_e32 v7, v1, attr0.x"]
    return program + [
        "image_sample_c_lz v12, v[7:9], s[28:35], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    ] + tail()


def alu_nsa_program(words):
    """Reference computed by ALU from constants (c0 * c1 + c2), in the NSA address form."""
    return prefix() + sampler_moves(words) + [
        "v_mov_b32_e32 v20, s62",
        "v_fma_f32 v20, s60, s61, v20",
        "image_sample_c_lz v12, [v20, v8, v9], s[28:35], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    ] + tail(reference="v20")


def edge_program(words):
    """Constant reference; u and v moved by constant word 3 in whole-quad mode, before the sample."""
    return ["s_mov_b64 s[50:51], exec", "s_wqm_b64 exec, exec"] + prefix() + [
        "v_add_f32_e32 v8, s63, v8",
        "v_add_f32_e32 v9, s63, v9",
        "s_and_b64 exec, exec, s[50:51]",
    ] + sampler_moves(words) + [
        "v_mov_b32_e32 v7, s60",
        "image_sample_c_lz v12, v[7:9], s[28:35], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    ] + tail(reference="v7")


# Constant words 0..2: the reference, and for the ALU program c0 * c1 + c2 = CONSTANT_REFERENCE exactly.
CONSTANTS = [CONSTANT_REFERENCE, 0.5, CONSTANT_REFERENCE / 2, 0.0]
assert f32(f32(CONSTANTS[0] * CONSTANTS[1]) + CONSTANTS[2]) == CONSTANT_REFERENCE

# Coordinate offsets for edge_program (constant word 3), all exact in binary32 and in the sums with uv.
# below-edge: u * 8 and v * 8 are 1/256 under an integer at every even pixel. Truncation (TRUNC_COORD 1, D3D)
#   takes the texel below; rounding at six fraction bits (TRUNC_COORD 0) takes the one above.
# on-edge: u * 8 and v * 8 are integers at even pixels and texel centres at odd pixels.
# centre: with bilinear filtering the same offset puts odd pixels on texel centres (one weight is 0)
#   and even pixels halfway between two centres.
BELOW_EDGE = Fraction(-65, 2048)
ON_EDGE = Fraction(-1, 32)

# Each case: program, filters, reference words, texture, coordinate offset, and whether the reference varies.
CASES = {
    "constant": (constant_program, ("point", "bilinear"), CONSTANTS, DEPTHS, 0),
    "attribute": (attribute_program, ("point", "bilinear"), CONSTANTS, DEPTHS, 0),
    "alu-nsa": (alu_nsa_program, ("point", "bilinear"), CONSTANTS, DEPTHS, 0),
    "below-edge": (edge_program, ("point",), CONSTANTS[:3] + [float(BELOW_EDGE)], DEPTHS, BELOW_EDGE),
    "on-edge": (edge_program, ("point",), CONSTANTS[:3] + [float(ON_EDGE)], DEPTHS, ON_EDGE),
    "centre": (edge_program, ("bilinear",), CONSTANTS[:3] + [float(ON_EDGE)], DEPTHS, ON_EDGE),
    "reference-0": (constant_program, ("point", "bilinear"), [0.0, 0.0, 0.0, 0.0], ENDPOINT_DEPTHS, 0),
    "reference-1": (constant_program, ("point", "bilinear"), [1.0, 0.0, 0.0, 0.0], ENDPOINT_DEPTHS, 0),
}
PROGRAMS = {kind: case[0] for kind, case in CASES.items()}


def reference_at(kind, px):
    if kind == "attribute":
        return f32(Fraction(2 * px + 1, 2 * TARGET))
    return CASES[kind][2][0]


def translate(program, words):
    return recorded_lower(program, 20, 24, words, constant_descriptor=64)


def expected_rows(kind, compare, filter_name, model=expected_compare, **options):
    _, _, _, depths, offset = CASES[kind]
    rows = []
    for py in range(TARGET):
        for px in range(TARGET):
            reference = reference_at(kind, px)
            green = 0.5 if kind == "attribute" else reference
            red = model(compare, filter_name, reference, px, py, offset=offset, depths=depths, **options)
            rows.append((red, green, 0.5, 1.0))
    return rows


def write_case(path, kind, d3d_filter, d3d_func, rows):
    _, _, constants, depths, _ = CASES[kind]
    data = struct.pack("<6I", d3d_filter, d3d_func, TEXTURE, TEXTURE, TARGET, TARGET)
    data += struct.pack("<64f", *(constants + [0.0] * (64 - len(constants))))
    data += struct.pack(f"<{len(depths)}f", *depths)
    for row in rows:
        data += struct.pack("<4f", *row)
    Path(path).write_bytes(data)


def controls():
    words = sampler_words("LESS", "point")
    # The decoder agrees with the independent encoder for every compare function and both filters.
    for compare, value in COMPARE.items():
        for filter_name in XY_FILTER:
            state = pixel_aot.comparison_sampler_state(sampler_words(compare, filter_name))
            assert state["compare"] == compare and state["d3d12_comparison_func"] == value + 1, state
            assert state["d3d12_filter"] == D3D12_FILTER[filter_name] and state["filter"] == filter_name, state
    for kind, program in PROGRAMS.items():
        source = translate(program(words), words)
        assert "SamplerComparisonState samp : register(s0);" in source and "Texture2D<float> tex" in source
        coordinate = "float2(alu" if program is edge_program else "float2(uv.x, uv.y), "
        assert source.count(".SampleCmpLevelZero(samp, " + coordinate) == 1, (kind, source)
    assert "SampleCmpLevelZero(samp, float2(uv.x, uv.y), uv.x)" in translate(attribute_program(words), words)
    # The depth neighbourhoods separate compare-then-filter from comparing a filtered depth.
    for kind in ("constant", "attribute"):
        pcf = expected_rows(kind, "LESS", "bilinear")
        filtered = expected_rows(kind, "LESS", "bilinear", expected_filtered_depth)
        assert sum(a != b for a, b in zip(pcf, filtered)) >= 16, kind
        assert any(row[0] not in (0.0, 1.0) for row in pcf), kind
    # Edge offsets are exact: the binary32 sum of each pixel-centre coordinate and the offset is the rational one.
    for offset in (BELOW_EDGE, ON_EDGE):
        for p in range(TARGET):
            centre = Fraction(2 * p + 1, 2 * TARGET)
            assert Fraction(f32(f32(centre) + f32(offset))) == centre + offset
    # Below an edge, truncation and rounding at six fraction bits select different texels, and the draws can
    # see it for every function that is not constant. On an edge, the texel below differs from D3D's floor.
    for compare in COMPARE:
        if compare in ("NEVER", "ALWAYS"):
            continue
        for kind, rule in (("below-edge", "round6"), ("on-edge", "other-side")):
            floor = expected_rows(kind, compare, "point")
            other = expected_rows(kind, compare, "point", rule=rule)
            assert sum(a != b for a, b in zip(floor, other)) >= 16, (kind, compare)
    assert expected_rows("centre", "LESS", "bilinear") != expected_rows("constant", "LESS", "bilinear")
    # References 0 and 1 meet texels equal to them, so the strict and inclusive functions split at the ends.
    for kind in ("reference-0", "reference-1"):
        assert CASES[kind][2][0] in ENDPOINT_DEPTHS
        for strict, inclusive in (("LESS", "LESS_EQUAL"), ("GREATER", "GREATER_EQUAL")):
            assert expected_rows(kind, strict, "point") != expected_rows(kind, inclusive, "point"), (kind, strict)
    # TRUNC_COORD: point sampling requires 1 and bilinear 0; the other two pairs are refused.
    for filter_name, trunc in (("point", False), ("bilinear", True)):
        bad = sampler_words("LESS", filter_name, trunc_coord=trunc)
        try:
            translate(constant_program(bad), bad)
        except ValueError as error:
            assert f"{filter_name} filtering requires TRUNC_COORD {int(not trunc)}" in str(error), error
        else:
            raise AssertionError(f"Accepted TRUNC_COORD {int(trunc)} with {filter_name} filtering")
        assert pixel_aot.comparison_sampler_state(sampler_words("LESS", filter_name))["trunc_coord"] == int(not trunc)
    # Descriptor fields outside the profile are refused, each with its own reason.
    for field, message in (
            (dict(w0=0x1), "CLAMP_LAST_TEXEL"),                  # X wrap (2 -> 3: mirror once)
            (dict(w0=0x4 << 3), "CLAMP_LAST_TEXEL"),             # Y clamp to border (2 -> 6)
            (dict(w0=1 << 9), "anisotropy"),
            (dict(w0=1 << 15), "unnormalized"),
            (dict(w0=1 << 19), "MC_COORD_TRUNC"),
            (dict(w0=1 << 20), "FORCE_DEGAMMA"),
            (dict(w0=1 << 29), "blend filter mode"),
            (dict(w0=2 << 29), "blend filter mode"),
            (dict(w1=1), "MIN_LOD"),
            (dict(w2=1), "LOD bias"),
            (dict(w2=1 << 14), "LOD bias"),
            (dict(w2=1 << 22), "one point or bilinear"),         # min bilinear, mag point
            (dict(w2=3 << 26), "mip filter"),
            (dict(w2=1 << 28), "word 2 bits 28-31"),
            (dict(w2=1 << 31), "word 2 bits 28-31"),
            (dict(w3=1 << 12), "word 3 bits 12-29"),
            (dict(w3=1 << 29), "UPGRADED_DEPTH")):
        bad = sampler_words("LESS", "point", **field)
        try:
            translate(constant_program(bad), bad)
        except ValueError as error:
            assert message in str(error), (field, error)
        else:
            raise AssertionError(f"Accepted sampler field {field}")
    anisotropic = sampler_words("LESS", "point", w2=2 << 20 | 2 << 22)
    must_reject(translate, constant_program(anisotropic), anisotropic)
    # Fields the 2D LOD-zero profile does not read are accepted: CLAMP_Z, aniso threshold/bias, MAX_LOD,
    # Z filter, mip filter, border colour.
    ignored = sampler_words("LESS", "point", w0=7 << 6 | 7 << 16 | 0x3f << 21,
                            w1=0xfff << 12, w2=2 << 24 | 2 << 26, w3=0xc0000fff)
    assert "D3D12_COMPARISON_FUNC 2 (LESS)" in translate(constant_program(ignored), ignored)
    program = constant_program(words)
    sample = next(i for i, line in enumerate(program) if line.startswith("image_sample_c_lz"))
    for old, new in (("dmask:0x1", "dmask:0x3"), ("v[7:9]", "v[7:8]"), ("v[7:9]", "v[7:10]"),
                     ("SQ_RSRC_IMG_2D", "SQ_RSRC_IMG_2D_ARRAY"), ("image_sample_c_lz", "image_sample_c")):
        changed = list(program)
        changed[sample] = changed[sample].replace(old, new)
        must_reject(translate, changed, words)
    # VGPRs end at v255: the destination, a range and an NSA operand past it are refused.
    nsa = alu_nsa_program(words)
    nsa_sample = next(i for i, line in enumerate(nsa) if line.startswith("image_sample_c_lz"))
    for source, index, old, new in ((program, sample, "image_sample_c_lz v12,", "image_sample_c_lz v256,"),
                                    (program, sample, "v[7:9]", "v[254:256]"),
                                    (nsa, nsa_sample, "[v20, v8, v9]", "[v20, v8, v256]")):
        changed = list(source)
        changed[index] = changed[index].replace(old, new)
        assert changed[index] != source[index]
        try:
            translate(changed, words)
        except ValueError as error:
            assert "VGPR out of range" in str(error), (new, error)
        else:
            raise AssertionError(f"Accepted {new}")
    changed = list(program)
    changed[sample] = changed[sample].replace("image_sample_c_lz v12,", "image_sample_c_lz v255,")
    changed[changed.index("exp mrt0 v12, v13, v14, v15 done vm")] = "exp mrt0 v255, v13, v14, v15 done vm"
    assert "SampleCmpLevelZero" in translate(changed, words)
    # A reference read from a further constant buffer (b1, the descriptor at 0x60 of the texture table).
    from_b1 = prefix() + sampler_moves(words) + [
        "s_load_dwordx4 s[48:51], s[20:21], 0x60",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dword s64, s[48:51], 0x4",
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v7, s64",
        "image_sample_c_lz v12, v[7:9], s[28:35], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    ] + tail(reference="v7")
    source = recorded_lower(from_b1, 20, 24, words, 64, extra_constants=[(20, 0x60)])
    assert "cbuffer Constants1 : register(b1) { float4 params1[16]; };" in source, source
    reference = next(line.split()[2] for line in source.splitlines() if line.endswith(" = params1[0].y;"))
    assert f"tex.SampleCmpLevelZero(samp, float2(uv.x, uv.y), {reference});" in source, source
    must_reject(translate, from_b1, words)
    # The sampler words must still match the caller binding.
    must_reject(translate, program, sampler_words("GREATER", "point"))
    # Reading the result before the wait, or a reference that is an outstanding sample result.
    early = list(program)
    early.remove("s_waitcnt vmcnt(0)")
    must_reject(translate, early, words)
    chained = list(program)
    chained[sample + 1:sample + 1] = [
        "image_sample_c_lz v7, v[7:9], s[28:35], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D",
        "image_sample_c_lz v12, v[7:9], s[28:35], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    ]
    must_reject(translate, chained, words)
    # An ordinary sample in the same program would need a second sampler at s0.
    mixed = list(program)
    mixed[sample:sample] = ["image_sample v[16:19], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D"]
    must_reject(translate, mixed, words)
    mixed = list(program)
    mixed[sample + 1:sample + 1] = [
        "s_waitcnt vmcnt(0)",
        "image_sample v[16:19], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D"]
    must_reject(translate, mixed, words)
    must_reject(pixel_aot.lower, program, 20, 24, words, 64, second_texture_offset=32)
    must_reject(pixel_aot.lower, program, 20, 24, None, 64)
    # The ordinary sampling paths keep their resources.
    plain = pixel_aot.lower(["s_mov_b32 m0, s24"] + prefix()[1:5] + [
        "s_mov_b32 s21, 0", "s_load_dwordx8 s[28:35], s[20:21], null",
        *sampler_moves([1, 2, 3, 4]), "s_waitcnt lgkmcnt(0)",
        "image_sample_lz v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)", "exp mrt0 v12, v13, v14, v15 done vm", "s_endpgm"], 20, 24, [1, 2, 3, 4])
    assert "SamplerState samp : register(s0);" in plain and "SamplerComparison" not in plain


def compile_shader(compiler, source, shader):
    subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True,
                   stdout=subprocess.DEVNULL)


def run_case(renderer, adapter, shader, case):
    result = subprocess.run([renderer, adapter, "--check-pixel-compare", shader, case], capture_output=True, text=True)
    if result.returncode == 77:
        return 77
    return result


def draw(directory, compiler, renderer, adapter):
    passed = controls_failed = 0
    for kind, (build, filters, *_) in CASES.items():
        for filter_name in filters:
            for compare in COMPARE:
                words = sampler_words(compare, filter_name)
                state = pixel_aot.comparison_sampler_state(words)
                source = directory / f"{kind}-{filter_name}-{compare}.hlsl"
                shader = source.with_suffix(".cso")
                case = source.with_suffix(".case")
                source.write_text(translate(build(words), words), encoding="ascii")
                compile_shader(compiler, source, shader)
                write_case(case, kind, state["d3d12_filter"], state["d3d12_comparison_func"],
                           expected_rows(kind, compare, filter_name))
                result = run_case(renderer, adapter, shader, case)
                if result == 77:
                    return 77
                summary = [l for l in result.stdout.splitlines() if "mismatches=" in l or "debug_errors=" in l]
                print(f"case={kind}-{filter_name}-{compare}", *summary)
                debug_clean = ("debug_layer=enabled" not in result.stdout
                               or "debug_errors=0 debug_warnings=0" in result.stdout)
                if result.returncode != 0 or not debug_clean:
                    sys.stderr.write(result.stdout + result.stderr)
                    raise AssertionError(f"Comparison draw failed: {kind} {filter_name} {compare}")
                passed += 1

    def expect_mismatch(kind, label, source_text, d3d_filter, d3d_func, rows):
        nonlocal controls_failed
        label = f"{kind}-{label}"
        source = directory / "control.hlsl"
        shader = directory / "control.cso"
        case = directory / "control.case"
        source.write_text(source_text, encoding="ascii")
        compile_shader(compiler, source, shader)
        write_case(case, kind, d3d_filter, d3d_func, rows)
        result = run_case(renderer, adapter, shader, case)
        if result == 77 or result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"Negative control escaped detection: {label}")
        controls_failed += 1
        print(f"control={label} status=detected {result.stderr.strip().splitlines()[0]}")

    # The descriptor says LESS; a decoder that read the field from the wrong bits or reversed the operands
    # would produce GREATER/GREATER_EQUAL-like images. The reference for those must not match.
    for kind in ("constant", "attribute"):
        words = sampler_words("LESS", "bilinear")
        state = pixel_aot.comparison_sampler_state(words)
        source = translate(PROGRAMS[kind](words), words)
        # LESS_EQUAL differs from LESS only where the reference equals a texel: the constant case only.
        for wrong in ("GREATER", "GREATER_EQUAL") + (("LESS_EQUAL",) if kind == "constant" else ()):
            expect_mismatch(kind, f"LESS-expected-{wrong}", source, state["d3d12_filter"],
                            state["d3d12_comparison_func"], expected_rows(kind, wrong, "bilinear"))
        # Filtering the depths first and comparing once is not what the sampler does.
        expect_mismatch(kind, "filter-then-compare", source, state["d3d12_filter"],
                        state["d3d12_comparison_func"],
                        expected_rows(kind, "LESS", "bilinear", expected_filtered_depth))
        # Point and bilinear filtering exchanged in the host sampler.
        expect_mismatch(kind, "filter-point", source, D3D12_FILTER["point"],
                        state["d3d12_comparison_func"], expected_rows(kind, "LESS", "bilinear"))
        # Reference ignored, and u/v exchanged, in recompiled HLSL.
        reference = "uv.x" if kind == "attribute" else "alu"
        line = next(l for l in source.splitlines() if "SampleCmpLevelZero" in l)
        argument = line.rsplit(", ", 1)[1].rstrip(");")
        assert argument.startswith(reference), line
        expect_mismatch(kind, "reference-ignored", source.replace(line, line.replace(f", {argument});", ", 0.5f);")),
                        state["d3d12_filter"], state["d3d12_comparison_func"], expected_rows(kind, "LESS", "bilinear"))
        expect_mismatch(kind, "uv-swapped", source.replace("float2(uv.x, uv.y)", "float2(uv.y, uv.x)"),
                        state["d3d12_filter"], state["d3d12_comparison_func"], expected_rows(kind, "LESS", "bilinear"))
    # Texel selection at and just below an edge: the images for TRUNC_COORD 0 (rounding at six fraction bits)
    # and for the texel on the other side of an exact edge must not match.
    for kind, rule in (("below-edge", "round6"), ("on-edge", "other-side")):
        words = sampler_words("LESS", "point")
        state = pixel_aot.comparison_sampler_state(words)
        expect_mismatch(kind, f"expected-{rule}", translate(edge_program(words), words), state["d3d12_filter"],
                        state["d3d12_comparison_func"], expected_rows(kind, "LESS", "point", rule=rule))
    # Bilinear on texel centres: the image without the coordinate offset must not match.
    words = sampler_words("LESS", "bilinear")
    state = pixel_aot.comparison_sampler_state(words)
    expect_mismatch("centre", "expected-without-offset", translate(edge_program(words), words), state["d3d12_filter"],
                    state["d3d12_comparison_func"], expected_rows("constant", "LESS", "bilinear"))
    # References at 0 and 1 against texels equal to them: the strict function is not the inclusive one.
    for kind, strict, inclusive in (("reference-0", "LESS", "LESS_EQUAL"), ("reference-1", "GREATER", "GREATER_EQUAL")):
        words = sampler_words(strict, "point")
        state = pixel_aot.comparison_sampler_state(words)
        expect_mismatch(kind, f"{strict}-expected-{inclusive}", translate(constant_program(words), words),
                        state["d3d12_filter"], state["d3d12_comparison_func"],
                        expected_rows(kind, inclusive, "point"))
    print(f"draws={passed} negative_controls_detected={controls_failed}")
    return 0


def check_binary(directory, assembler, objcopy, library):
    for compare in ("LESS", "GREATER_EQUAL"):
        for build in dict.fromkeys(PROGRAMS.values()):
            words = sampler_words(compare, "bilinear")
            instructions = build(words)
            source = directory / "fixture.s"
            obj = directory / "fixture.o"
            binary = directory / "fixture.bin"
            source.write_text("\n".join(instructions) + "\n", encoding="ascii")
            subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", source, "-o", obj],
                           check=True)
            subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, binary], check=True)
            archive = directory / "fixture.rpf"
            archive.write_bytes(envelope(binary.read_bytes()))
            _, native = native_program(archive, "fixture")
            decoded = disassemble(native, library)
            sample = next(line for line in decoded if line.startswith("image_sample_c_lz"))
            print(sample)
            assert translate(decoded, words) == translate(instructions, words)


def main():
    controls()
    if len(sys.argv) == 1:
        print("pixel_compare controls passed")
        return 0
    mode, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "binary":
            check_binary(Path(tmp), *args)
            return 0
        if mode == "draw":
            return draw(Path(tmp), *args)
    raise ValueError("Expected no mode, binary or draw")


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        write_record(os.environ.get("FORM_COVERAGE_RECORD"))

"""Two descriptor tables, depth export, and a physical CBV swap."""

from pathlib import Path
import math
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from depth_mask_aot import lower

VS = """
float4 main(uint id : SV_VertexID) : SV_Position {
    return float4(float2((id << 1) & 2, id & 2) * float2(2,-2) + float2(-1,1), 0.75, 1);
}
"""


# One binary32 VOP2 form per fixture: v6 = op(0.5, v7) feeds the discard compare.
# Each tuple is the opcode, the CPU reference and the exact HLSL line it must lower to,
# with {a} the result register of v_fract and {n} the instruction's own index.
OPS = {
    "v_add_f32_e32": (lambda c, v: c + v, "precise float value{n} = (0.5f + value{a});"),
    "v_sub_f32_e32": (lambda c, v: c - v, "precise float value{n} = (0.5f - value{a});"),
    "v_subrev_f32_e32": (lambda c, v: v - c, "precise float value{n} = (value{a} - 0.5f);"),
    "v_mul_f32_e32": (lambda c, v: c * v, "precise float value{n} = (0.5f * value{a});"),
    "v_min_f32_e32": (lambda c, v: min(c, v), "precise float value{n} = min(0.5f, value{a});"),
    "v_max_f32_e32": (lambda c, v: max(c, v), "precise float value{n} = max(0.5f, value{a});"),
}


# One ordered binary32 compare per fixture: it discards the lanes where threshold OP v7
# holds, v7 being the fractional position. The CPU reference is written with Python
# operators, the HLSL line is pinned literally, and `{a}` is the register v_fract wrote.
CMPX = {
    "v_cmpx_lt_f32_e32": (lambda t, v: t < v, "if (asfloat(second[0].x) < value{a}) discard;"),
    "v_cmpx_le_f32_e32": (lambda t, v: t <= v, "if (asfloat(second[0].x) <= value{a}) discard;"),
    "v_cmpx_gt_f32_e32": (lambda t, v: t > v, "if (asfloat(second[0].x) > value{a}) discard;"),
    "v_cmpx_ge_f32_e32": (lambda t, v: t >= v, "if (asfloat(second[0].x) >= value{a}) discard;"),
}
# Odd multiples of 1/64 are exactly the values v7 takes, so each threshold lands on
# one row's value: that row tells lt from le and gt from ge.
CMPX_CASES = ((7, 17 / 64), (29, 49 / 64))


def program(extended=False, op=None, cmpx=None):
    # Any named form, including the lt compare, runs the extended (Y-sampling) fixture.
    extended = extended or op is not None or cmpx is not None
    cmpx = cmpx or "v_cmpx_lt_f32_e32"
    # Original fixture. The offsets here are instruction addresses in a decoded stream.
    prefix = [
        "s_mov_b32 s7, 0",
        "s_mov_b32 s11, 0",
        "s_mov_b64 s[22:23], exec",
        "s_mov_b64 s[24:25], exec",
        "s_load_dwordx4 s[12:15], s[6:7], 0x20",
        "s_load_dwordx4 s[16:19], s[10:11], 0x10",
        "s_waitcnt lgkmcnt(0)",
        f"s_buffer_load_dword s30, s[12:15], {0x18 if extended else 0x4:#x}",
        "s_buffer_load_dword s31, s[16:19], null",
        "s_waitcnt lgkmcnt(0)",
    ]
    if extended:
        prefix += ["s_add_i32 s30, s30, 12", "s_and_b32 s30, s30, 31"]
    # The extended fixture samples Y in a 32-pixel period.
    position, scale = ("v3", "0x3d000000") if extended else ("v2", "0x3e000000")
    prefix += [
        "v_cvt_f32_ubyte0_e32 v5, s30",
        f"v_mul_legacy_f32_e32 v7, {scale}, {position}",
        f"v_fmac_f32_e32 v7, {scale}, v5",
        "v_fract_f32_e32 v7, v7",
    ]
    if op:
        prefix.append(f"{op} v6, 0.5, v7")
    exits = [
        f"{cmpx} s31, {'v6' if op else 'v7'}",
        "s_andn2_b64 s[22:23], s[22:23], exec",
        "s_cbranch_scc0 4",
        "s_and_b64 exec, s[24:25], s[22:23]",
        "exp mrtz, v4, off, off, off done vm",
        "s_endpgm",
        "s_mov_b64 exec, s[22:23]",
        "exp null, off, off, off, off done vm",
        "s_endpgm",
    ]
    return prefix + exits


TABLES = [(6, 32), (10, 16)]


def decoded_stream(lines):
    sizes = [
        8 if line.startswith(("s_load_", "s_buffer_load_", "exp ")) else 4
        for line in lines
    ]
    return [(sum(sizes[:i]), line) for i, line in enumerate(lines)]


def op_source(op):
    """Lower one binary32 form and pin the line it becomes, then change that one word."""
    lines = program(op=op)
    word = lines.index(f"{op} v6, 0.5, v7")

    def hlsl(name):
        return "    " + OPS[name][1].format(n=word, a=word - 1)

    def lowered(replacement):
        changed = decoded_stream(lines)
        changed[word] = (changed[word][0], replacement)
        return lower(changed, TABLES)

    text = source(op=op)
    assert hlsl(op) in text.splitlines()
    for other in OPS:
        if other != op:
            wrong = lowered(f"{other} v6, 0.5, v7")
            assert hlsl(other) in wrong.splitlines() and hlsl(op) not in wrong, other
    for bad in (
        f"{op} v6, 0.5, v9",  # vsrc1 was never written
        f"{op} v6, 0.5, s30",  # vsrc1 is a VGPR in every e32 form
        f"{op} v6, 0.5",  # missing operand
        op.replace("_e32", "_e64") + " v6, 0.5, v7",  # other encoding
    ):
        try:
            lowered(bad)
        except ValueError:
            continue
        raise AssertionError(f"Invalid binary32 form accepted: {bad}")
    return text


def cmpx_source(name):
    """Lower one compare, pin its line, then change that one instruction word."""
    lines = program(cmpx=name)
    word = lines.index(f"{name} s31, v7")

    def hlsl(form):
        return "    " + CMPX[form][1].format(a=word - 1)

    def lowered(replacement):
        changed = decoded_stream(lines)
        changed[word] = (changed[word][0], replacement)
        return lower(changed, TABLES)

    text = source(cmpx=name)
    assert hlsl(name) in text.splitlines()
    for other in CMPX:
        if other != name:
            wrong = lowered(f"{other} s31, v7")
            assert hlsl(other) in wrong.splitlines() and hlsl(name) not in wrong, other
    for bad in (
        name.replace("_lt_", "_nlt_").replace("_le_", "_nle_").replace("_gt_", "_ngt_").replace("_ge_", "_nge_") + " s31, v7",  # true for NaN
        name.replace("_f32_", "_f16_") + " s31, v7",  # other width
        name.replace("_f32_", "_i32_") + " s31, v7",  # integer compare
        name.replace("_e32", "_e64") + " s31, v7",  # other encoding
        f"{name} s31, v9",  # vsrc1 was never written
        f"{name} s31, s30",  # vsrc1 is a VGPR in every e32 form
        f"{name} s31",  # missing operand
    ):
        try:
            lowered(bad)
        except ValueError:
            continue
        raise AssertionError(f"Invalid compare accepted: {bad}")
    return text


def source(extended=False, op=None, cmpx=None):
    extended = extended or op is not None or cmpx is not None
    lines = program(extended, op, cmpx)
    decoded = decoded_stream(lines)
    offsets = [pc for pc, _ in decoded]
    tables = TABLES
    for bad in ([(6, 16), (10, 32)], [(10, 32), (6, 16)]):
        try:
            lower(decoded, bad)
        except ValueError:
            continue
        raise AssertionError("Unbound descriptor accepted")
    for original, replacement in (
        ("s_waitcnt lgkmcnt(0)", "s_nop 0"),
        ("s_cbranch_scc0 4", "s_cbranch_scc0 3"),
    ):
        changed = [
            (pc, replacement if line == original else line) for pc, line in decoded
        ]
        try:
            lower(changed, tables)
        except ValueError:
            continue
        raise AssertionError("Unverified mask or pending load accepted")
    # These edits keep the decoded instruction addresses and widths intact.
    for case, edits in (
        ("partial wait", (("s_waitcnt lgkmcnt(0)", "s_waitcnt lgkmcnt(1)"),)),
        (
            "restore mask operand",
            (("s_and_b64 exec, s[24:25], s[22:23]", "s_and_b64 exec, s[24:25], s[24:25]"),),
        ),
        (
            "restore mask alias",
            (("s_and_b64 exec, s[24:25], s[22:23]", "s_and_b64 exec, s[22:23], s[22:23]"),),
        ),
        (
            "constant buffer end",
            ((
                f"s_buffer_load_dword s30, s[12:15], {0x18 if extended else 0x4:#x}",
                "s_buffer_load_dword s30, s[12:15], 0x800",
            ),),
        ),
        (
            "descriptor register alignment",
            (
                ("s_load_dwordx4 s[16:19], s[10:11], 0x10", "s_load_dwordx4 s[18:21], s[10:11], 0x10"),
                ("s_buffer_load_dword s31, s[16:19], null", "s_buffer_load_dword s31, s[18:21], null"),
            ),
        ),
        (
            "null export valid mask",
            (("exp null, off, off, off, off done vm", "exp null, off, off, off, off done"),),
        ),
    ):
        changed = decoded.copy()
        for original, replacement in edits:
            index = lines.index(original)
            changed[index] = (offsets[index], replacement)
        try:
            lower(changed, tables)
        except ValueError:
            continue
        raise AssertionError(f"Invalid depth program accepted: {case}")
    return lower(decoded, tables)


def compile_pair(dxc, tmp, shader, stem="pixel"):
    vs, ps = tmp / "vertex.dxil", tmp / (stem + ".dxil")
    for name, profile, text, binary in [("v", "vs", VS, vs), (stem, "ps", shader, ps)]:
        path = tmp / (name + ".hlsl")
        path.write_text(text)
        subprocess.run(
            [dxc, "-T", profile + "_6_0", "-E", "main", "-Fo", binary, path], check=True
        )
    return vs, ps


def op_picture(op, seed, threshold):
    """Independent CPU reference: the 32x32 depth the lowered program must leave."""
    function = OPS[op][0]
    values = bytearray()
    for row in range(32):
        # The fixture's v7 is the fractional part of (row + 0.5 + (seed + 12) mod 32) / 32.
        v7 = ((row + 0.5 + ((seed + 12) & 31)) % 32) / 32
        depth = 0.75 if function(0.5, v7) <= threshold else 0.9375
        values += struct.pack("<f", depth) * 32
    return bytes(values)


def cmpx_picture(name, seed, threshold):
    """Independent CPU reference for the compare fixtures: survivors keep depth 0.75."""
    function = CMPX[name][0]
    values = bytearray()
    for row in range(32):
        v7 = ((row + 0.5 + ((seed + 12) & 31)) % 32) / 32
        depth = 0.9375 if function(threshold, v7) else 0.75
        values += struct.pack("<f", depth) * 32
    return bytes(values)


def run_depth_mask(renderer, adapter, tmp, vs, ps, seed, threshold, reference):
    a, b = bytearray(2048), bytearray(2048)
    struct.pack_into("<I", a, 0x18, seed)
    struct.pack_into("<f", b, 0, threshold)
    fixture = tmp / "draw.bin"
    fixture.write_bytes(a + b + reference)
    result = subprocess.run(
        [renderer, adapter, "--depth-mask", vs, ps, fixture], capture_output=True, text=True
    )
    print(result.stdout, result.stderr)
    return result


def draw_ops(dxc, renderer, adapter, tmp):
    """Each binary32 form against its CPU reference, then against every other form's."""
    cases = ((7, 0.25), (29, 0.75))
    return draw_forms(dxc, renderer, adapter, tmp, OPS, op_picture, op_source, cases)


def draw_cmpx(dxc, renderer, adapter, tmp):
    """Each ordered compare against its CPU reference, then against every other's."""
    return draw_forms(
        dxc, renderer, adapter, tmp, CMPX, cmpx_picture, cmpx_source, CMPX_CASES
    )


def draw_forms(dxc, renderer, adapter, tmp, forms, picture, lowered, cases):
    pictures = {op: [picture(op, *case) for case in cases] for op in forms}
    # Every form must be visible: no other form may leave the same pictures.
    for op, shown in pictures.items():
        assert all(shown != pictures[other] for other in forms if other != op), op
    for op in forms:
        vs, ps = compile_pair(dxc, tmp, lowered(op))
        for (seed, threshold), reference in zip(cases, pictures[op]):
            result = run_depth_mask(renderer, adapter, tmp, vs, ps, seed, threshold, reference)
            if result.returncode == 77:
                return 77
            result.check_returncode()
        # A reference from another form must fail with exactly the pixels where they differ.
        for other in forms:
            if other == op:
                continue
            for (seed, threshold), right, wrong in zip(
                cases, pictures[op], pictures[other]
            ):
                if right == wrong:
                    continue
                count = sum(
                    right[i : i + 4] != wrong[i : i + 4] for i in range(0, 4096, 4)
                )
                result = run_depth_mask(renderer, adapter, tmp, vs, ps, seed, threshold, wrong)
                if (
                    result.returncode != 1
                    or "Draw reference mismatch" not in result.stderr
                    or f"pixels=1024 mismatches={count}\n" not in result.stdout
                    or "debug_errors=0" not in result.stdout
                    and "debug_layer=unavailable" not in result.stdout
                ):
                    raise AssertionError(f"{op} must not match the {other} reference")
                break
    return 0


def draw(dxc, renderer, adapter, tmp, shader, word_offset, noise, *, thresholds=()):
    vs, ps = compile_pair(dxc, tmp, shader)
    a, b = bytearray(2048), bytearray(2048)
    for buf, threshold, seed in ((a, 0.25, 7), (b, 0.625, 29)):
        struct.pack_into("<f", buf, 0, threshold)
        struct.pack_into("<I", buf, word_offset, seed)

    def picture(seed, threshold):
        return b"".join(
            struct.pack(
                "<f", 0.75 if noise(x + 0.5, y + 0.5, seed) <= threshold else 0.9375
            )
            for y in range(32)
            for x in range(32)
        )

    expected = picture(7, 0.625)
    exchanged = picture(29, 0.25)
    assert (
        expected != exchanged
        and picture(29, 0.625) != expected
        and picture(7, 0.25) != expected
    )
    for swapped, seed, threshold, reference, negative in (
        (False, 7, 0.625, expected, False),
        (False, 29, 0.625, picture(29, 0.625), False),
        (False, 7, 0.25, picture(7, 0.25), False),
        (True, 7, 0.625, expected, True),
        (True, 7, 0.625, exchanged, False),
        *((False, 7, threshold, picture(7, threshold), False) for threshold in thresholds),
    ):
        struct.pack_into("<I", a, word_offset, seed)
        struct.pack_into("<f", b, 0, threshold)
        fixture = tmp / "draw.bin"
        fixture.write_bytes(a + b + reference)
        mode = "--depth-mask-swapped" if swapped else "--depth-mask"
        result = subprocess.run(
            [renderer, adapter, mode, vs, ps, fixture], capture_output=True, text=True
        )
        print(result.stdout, result.stderr)
        if result.returncode == 77 and not swapped:
            return 77
        if negative:
            if (
                result.returncode != 1
                or "Draw reference mismatch" not in result.stderr
                or "debug_errors=0" not in result.stdout
                and "debug_layer=unavailable" not in result.stdout
            ):
                raise AssertionError(
                    "Swapped descriptors must fail at depth comparison"
                )
        else:
            result.check_returncode()
    return 0


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fract_reference(bits):
    """v_fract_f32 on binary32 bits: x - floor(x) capped at 0x3f7fffff, a NaN stays a NaN."""
    x = from_bits(bits)
    if not math.isfinite(x):
        return math.nan
    return min(f32(x - math.floor(x)), from_bits(0x3F7FFFFF))


def evaluate_fract(expression, bits):
    """Evaluate one emitted v_fract expression on binary32 bits with D3D semantics."""
    x = from_bits(bits)
    frac = f32(x - math.floor(x)) if math.isfinite(x) else math.nan
    cap = from_bits(0x3F7FFFFF)
    if re.fullmatch(r"\(frac\((\w+)\) >= 1\.0f \? asfloat\(0x3f7fffffu\) : frac\(\1\)\)", expression):
        return cap if frac >= 1.0 else frac  # NaN fails the comparison and passes through
    if re.fullmatch(r"min\(frac\((\w+)\), asfloat\(0x3f7fffffu\)\)", expression):
        return cap if frac != frac else min(frac, cap)  # D3D min returns the non-NaN operand
    if re.fullmatch(r"frac\((\w+)\)", expression):
        return frac
    raise AssertionError(expression)


def same_word(got, want):
    return (got != got and want != want) or struct.pack("<f", got) == struct.pack("<f", want)


def fract_nan():
    """The v_fract_f32 cap keeps a NaN a NaN and caps -2^-30, and each wrong cap is caught."""
    lines = program()
    index = lines.index("v_fract_f32_e32 v7, v7")
    prefix = f"    precise float value{index} = "
    line = next(l for l in lower(decoded_stream(lines), TABLES).splitlines() if l.startswith(prefix))
    expression = line[len(prefix) : -1]
    argument = re.search(r"frac\((\w+)\)", expression)[1]
    assert expression == f"(frac({argument}) >= 1.0f ? asfloat(0x3f7fffffu) : frac({argument}))", line
    nan, tiny = (0x7FC00000, 0x7FC00001, 0xFF800000), (0xB0800000, 0xB1800000)
    ordinary = (0x3F000000, 0xBF000000, 0x00000000, 0x80000000, 0x3F800000, 0xBF800000)
    for bits in nan + tiny + ordinary:
        assert same_word(evaluate_fract(expression, bits), fract_reference(bits)), hex(bits)
    # The reference itself: a NaN stays a NaN and -2^-30 gives 0x3f7fffff, not 1.0.
    assert all(fract_reference(b) != fract_reference(b) for b in nan)
    assert all(fract_reference(b) == from_bits(0x3F7FFFFF) for b in tiny)
    # Previous min cap: a NaN input becomes finite. No cap: a tiny negative input becomes 1.0.
    old = f"min(frac({argument}), asfloat(0x3f7fffffu))"
    bare = f"frac({argument})"
    for bits in nan:
        assert not same_word(evaluate_fract(old, bits), fract_reference(bits)), hex(bits)
    for bits in tiny:
        assert not same_word(evaluate_fract(bare, bits), fract_reference(bits)), hex(bits)
    assert all(same_word(evaluate_fract(old, b), fract_reference(b)) for b in tiny + ordinary)


def binary(tmp, assembler, objcopy, library):
    fract_nan()
    translate(tmp, assembler, objcopy, library, program(), source())
    for op in OPS:
        translate(tmp, assembler, objcopy, library, program(op=op), op_source(op))
    for name in CMPX:
        translate(tmp, assembler, objcopy, library, program(cmpx=name), cmpx_source(name))


def translate(tmp, assembler, objcopy, library, lines, expected):
    from pixel_aot_check import envelope
    from pixel_aot import _read_named_container

    src, obj, raw = (tmp / name for name in ("fixture.s", "fixture.o", "fixture.bin"))
    src.write_text(
        "\n".join(
            s.replace("mrtz,", "mrtz")
            .replace("null,", "null")
            .replace("prim,", "prim")
            .replace("pos0,", "pos0")
            for s in lines
        )
        + "\n"
    )
    subprocess.run(
        [
            assembler,
            "-triple=amdgcn--amdpal",
            "-mcpu=gfx1013",
            "-filetype=obj",
            src,
            "-o",
            obj,
        ],
        check=True,
    )
    subprocess.run(
        [objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True
    )
    code = raw.read_bytes()
    padded = code + bytes.fromhex("00009fbf") * (4 - (len(code) // 4 + 10) % 4)
    record = envelope(padded, directory_end=(40 + len(padded), 288 + len(padded)))
    container = tmp / "input.dxbc"
    container.write_bytes(_read_named_container(record, len(b"fixture") + 2))
    translated = tmp / "out.hlsl"
    tables = ["--table", "6", "32", "--table", "10", "16"]
    tool = Path(__file__).resolve().parents[1] / "tools" / "depth_mask_aot.py"
    subprocess.run(
        [
            sys.executable,
            "-B",
            tool,
            container,
            "--llvm",
            library,
            "--output",
            translated,
            *tables,
        ],
        check=True,
    )
    assert translated.read_text() == expected


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        if sys.argv[1] == "binary":
            binary(Path(directory), *sys.argv[2:])
            sys.exit(0)
        if sys.argv[1] == "ops":
            sys.exit(draw_ops(*sys.argv[2:], Path(directory)))
        if sys.argv[1] == "cmpx":
            sys.exit(draw_cmpx(*sys.argv[2:], Path(directory)))
        draw_args = (*sys.argv[1:], Path(directory))
        result = draw(*draw_args, source(), 4, lambda x, y, seed: ((x + seed) / 8) % 1)
        if result:
            sys.exit(result)
        sys.exit(
            draw(
                *draw_args,
                source(extended=True),
                0x18,
                lambda x, y, seed: ((y + seed + 12) % 32) / 32,
                thresholds=(62 / 64, 63 / 64, 0.0),
            )
        )

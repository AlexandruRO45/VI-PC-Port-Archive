"""A source-built indexed texture selection and coverage-only pixel export."""

import json
import math
import operator
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from pixel_aot_check import envelope, must_reject
from pixel_aot import _read_named_container
from sampled_coverage_aot import lower

# Original fixture with four independent descriptor tables. The two buffered
# floats are mapped to different slices of a single desktop upload.
PROGRAM = """
s_mov_b32 s11, 0
s_mov_b32 s21, 0
s_mov_b32 s31, 0
s_mov_b32 s41, 0
s_load_dwordx4 s[60:63], s[30:31], 0x10
s_load_dwordx4 s[64:67], s[20:21], 0x40
s_load_dwordx4 s[68:71], s[20:21], 0x70
s_load_dwordx4 s[72:75], s[10:11], null
s_mov_b64 s[48:49], exec
s_wqm_b64 exec, exec
s_mov_b64 s[52:53], exec
s_mov_b32 m0, s33
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s6, s[72:75], s7
s_buffer_load_dword s9, s[68:71], 0x8
s_buffer_load_dword s3, s[64:67], 0x4
v_interp_p1_f32_e32 v8, v0, attr0.x
v_interp_p2_f32_e32 v8, v1, attr0.x
v_interp_p1_f32_e32 v9, v0, attr0.y
v_interp_p2_f32_e32 v9, v1, attr0.y
s_waitcnt lgkmcnt(0)
s_lshl_b32 s12, s6, 5
s_load_dwordx8 s[64:71], s[40:41], s12
s_waitcnt lgkmcnt(0)
image_sample v10, v[8:9], s[64:71], s[60:63] dmask:0x8 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_legacy_f32_e32 v11, s3, v10
v_cmpx_gt_f32_e32 s9, v11
s_andn2_b64 s[48:49], s[48:49], exec
s_cbranch_scc0 6
s_wqm_b64 s[54:55], s[48:49]
s_and_b64 exec, s[52:53], s[54:55]
s_and_b64 exec, exec, s[48:49]
exp null off, off, off, off done vm
s_endpgm
s_and_b64 exec, exec, s[48:49]
s_mov_b64 exec, s[48:49]
exp null off, off, off, off done vm
s_endpgm
""".strip().splitlines()


def variant(multiply, compare):
    return [
        line.replace("v_mul_legacy_f32_e32", multiply).replace(
            "v_cmpx_gt_f32_e32", compare
        )
        for line in PROGRAM
    ]


# The IEEE multiply with each kill relation that the translator accepts. The
# legacy multiply keeps its original greater-than program above.
RELATIONS = ("lt", "le", "gt", "ge")
OP_PROGRAMS = {
    relation: variant("v_mul_f32_e32", f"v_cmpx_{relation}_f32_e32")
    for relation in RELATIONS
}
PROGRAMS = [PROGRAM, *OP_PROGRAMS.values()]


def bindings(slot):
    return dict(
        interp=33,
        words=[[7, 16]],
        selector_byte=16,
        descriptors=[
            [10, 0, "selector", slot],
            [20, 64, "cb", 0],
            [20, 112, "cb", 4],
            [30, 16, "sampler", 0],
            [40, 0, "texture", 0],
            [40, 32, "texture", 1],
        ],
    )


def decoded(lines):
    # The relative branch only spans four-byte scalar instructions and one
    # eight-byte export. The binary test independently checks real LLVM PCs.
    pc = 0
    result = []
    for line in lines:
        result.append((pc, line))
        pc += (
            8
            if line.startswith(("exp ", "s_load_", "s_buffer_load_", "image_sample"))
            else 4
        )
    return result


def controls():
    program = decoded(PROGRAM)
    for old, new in [
        ("s_cbranch_scc0 6", "s_cbranch_scc0 5"),
        ("vmcnt(0)", "expcnt(0)"),
        ("s[52:53], s[54:55]", "s[54:55], s[54:55]"),
        ("s[60:63] dmask", "s[72:75] dmask"),
    ]:
        must_reject(
            lower, [(pc, line.replace(old, new)) for pc, line in program], bindings(0)
        )
    must_reject(lower, program, bindings(2))
    must_reject(lower, program, bindings(0) | {"selector_byte": 12})

    source = "\n".join(PROGRAM)
    selector_load = "s_buffer_load_dword s6, s[72:75], s7"
    zero_offset = source.replace(selector_load, selector_load.replace(", s7", ", null"))
    must_reject(lower, decoded(zero_offset.splitlines()), bindings(0) | {"selector_byte": False})

    overlap = bindings(0)
    overlap["descriptors"].append([20, 64, "cb", 0])
    must_reject(lower, program, overlap)

    wide_sampler = source.replace(
        "s_load_dwordx4 s[60:63], s[30:31], 0x10",
        "s_load_dwordx8 s[56:63], s[30:31], 0x10",
    ).replace("s[60:63] dmask", "s[56:59] dmask")
    must_reject(lower, decoded(wide_sampler.splitlines()), bindings(0))

    scalar_wait = "s_waitcnt lgkmcnt(0)"
    early_selector = source.replace(
        f"{scalar_wait}\n{selector_load}", f"{selector_load}\n{scalar_wait}"
    )
    must_reject(lower, decoded(early_selector.splitlines()), bindings(0))

    swapped_coords = source.replace("attr0.x", "attr0.y").replace(
        "v9, v0, attr0.y", "v9, v0, attr0.x"
    ).replace("v9, v1, attr0.y", "v9, v1, attr0.x")
    must_reject(lower, decoded(swapped_coords.splitlines()), bindings(0))

    sample_wait = "s_waitcnt vmcnt(0)"
    product = "v_mul_legacy_f32_e32 v11, s3, v10"
    early_product = source.replace(f"{sample_wait}\n{product}", f"{product}\n{sample_wait}")
    must_reject(lower, decoded(early_product.splitlines()), bindings(0))

    for relation, program in OP_PROGRAMS.items():
        for slot in range(2):
            lower(decoded(program), bindings(slot))
    ieee = "\n".join(OP_PROGRAMS["lt"])
    for old, new in [
        ("v_cmpx_lt_f32_e32 s9, v11", "v_cmpx_lg_f32_e32 s9, v11"),
        ("v_cmpx_lt_f32_e32 s9, v11", "v_cmpx_lt_f64_e32 s9, v11"),
        ("v_cmpx_lt_f32_e32 s9, v11", "v_cmpx_lt_f32_e32 v11, s9"),
        ("v_mul_f32_e32 v11, s3, v10", "v_mul_f32_e32 v11, v10, v10"),
        ("v_mul_f32_e32 v11, s3, v10", "v_mul_f16_e32 v11, s3, v10"),
        ("v_mul_f32_e32 v11, s3, v10", "v_add_f32_e32 v11, s3, v10"),
    ]:
        must_reject(lower, decoded(ieee.replace(old, new).splitlines()), bindings(0))
    ieee_product = "v_mul_f32_e32 v11, s3, v10"
    early_ieee = ieee.replace(
        f"{sample_wait}\n{ieee_product}", f"{ieee_product}\n{sample_wait}"
    )
    must_reject(lower, decoded(early_ieee.splitlines()), bindings(0))
    # Every wrong relation, and the legacy multiply, must give a different
    # reference somewhere, otherwise the readback controls prove nothing.
    for slot in range(2):
        for relation in RELATIONS:
            for other in RELATIONS:
                if relation != other:
                    distinguishing_case(slot, relation, False, other, False)
    for relation in RELATIONS:
        distinguishing_case(0, relation, False, relation, True)

    wrong_quad_source = source.replace(
        "s_wqm_b64 s[54:55], s[48:49]", "s_wqm_b64 s[54:55], s[52:53]"
    )
    must_reject(lower, decoded(wrong_quad_source.splitlines()), bindings(0))


def binary(tmp, *tools):
    for program in PROGRAMS:
        binary_program(tmp, program, *tools)


def binary_program(tmp, program, assembler, objcopy, library):
    src, obj, raw = (tmp / n for n in ("synthetic.s", "synthetic.o", "synthetic.bin"))
    src.write_text("\n".join(program) + "\n")
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
    container = tmp / "test.dxbc"
    record = envelope(padded, directory_end=(40 + len(padded), 288 + len(padded)))
    container.write_bytes(_read_named_container(record, len(b"fixture") + 2))
    for slot in range(2):
        manifest = tmp / "bindings.json"
        manifest.write_text(json.dumps(bindings(slot)))
        out = tmp / "compiled.hlsl"
        tool = Path(__file__).resolve().parents[1] / "tools/sampled_coverage_aot.py"
        subprocess.run(
            [
                sys.executable,
                "-B",
                tool,
                container,
                "--bindings",
                manifest,
                "--llvm",
                library,
                "--output",
                out,
            ],
            check=True,
        )
        assert out.read_text() == lower(decoded(program), bindings(slot))


def red_bindings(slot, scale_base=0):
    config = bindings(0x00800000)
    config["descriptors"][1] = [20, 64, "cb", scale_base]
    config["descriptors"][-4] = [20, 112, "cb", 8]
    config["descriptors"][-2:] = [
        [40, 0x80000000, "texture", slot],
        [40, 0x00800000, "texture", 1 - slot],
        [40, 0, "texture", 1 - slot],
    ]
    return config


RED_PROGRAM = [
    line.replace("s_lshl_b32 s12, s6, 5", "s_lshl_b32 s12, s6, 8")
    .replace("dmask:0x8", "dmask:0x1")
    for line in PROGRAM
]


def case(slot, scale, threshold, scale_word, threshold_word, red=False):
    cb = bytearray(256)
    struct.pack_into("<f", cb, 4 * scale_word, scale)
    struct.pack_into("<f", cb, 4 * threshold_word, threshold)
    ref = bytearray()
    for y in range(8):
        for x in range(8):
            byte = (41 * x + 19 * y) & 255
            sample = 1 - slot if red else (255 - byte if slot else byte) / 255
            ref.append(int(sample * scale >= threshold))
    return cb + ref


OP_CASES = (
    (1, 0),
    (1, 1),
    (1, 0.5),
    (-1, -1),
    (-1, 0),
    (2, 1.5),
    (math.inf, 0.25),
    (math.inf, -0.25),
)


def f32(value):
    if math.isnan(value) or math.isinf(value):
        return value
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def op_case(slot, scale, limit, relation, legacy=False):
    # An independent CPU model: round the unorm sample and the product to
    # binary32, then kill a pixel when the relation holds with the limit first.
    kill = getattr(operator, relation)
    cb = bytearray(256)
    struct.pack_into("<f", cb, 4, scale)
    struct.pack_into("<f", cb, 24, limit)
    ref = bytearray()
    for y in range(8):
        for x in range(8):
            byte = (41 * x + 19 * y) & 255
            sample = f32((255 - byte if slot else byte) / 255)
            product = 0.0 if legacy and (scale == 0 or sample == 0) else f32(scale * sample)
            ref.append(int(not kill(limit, product)))
    return cb + ref


def distinguishing_case(slot, relation, legacy, other, other_legacy):
    for scale, limit in OP_CASES:
        if op_case(slot, scale, limit, relation, legacy) != op_case(
            slot, scale, limit, other, other_legacy
        ):
            return scale, limit
    raise AssertionError(f"{relation} and {other} are not told apart on texture {slot}")


def draw_ops(tmp, dxc, renderer, adapter):
    for relation, program in OP_PROGRAMS.items():
        for slot in range(2):
            src, ps, ref = tmp / "draw.hlsl", tmp / "draw.dxil", tmp / "samples.bin"
            src.write_text(lower(decoded(program), bindings(slot)))
            subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Fo", ps, src], check=True)
            args = [renderer, adapter, "--sampled-coverage", ps, ref]
            for scale, limit in OP_CASES:
                ref.write_bytes(op_case(slot, scale, limit, relation))
                result = subprocess.run(args)
                if result.returncode == 77:
                    return 77
                result.check_returncode()
            wrong = [(other, False) for other in RELATIONS if other != relation]
            if slot == 0:
                wrong.append((relation, True))
            for other, legacy in wrong:
                scale, limit = distinguishing_case(slot, relation, False, other, legacy)
                ref.write_bytes(op_case(slot, scale, limit, other, legacy))
                result = subprocess.run(args, capture_output=True, text=True)
                if result.returncode != 1 or "Sample coverage mismatch" not in result.stderr:
                    raise AssertionError(
                        f"A {other} reference must fail the {relation} coverage readback"
                    )
    return 0


def draw(tmp, dxc, renderer, adapter, sources, scale_word, threshold_word, red=False):
    cases = ((1, 0.75),) if red else (
        (1, 0.5), (0.5, 0.375), (1, 0), (0, 0.25), (-1, -0.5)
    )
    for slot, source in enumerate(sources):
        src, ps = tmp / "draw.hlsl", tmp / "draw.dxil"
        src.write_text(source)
        subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Fo", ps, src], check=True)
        for scale, threshold in cases:
            ref = tmp / "samples.bin"
            ref.write_bytes(case(slot, scale, threshold, scale_word, threshold_word, red))
            args = [renderer, adapter, "--sampled-coverage", ps, ref]
            result = subprocess.run(args)
            if result.returncode == 77:
                return 77
            result.check_returncode()
        scale, threshold = cases[0]
        ref.write_bytes(case(1 - slot, scale, threshold, scale_word, threshold_word, red))
        wrong = subprocess.run(args, capture_output=True, text=True)
        if wrong.returncode != 1 or "Sample coverage mismatch" not in wrong.stderr:
            raise AssertionError(
                "A reference for the other texture must fail at coverage readback"
            )
    return 0


if __name__ == "__main__":
    controls()
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        if sys.argv[1] == "binary":
            binary(tmp, *sys.argv[2:])
        else:
            sys.exit(
                draw(
                    tmp,
                    *sys.argv[1:],
                    [lower(decoded(PROGRAM), bindings(slot)) for slot in range(2)],
                    1,
                    6,
                )
                or draw(
                    tmp,
                    *sys.argv[1:],
                    [lower(decoded(RED_PROGRAM), red_bindings(slot)) for slot in range(2)],
                    1,
                    10,
                    red=True,
                )
                or draw(
                    tmp,
                    *sys.argv[1:],
                    [lower(decoded(RED_PROGRAM), red_bindings(0, scale_base=2))],
                    3,
                    10,
                    red=True,
                )
                or draw_ops(tmp, *sys.argv[1:])
            )

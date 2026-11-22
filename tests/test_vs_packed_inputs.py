"""D3D-388: independently bound vertex elements packed into one VS register.

Run from the repository root:  python -m pytest tests/test_vs_packed_inputs.py
Every shader and DXBC container here is synthetic; no game data is used.
"""
from pathlib import Path
import struct
import sys

import pytest

import rdna2spirv
import spirv_stubs
from conftest import make_dxbc, requires_tools

sys.path.insert(0, str(Path(__file__).parent / "semantic"))
from spirv_mini_interp import run_fragment

pytestmark = requires_tools

VS_PACKED = (Path(__file__).parent / "semantic/fixtures/vs_packed_fetch.asm").read_text(encoding="utf-8")

VS_NO_FETCH = """
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 1.0
	exp pos0 v4, v4, v4, v5 done
	s_endpgm
"""

VS_SHIFTED = """
	s_swappc_b64 s[4:5], s[4:5]
	v_mov_b32_e32 v0, 0
	exp pos0 v8, v9, v10, v11 done
	exp param0 v0, v0, v6, v7
	s_endpgm
"""

VS_TWO_REGISTERS = """
	s_swappc_b64 s[4:5], s[4:5]
	v_add_f32 v1, v4, v8
	v_add_f32 v2, v7, v11
	exp pos0 v1, v2, v1, v1 done
	exp param0 v4, v5, v6, v7
	exp param1 v8, v9, v10, v11
	s_endpgm
"""


def isg1(elements):
    """Elements are (register, system value, mask, component type, semantic index)."""
    payload = bytearray(struct.pack("<II", len(elements), 8))
    name_offset = 8 + 32 * len(elements)
    for reg, sysv, mask, ctype, index in elements:
        element = bytearray(32)
        struct.pack_into("<6I", element, 0, 0, name_offset, index, sysv, ctype, reg)
        element[24], element[25] = mask, mask & -mask
        payload += element
    return b"ISG1", bytes(payload) + b"TEXCOORD\x00"


def test_no_fetch_with_packed_signature_declares_empty_vertex_interface(make_shader):
    shader = make_shader(VS_NO_FETCH, "vs_packed_unused",
                         dxbc=make_dxbc([isg1([(0, 0, 0x3, 3, 3), (0, 0, 0xC, 3, 7)])]))
    spvasm = shader.translate()
    shader.validate(spvasm)
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "%vin" not in spvasm
    assert meta["vertex_inputs"] == []
    assert [(e["semantic_index"], e["register"], e["mask"]) for e in meta["input_signature"]] == [
        (3, 0, 0x3), (7, 0, 0xC),
    ]


@pytest.mark.parametrize("ctype, scalar, pair, words", [
    (1, 0xDEADBEEF, 0x12345678, (0xDEADBEEF, 0x12345678)),
    (2, -7, -19, (0xFFFFFFF9, 0xFFFFFFED)),
    (3, -2.5, 0.125, (0xC0200000, 0x3E000000)),
])
@pytest.mark.parametrize("narrow", [False, True])
def test_standalone_shifted_mask_routes_format_components(make_shader, ctype, scalar, pair, words, narrow):
    shader = make_shader(VS_SHIFTED, "vs_shifted_mask",
                         dxbc=make_dxbc([isg1([(0, 0, 0xC, ctype, 7), (1, 0, 0x7, 3, 0)])]))
    spvasm = shader.translate()
    shader.validate(spvasm)
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    name, pointer = {1: ("uint", "%_ptr_Input_v4u"), 2: ("sint", "%_ptr_Input_v4i"),
                     3: ("float", "%_ptr_Input_v4")}[ctype]
    assert meta["vertex_inputs"] == [{"location": 0, "type": name}, {"location": 1, "type": "float"}]
    assert "%vin0 = OpVariable " + pointer + " Input" in spvasm
    assert meta["input_signature"][0]["mask"] == 0xC and meta["input_signature"][0]["location"] == 0
    # Vulkan supplies y=0 for a scalar format, and z=0, w=1 for both source formats.
    zero, one = (0.0, 1.0) if ctype == 3 else (0, 1)
    out = run_fragment(spvasm, inputs={"%vin0": (scalar, zero if narrow else pair, zero, one),
                                      "%vin1": (0.25, 0.5, 0.75, 1.0)})
    assert out["%out_param0"] == (0, 0, words[0], 0 if narrow else words[1])
    assert out["%out_pos"] == (0x3E800000, 0x3F000000, 0x3F400000, 0x3F800000)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("types", [(1, 1), (2, 2), (3, 3), (1, 2)])
def test_packed_register_routes_both_semantics(make_shader, reverse, types):
    lo, hi = types
    lo_index, hi_index = (7, 3) if lo != hi else (3, 7)
    packed = [(0, 0, 0x3, lo, lo_index), (0, 0, 0xC, hi, hi_index)]
    elements = packed[::-1] if reverse else packed
    elements += [(1, 0, 0x7, 3, 0)]
    shader = make_shader(VS_PACKED, "vs_packed_input",
                         dxbc=make_dxbc([isg1(elements)]))
    spvasm = shader.translate()
    shader.validate(spvasm)

    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    names = {1: "uint", 2: "sint", 3: "float"}
    assert meta["vertex_inputs"] == [
        {"location": 0, "type": names[lo]}, {"location": 2, "type": names[hi]}, {"location": 1, "type": "float"},
    ]
    assert [(e["semantic"], e["semantic_index"], e["register"], e["location"], e["mask"])
            for e in meta["input_signature"][:2]] == [
        ("TEXCOORD", index, reg, {lo_index: 0, hi_index: 2}[index], mask) for reg, _, mask, _, index in elements[:2]
    ]
    assert "location" not in meta["input_signature"][2]
    assert [(e["register"], e["mask"]) for e in meta["vertex_input_signature"]] == [
        (reg, mask) for reg, _, mask, _, _ in elements
    ]
    assert spirv_stubs.required_stubs_for_meta(meta) == [("ps", (0,))]

    # Each attribute supplies x/y, with Vulkan's z/w defaults for a two-component format.
    values = {1: (0x12345678, 0xABCDEF01, 0x3456789A, 0xFEDCBA98),
              2: (-17, 23, -91, 123), 3: (-2.5, 0.125, 3.5, -0.75)}
    xy, zw = values[lo][:2], values[hi][2:]
    position = (0.25, 0.5, 0.75, 1.0)
    out = run_fragment(spvasm, inputs={"%vin0": (*xy, 0, 1), "%vin2": (*zw, 0, 1), "%vin1": position})
    word = lambda x: struct.unpack("<I", struct.pack("<f", x))[0] if isinstance(x, float) else x & 0xFFFFFFFF
    assert out["%out_param0"] == tuple(word(v) for v in (*xy, *zw))
    assert out["%out_pos"] == tuple(word(v) for v in position)


def test_distinct_integer_registers_keep_their_component_types(make_shader):
    shader = make_shader(VS_TWO_REGISTERS, "vs_typed_inputs",
                         dxbc=make_dxbc([isg1([(0, 0, 0xF, 1, 3), (1, 0, 0xF, 2, 7)])]))
    spvasm = shader.translate()
    shader.validate(spvasm)

    gen = rdna2spirv.translate.last_gen
    assert gen.vertex_inputs == [(0, 1), (1, 2)]
    assert "%vin0 = OpVariable %_ptr_Input_v4u Input" in spvasm
    assert "%vin1 = OpVariable %_ptr_Input_v4i Input" in spvasm
    assert rdna2spirv.shader_meta(gen, "")["vertex_inputs"] == [
        {"location": 0, "type": "uint"}, {"location": 1, "type": "sint"}]


@pytest.mark.parametrize("mask, reason", [(0x6, "overlapping packed masks"), (0x4, "fetch mapping")])
def test_packed_register_cannot_guess_a_source(make_shader, mask, reason):
    shader = make_shader(VS_PACKED, "vs_bad_packed_input",
                         dxbc=make_dxbc([isg1([(0, 0, 0x3, 1, 3), (0, 0, mask, 3, 7), (1, 0, 0xF, 3, 0)])]))
    with pytest.raises(rdna2spirv.Unsupported, match=reason):
        shader.translate()

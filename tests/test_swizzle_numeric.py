"""Regression coverage for numeric ds_swizzle offsets emitted by LLVM."""

import os
import sys

import pytest

from conftest import ROOT, requires_tools

sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))
from support import translate_lines


@requires_tools
@pytest.mark.parametrize("offset, opcode", [
    (0x801b, "OpSelect"),
    (0xc020, "OpIAdd"),
    (0xc420, "OpIAdd"),
    (0xc078, "OpIAdd"),
    (0xe000, "OpBitReverse"),
    (0xe010, "OpBitReverse"),
    (0xe01f, "OpBitReverse"),
])
def test_swizzle_translation(make_shader, offset, opcode):
    shader = make_shader(f"ds_swizzle_b32 v1, v0 offset:{offset}\ns_endpgm\n", "cs_swizzle")
    assembly = shader.translate()
    assert opcode in assembly
    assert "OpGroupNonUniformShuffle" in assembly
    shader.validate(assembly)

@pytest.mark.parametrize("symbolic, numeric", [
    ("swizzle(ROTATE,0,1)", "0xc020"),
    ("swizzle(ROTATE,1,1)", "0xc420"),
    ("swizzle(FFT,3)", "0xe003"),
    ("swizzle(ROTATE,0,3) ; encoding: [0x78,0xc0,0xd4,0xd8,0,0,0,1]", "0xc078"),
])
def test_symbolic_swizzle_matches_numeric(symbolic, numeric):
    def translate(offset):
        return translate_lines("ps_probe", [f"ds_swizzle_b32 v1, v0 offset:{offset}", "s_endpgm"])

    assert translate(symbolic) == translate(numeric)

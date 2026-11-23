"""Regression coverage for numeric ds_swizzle offsets emitted by LLVM."""

import pytest

from conftest import requires_tools


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

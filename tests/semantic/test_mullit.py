"""V_MULLIT_F32 in the inferred mode (IEEE off, f32 denormals flushed)."""
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import Shader, requires_tools
from support import mrt0_words, rdna2spirv


@requires_tools
def test_mullit(make_shader: Callable[[str, str], Shader]) -> None:
    fixture = Path(__file__).with_name("fixtures") / "ps_mullit.asm"
    shader = make_shader(fixture.read_text(encoding="ascii"), fixture.stem)
    module = shader.translate()
    shader.validate(module)
    lowest, inf, one = 0xFF7FFFFF, 0x7F800000, 0x3F800000
    for lhs, rhs, gate, expected in (
        (0x3E800000, 0x3F000000, one, 0x3E000000),       # lit: 0.25 * 0.5
        (one, one, 0, lowest), (one, one, 0xBF800000, lowest),
        (one, one, 1, lowest), (one, one, 0x7FC00001, lowest),
        (0, lowest, one, lowest), (one, 0xFF800000, one, lowest),
        (one, 0x7FC00001, one, lowest), (0, inf, one, 0),
        (0x7FC12345, 0, one, 0), (0x7FA12345, one, one, 0x7FA12345),
        (0x80000001, one, one, 0), (0x80800000, 0x3F000000, one, 0x80000000),
        (0x3F7FFFFE, 0x00800001, one, 0x00800000),
        (0x3F7FFFFF, 0x00800000, one, 0),
        (0x7F7FFFFF, 0x40000000, inf, inf),
    ):
        words = mrt0_words(module, push_constants=[lhs, rhs, gate] + [0] * 29)
        assert words[3] == expected ^ 0x40000000
        assert words[2] == lowest
        if lhs == 0x3E800000:
            assert words[0:2] == [0x3E000000, 0xBE000000]
        if rhs == 0xFF800000 and lhs == one:
            assert words[1] == 0xFF800000             # -1 * abs(-inf)


@requires_tools
@pytest.mark.parametrize("modifier", ["clamp", "mul:2"])
def test_mullit_unverified_output_modifier(make_shader: Callable[[str, str], Shader], modifier: str) -> None:
    shader = make_shader(f"v_mullit_f32 v4, v5, v6, v7 {modifier}\ns_endpgm\n", "ps_mullit_mod")
    with pytest.raises(rdna2spirv.Unsupported, match="v_mullit_f32 output modifiers"):
        shader.translate()

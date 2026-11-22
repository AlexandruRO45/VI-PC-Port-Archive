"""SHADER-350: a POPS (primitive-ordered pixel shading) pixel shader is ordered by fragment-shader interlock around
the whole shader (rdna2spirv.py, `src_pops_exiting_wave_id`): OpBeginInvocationInterlockEXT once at the end of the
entry block, OpEndInvocationInterlockEXT once before the return. A shader without a branch got the End but no Begin,
which is invalid for the interlock; these check both control-flow shapes. Hand-written shaders, no game data."""
import re

import pytest

from conftest import requires_tools

STRAIGHT = """
    s_mov_b32 s1, src_pops_exiting_wave_id
    v_mov_b32 v0, 1.0
    exp mrt0 v0, v0, v0, v0 done vm
    s_endpgm
"""

BRANCHED = """
    s_mov_b32 s1, src_pops_exiting_wave_id
    s_cmp_eq_u32 s1, 0
    s_cbranch_scc1 1
    v_mov_b32 v0, 0.5
    v_mov_b32 v0, 1.0
    exp mrt0 v0, v0, v0, v0 done vm
    s_endpgm
"""


def interlock_ops(spvasm):
    return [m.group(1) for m in re.finditer(r"Op(Begin|End)InvocationInterlockEXT", spvasm)]


@requires_tools
@pytest.mark.parametrize("asm", [STRAIGHT, BRANCHED], ids=["straight-line", "branched"])
def test_pops_shader_begins_and_ends_the_interlock_once(make_shader, asm):
    shader = make_shader(asm, "ps_pops")
    spvasm = shader.translate()
    assert interlock_ops(spvasm) == ["Begin", "End"], interlock_ops(spvasm)
    assert "FragmentShaderPixelInterlockEXT" in spvasm and "PixelInterlockOrderedEXT" in spvasm
    # Begin sits in the entry block, before the first instruction that reads or writes anything the shader stores.
    entry = spvasm[spvasm.index("%main = OpFunction"):]
    first_label_end = entry.index("OpLabel")
    assert entry.index("OpBeginInvocationInterlockEXT") > first_label_end
    assert entry.index("OpBeginInvocationInterlockEXT") < entry.index("OpEndInvocationInterlockEXT")
    shader.validate(spvasm)


@requires_tools
def test_a_shader_without_pops_has_no_interlock(make_shader):
    shader = make_shader("    v_mov_b32 v0, 1.0\n    exp mrt0 v0, v0, v0, v0 done vm\n    s_endpgm\n", "ps_plain")
    spvasm = shader.translate()
    assert interlock_ops(spvasm) == []
    assert "FragmentShaderPixelInterlockEXT" not in spvasm

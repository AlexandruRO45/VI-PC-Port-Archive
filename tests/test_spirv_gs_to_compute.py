"""tools/spirv_gs_to_compute.py: a translated geometry shader as a compute capture pass (SHADER-159)."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from conftest import HAVE_SPIRV, make_dxbc, requires_tools, signature_chunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spirv_gs_to_compute as gsc  # noqa: E402

# gs_tri's signature, as scripts/run_gpu_tests.py builds it: SV_Position in register 0, TEXCOORD0 in register 1.
GS_DXBC = make_dxbc([signature_chunk(b"ISG1", [(0, 1, 0xF), (1, 0, 0xF)]),
                     signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 0, 0xF)])])


@pytest.fixture
def gs_tri(make_shader):
    asm = open(os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm"), encoding="utf-8").read()
    shader = make_shader(asm, name="gs_tri", dxbc=GS_DXBC)
    spvasm = shader.translate()
    assert "OpEntryPoint Geometry" in spvasm
    return shader, spvasm


@requires_tools
def test_layout(gs_tri):
    _, spvasm = gs_tri
    _, lay = gsc.rewrite(spvasm)
    assert lay == {"verts_in": 3, "max_verts": 3, "topology": "triangle_strip",
                   "inputs": ["position", "location 1"], "outputs": ["position", "location 1"], "local_size": 64}


@requires_tools
def test_compute_module(gs_tri):
    shader, spvasm = gs_tri
    out, _ = gsc.rewrite(spvasm)
    assert "OpCapability Geometry" not in out
    assert re.search(r"OpEntryPoint GLCompute %\w+ \"main\"", out)
    assert re.search(r"OpExecutionMode %\w+ LocalSize 64 1 1", out)
    assert "OpEmitVertex" not in out and "OpEndPrimitive" not in out
    # no outputs are left, and the only inputs are GlobalInvocationId and the subgroup lane the GS already read
    assert not re.search(r"OpVariable %\w+ Output\b", out)
    inputs = re.findall(r"(%\w+) = OpVariable %\w+ Input\b", out)
    builtins = {v: re.search(rf"OpDecorate {v} BuiltIn (\w+)", out).group(1) for v in inputs}
    assert sorted(builtins.values()) == ["GlobalInvocationId", "SubgroupLocalInvocationId"]
    # three emits, two outputs each, go to the record buffer
    assert len(re.findall(r"OpAccessChain %gsc_ptr_sb_v4 %gsc_rec ", out)) == 6
    for b in (0, 1, 2):
        assert f"Binding {b}" in out
    shader.validate(out)    # spirv-as + spirv-val, as for the translator's own output


@requires_tools
def test_rejects_non_geometry(make_shader):
    shader = make_shader("v_mov_b32 v1, 1.0\ns_endpgm", name="cs_not_gs")
    with pytest.raises(gsc.Unsupported):
        gsc.rewrite(shader.translate())


@requires_tools
def test_cli_writes_module_and_layout(gs_tri, tmp_path):
    shader, spvasm = gs_tri
    sa, sv = tmp_path / "gs.spvasm", tmp_path / "gs.spv"
    sa.write_text(spvasm)
    subprocess.run(["spirv-as", str(sa), "-o", str(sv), "--target-env", "vulkan1.2"], check=True)
    out = tmp_path / "cs.spv"
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "spirv_gs_to_compute.py"), str(sv),
                        str(out), "--layout", str(tmp_path / "lay.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "lay.json").exists()
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(out)], check=True)


# A hand-built GS whose entry block branches, with an OpPhi naming the entry label as a predecessor (the translator
# emits those for guarded loads: its block_label is %entry until the first new label). The primitive-count check
# splits the entry block, so the OpPhi must name the block that now ends the entry code (review of #1331).
PHI_GS = """\
OpCapability Shader
OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %in_pos %out_pos
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpDecorate %in_pos BuiltIn Position
OpDecorate %out_pos BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%u0 = OpConstant %uint 0
%u1 = OpConstant %uint 1
%u3 = OpConstant %uint 3
%arr = OpTypeArray %v4float %u3
%ptr_in_arr = OpTypePointer Input %arr
%ptr_in_v4 = OpTypePointer Input %v4float
%ptr_out_v4 = OpTypePointer Output %v4float
%in_pos = OpVariable %ptr_in_arr Input
%out_pos = OpVariable %ptr_out_v4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%c = OpIEqual %bool %u0 %u1
OpSelectionMerge %m None
OpBranchConditional %c %t %m
%t = OpLabel
OpBranch %m
%m = OpLabel
%x = OpPhi %uint %u1 %t %u0 %entry
%p = OpAccessChain %ptr_in_v4 %in_pos %x
%v = OpLoad %v4float %p
OpStore %out_pos %v
OpEmitVertex
OpEmitVertex
OpEmitVertex
OpEndPrimitive
OpReturn
OpFunctionEnd
"""


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
def test_phi_from_entry_block_names_the_split_block(tmp_path):
    out, _ = gsc.rewrite(PHI_GS)
    assert "%x = OpPhi %uint %u1 %t %u0 %gsc_body" in out
    sa, sv = tmp_path / "cs.spvasm", tmp_path / "cs.spv"
    sa.write_text(out)
    subprocess.run(["spirv-as", str(sa), "-o", str(sv), "--target-env", "vulkan1.2"], check=True)
    r = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(sv)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
@pytest.mark.parametrize(("component_chain", "use_alias"), [
    (None, False),
    ("OpAccessChain", False),
    ("OpInBoundsAccessChain", False),
    ("OpAccessChain", True),
    ("OpInBoundsAccessChain", True),
])
def test_lowers_in_bounds_geometry_interface_access_chains(
        tmp_path: Path, component_chain: str | None, use_alias: bool) -> None:
    geometry = PHI_GS.replace(
        "%p = OpAccessChain %ptr_in_v4 %in_pos %x",
        "%p = OpInBoundsAccessChain %ptr_in_v4 %in_pos %x",
    ).replace(
        "%out_pos = OpVariable %ptr_out_v4 Output",
        "%ptr_out_float = OpTypePointer Output %float\n%out_pos = OpVariable %ptr_out_v4 Output",
    ).replace(
        "OpStore %out_pos %v",
        "OpStore %out_pos %v\n"
        "%out_x = OpInBoundsAccessChain %ptr_out_float %out_pos %u0\n"
        "%x_value = OpCompositeExtract %float %v 0\n"
        "OpStore %out_x %x_value",
    )

    if component_chain:
        component_base = "%alias" if use_alias else "%p"
        alias = f"%alias = {component_chain} %ptr_in_v4 %p\n" if use_alias else ""
        geometry = geometry.replace(
            "%in_pos = OpVariable %ptr_in_arr Input",
            "%ptr_in_float = OpTypePointer Input %float\n%in_pos = OpVariable %ptr_in_arr Input",
        ).replace(
            "%v = OpLoad %v4float %p",
            f"{alias}%in_x = {component_chain} %ptr_in_float {component_base} %u1\n"
            "%value = OpLoad %float %in_x\n"
            "%v = OpCompositeConstruct %v4float %value %value %value %value",
        )

    validate_vulkan12(tmp_path / "geometry.spv", geometry)
    lowered, _ = gsc.rewrite(geometry)
    assert "%p = OpAccessChain %gsc_ptr_sb_v4 %gsc_in " in lowered
    assert "%out_x = OpAccessChain %gsc_ptr_pv_f %gsc_out0 %u0" in lowered
    validate_vulkan12(tmp_path / "compute.spv", lowered)
    if component_chain:
        assert f"%in_x = {component_chain} %gsc_ptr_sb_f {component_base} %u1" in lowered
        if use_alias:
            assert f"%alias = {component_chain} %gsc_ptr_sb_v4 %p" in lowered


def validate_vulkan12(module: Path, assembly: str) -> None:
    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", str(module)],
                   input=assembly, text=True, check=True)
    result = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(module)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def geometry_input_alias(array_chain: str) -> str:
    return PHI_GS.replace(
        '"main" %in_pos %out_pos', '"main" %in_pos %in_color %out_pos',
    ).replace(
        "OpDecorate %in_pos BuiltIn Position",
        'OpName %array_alias "array_alias"\nOpName %nested_alias "nested_alias"\n'
        'OpName %p "vertex_ptr"\nOpDecorate %in_color Location 0\nOpDecorate %in_pos BuiltIn Position',
    ).replace(
        "%in_pos = OpVariable %ptr_in_arr Input",
        "%in_pos = OpVariable %ptr_in_arr Input\n%in_color = OpVariable %ptr_in_arr Input",
    ).replace(
        "%p = OpAccessChain %ptr_in_v4 %in_pos %x",
        f"%array_alias = {array_chain} %ptr_in_arr %in_color\n"
        f"%nested_alias = {array_chain} %ptr_in_arr %array_alias\n"
        f"%p = {array_chain} %ptr_in_v4 %nested_alias %u1",
    )


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
@pytest.mark.parametrize("array_chain", ["OpAccessChain", "OpInBoundsAccessChain"])
def test_lowers_alias_of_geometry_input_array(tmp_path, array_chain):
    geometry = geometry_input_alias(array_chain)
    validate_vulkan12(tmp_path / "geometry.spv", geometry)
    lowered, layout = gsc.rewrite(geometry)
    validate_vulkan12(tmp_path / "compute.spv", lowered)

    assert "%array_alias" not in lowered and "%nested_alias" not in lowered
    assert 'OpName %p "vertex_ptr"' in lowered
    assert layout["inputs"] == ["position", "location 0"]
    defs, _ = gsc.parse(lowered)
    op, (ptr, base, member, offset) = defs["%p"]
    assert (op, ptr, base, member) == ("OpAccessChain", "%gsc_ptr_sb_v4", "%gsc_in", "%gsc_k2")
    assert defs[ptr] == ("OpTypePointer", ["StorageBuffer", "%v4float"])
    op, (result_type, vertex_stride, slot) = defs[offset]
    assert (op, result_type, slot) == ("OpIAdd", "%uint", "%gsc_k1")
    op, (result_type, vertex, stride) = defs[vertex_stride]
    assert (op, result_type, stride) == ("OpIMul", "%uint", "%gsc_k2")
    assert defs[vertex] == ("OpIAdd", ["%uint", "%gsc_inbase", "%u1"])
    assert defs["%v"] == ("OpLoad", ["%v4float", "%p"])


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
def test_rejects_whole_geometry_input_array_load_through_alias(tmp_path):
    geometry = geometry_input_alias("OpAccessChain").replace(
        "%v = OpLoad %v4float %p",
        "%array = OpLoad %arr %nested_alias\n%v = OpCompositeExtract %v4float %array 1",
    )
    validate_vulkan12(tmp_path / "geometry.spv", geometry)
    with pytest.raises(gsc.Unsupported, match=r"unhandled use of %nested_alias: .*OpLoad"):
        gsc.rewrite(geometry)


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
@pytest.mark.parametrize(("targets", "remaining"), [
    ("%v %array_alias %p %nested_alias", "%v %p"),
    ("%array_alias %nested_alias", ""),
    ("%v  %p", "%v  %p"),
], ids=["mixed", "all-folded", "live-only"])
def test_prunes_folded_input_aliases_from_decoration_groups(tmp_path, targets, remaining):
    comment = "; keep %array_alias and %nested_alias here; comment only"
    geometry = geometry_input_alias("OpInBoundsAccessChain").replace(
        "%void = OpTypeVoid",
        "%relaxed = OpDecorationGroup\n"
        "OpDecorate %relaxed RelaxedPrecision\n"
        f"OpGroupDecorate %relaxed {targets} {comment}\n"
        "%void = OpTypeVoid",
    )
    validate_vulkan12(tmp_path / "geometry.spv", geometry)
    lowered, _ = gsc.rewrite(geometry)
    validate_vulkan12(tmp_path / "compute.spv", lowered)

    group_uses = [line for line in lowered.splitlines() if line.startswith("OpGroupDecorate ")]
    assert group_uses == ([f"OpGroupDecorate %relaxed {remaining} {comment}"] if remaining else [])
    assert "OpDecorate %relaxed RelaxedPrecision" in lowered
    assert 'OpName %p "vertex_ptr"' in lowered
    assert "OpName %array_alias" not in lowered and "OpName %nested_alias" not in lowered


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
@pytest.mark.parametrize("component_chain", ["OpAccessChain", "OpInBoundsAccessChain"])
def test_lowers_copies_of_geometry_input_element_pointers(tmp_path, component_chain):
    geometry = PHI_GS.replace(
        "OpDecorate %in_pos BuiltIn Position",
        'OpName %copied_vertex "vertex_copy"\nOpName %local_copy "local_copy"\n'
        "OpDecorate %in_pos BuiltIn Position",
    ).replace(
        "%in_pos = OpVariable %ptr_in_arr Input",
        "%ptr_in_float = OpTypePointer Input %float\n"
        "%ptr_fn_v4 = OpTypePointer Function %v4float\n%ptr_fn_float = OpTypePointer Function %float\n"
        "%in_pos = OpVariable %ptr_in_arr Input",
    ).replace(
        "%entry = OpLabel", "%entry = OpLabel\n%local = OpVariable %ptr_fn_v4 Function",
    ).replace(
        "%v = OpLoad %v4float %p",
        "%copied_vertex = OpCopyObject %ptr_in_v4 %p\n"
        f"%component = {component_chain} %ptr_in_float %copied_vertex %u1\n"
        "%copied_component = OpCopyObject %ptr_in_float %component\n"
        "%value = OpLoad %float %copied_component\n"
        "%local_copy = OpCopyObject %ptr_fn_v4 %local\n"
        f"%local_component = {component_chain} %ptr_fn_float %local_copy %u1\n"
        "OpStore %local_component %value\n"
        "%local_value = OpLoad %float %local_component\n"
        "%v = OpCompositeConstruct %v4float %value %local_value %value %value",
    )
    validate_vulkan12(tmp_path / "geometry.spv", geometry)
    lowered, _ = gsc.rewrite(geometry)
    validate_vulkan12(tmp_path / "compute.spv", lowered)

    defs, _ = gsc.parse(lowered)
    assert defs["%p"][1][:3] == ["%gsc_ptr_sb_v4", "%gsc_in", "%gsc_k2"]
    assert defs["%gsc_ptr_sb_v4"] == ("OpTypePointer", ["StorageBuffer", "%v4float"])
    assert defs["%gsc_ptr_sb_f"] == ("OpTypePointer", ["StorageBuffer", "%float"])
    assert defs["%copied_vertex"] == ("OpCopyObject", ["%gsc_ptr_sb_v4", "%p"])
    assert defs["%component"] == (component_chain, ["%gsc_ptr_sb_f", "%copied_vertex", "%u1"])
    assert defs["%copied_component"] == ("OpCopyObject", ["%gsc_ptr_sb_f", "%component"])
    assert defs["%value"] == ("OpLoad", ["%float", "%copied_component"])
    assert 'OpName %copied_vertex "vertex_copy"' in lowered
    assert 'OpName %local_copy "local_copy"' in lowered
    original_defs, _ = gsc.parse(geometry)
    for local in ("%ptr_fn_v4", "%ptr_fn_float", "%local", "%local_copy", "%local_component"):
        assert defs[local] == original_defs[local]


@pytest.fixture
def gs_module(tmp_path):
    if not (HAVE_SPIRV and shutil.which("spirv-dis")):
        pytest.skip("needs SPIRV-Tools on PATH")
    source = tmp_path / "gs.spv"
    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", str(source)],
                   input=PHI_GS, text=True, check=True)
    return source


@pytest.mark.parametrize("overlap", ["input-output", "input-layout", "output-layout", "hardlinked-layout"])
def test_cli_rejects_overlapping_paths_without_writing(tmp_path, gs_module, overlap):
    source = gs_module
    output = tmp_path / "cs.spv"
    layout = tmp_path / "layout.json"
    original = source.read_bytes()
    if overlap == "input-output":
        output = source
    elif overlap == "input-layout":
        layout = source
    elif overlap == "output-layout":
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        layout = subdir / ".." / output.name
    else:
        try:
            os.link(source, layout)
        except OSError as e:
            pytest.skip(f"hard links unavailable: {e}")

    result = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "spirv_gs_to_compute.py"),
                             str(source), str(output), "--layout", str(layout)], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert "same file" in result.stderr
    assert source.read_bytes() == original
    assert not (tmp_path / "cs.spv").exists()
    if overlap == "hardlinked-layout":
        assert layout.read_bytes() == original


def test_cli_accepts_distinct_module_and_layout_paths(tmp_path, gs_module):
    output = tmp_path / "cs.spv"
    layout = tmp_path / "layout.json"
    original = gs_module.read_bytes()
    result = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "spirv_gs_to_compute.py"),
                             str(gs_module), str(output), "--layout", str(layout)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert gs_module.read_bytes() == original
    assert json.loads(layout.read_text(encoding="utf-8"))["topology"] == "triangle_strip"
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(output)], check=True)


def test_help_is_a_clean_exit_zero(capsys):
    for flag in ("-h", "--help"):
        assert gsc.main(["spirv_gs_to_compute.py", flag]) == 0
        out = capsys.readouterr().out
        assert "spirv_gs_to_compute.py in.spv out.spv" in out
        assert "SPIRV-Tools" in out


def test_builtin_load_and_output_store_with_memory_operands():
    gs = PHI_GS.replace(
        'OpEntryPoint Geometry %main "main" %in_pos %out_pos',
        'OpEntryPoint Geometry %main "main" %in_pos %out_pos %prim_id %inv_id',
    ).replace(
        "OpDecorate %out_pos BuiltIn Position",
        "OpDecorate %out_pos BuiltIn Position\n"
        "OpDecorate %prim_id BuiltIn PrimitiveId\n"
        "OpDecorate %inv_id BuiltIn InvocationId",
    ).replace(
        "%out_pos = OpVariable %ptr_out_v4 Output",
        "%ptr_in_u = OpTypePointer Input %uint\n"
        "%out_pos = OpVariable %ptr_out_v4 Output\n"
        "%prim_id = OpVariable %ptr_in_u Input\n"
        "%inv_id = OpVariable %ptr_in_u Input",
    ).replace(
        "%v = OpLoad %v4float %p\nOpStore %out_pos %v",
        "%pid = OpLoad %uint %prim_id Aligned 4\n"
        "%iid = OpLoad %uint %inv_id Volatile\n"
        "%v = OpLoad %v4float %p\n"
        "OpStore %out_pos %v Aligned 16",
    )
    out, _ = gsc.rewrite(gs)
    assert "%prim_id" not in out and "%inv_id" not in out
    assert "%pid = OpCopyObject %uint %gsc_prim" in out
    assert "%iid = OpCopyObject %uint " in out
    assert "OpStore %gsc_out0 %v" in out
    if HAVE_SPIRV:
        res = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", os.devnull],
                             input=out, text=True, capture_output=True)
        assert res.returncode == 0, res.stderr

    bad_gs = gs.replace("%pid = OpLoad %uint %prim_id Aligned 4", "OpCopyMemory %prim_id %inv_id")
    with pytest.raises(gsc.Unsupported, match="%prim_id"):
        gsc.rewrite(bad_gs)

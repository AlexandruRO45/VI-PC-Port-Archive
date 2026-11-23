"""tools/spirv_metal_set0.py: set 0's descriptor arrays get fixed, non-overlapping Metal IDs (SHADER-161)."""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import spirv_metal_set0 as m  # noqa: E402

# The set-0 part of a translated pixel shader (spirv-dis, names kept): sampled 2D images, samplers, the hash table.
MODULE = """\
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
               OpDecorate %samplers DescriptorSet 0
               OpDecorate %samplers Binding 1
               OpDecorate %desc_map DescriptorSet 0
               OpDecorate %desc_map Binding 2
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
        %smp = OpTypeSampler
     %rt_img = OpTypeRuntimeArray %img2d
     %rt_smp = OpTypeRuntimeArray %smp
    %ptr_img = OpTypePointer UniformConstant %rt_img
    %ptr_smp = OpTypePointer UniformConstant %rt_smp
     %rt_map = OpTypeRuntimeArray %uint
        %map = OpTypeStruct %rt_map
    %ptr_map = OpTypePointer StorageBuffer %map
     %images = OpVariable %ptr_img UniformConstant
   %samplers = OpVariable %ptr_smp UniformConstant
   %desc_map = OpVariable %ptr_map StorageBuffer
"""


def test_capacities_match_vulkan_bindless():
    counts = m.capacities()
    assert counts[:5] == [4096, 1024, 1, 1024, 1024]
    assert len(counts) == 26


def test_metal_ids_are_running_sums():
    assert m.metal_ids([4096, 1024, 1, 1024]) == [0, 4096, 5120, 5121]


def test_rewrite_sizes_arrays_and_spaces_bindings():
    counts = m.capacities()
    out = m.rewrite(MODULE, counts)
    assert "OpDecorate %images Binding 0" in out
    assert "OpDecorate %samplers Binding 4096" in out
    assert "OpDecorate %desc_map Binding 5120" in out
    assert "%msl_cap_b0 = OpConstant %uint 4096" in out
    assert "%msl_arr_b0 = OpTypeArray %img2d %msl_cap_b0" in out
    assert "%images = OpVariable %msl_ptr_b0 UniformConstant" in out
    assert "%msl_arr_b1 = OpTypeArray %smp %msl_cap_b1" in out
    assert "%samplers = OpVariable %msl_ptr_b1 UniformConstant" in out
    # the hash table is a storage buffer, not a descriptor array: only its binding moves
    assert "%desc_map = OpVariable %ptr_map StorageBuffer" in out
    assert "msl_arr_b2" not in out
    # each new type comes after the pointer type it replaces, so its operands are already declared
    assert out.index("%ptr_img = OpTypePointer") < out.index("%msl_cap_b0")


def test_help_is_a_clean_exit_zero(capsys):
    for flag in ("-h", "--help"):
        assert m.main(["spirv_metal_set0.py", flag]) == 0
        assert "Usage: spirv_metal_set0.py" in capsys.readouterr().out


# Both sampler arrays use the same input type, but only the set-0 array is resized.
SAMPLE_FRAGMENT = """\
OpCapability Shader
OpCapability RuntimeDescriptorArray
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %images %samplers %foreign_samplers %color
OpExecutionMode %main OriginUpperLeft
OpName %images "images"
OpName %samplers "samplers"
OpName %foreign_samplers "foreign_samplers"
OpName %foreign_copy "foreign_copy"
OpName %foreign_zero "foreign_zero"
OpName %image_access "image_access"
OpName %sampler_access "sampler_access"
OpName %foreign_access "foreign_access"
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 0
OpDecorate %samplers DescriptorSet 0
OpDecorate %samplers Binding 1
OpDecorate %foreign_samplers DescriptorSet 1
OpDecorate %foreign_samplers Binding 1
OpDecorate %color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v2f = OpTypeVector %float 2
%v4f = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%smp = OpTypeSampler
%sampled = OpTypeSampledImage %image
%image_array = OpTypeRuntimeArray %image
%sampler_array = OpTypeRuntimeArray %smp
%images_ptr = OpTypePointer UniformConstant %image_array
%samplers_ptr = OpTypePointer UniformConstant %sampler_array
%image_ptr = OpTypePointer UniformConstant %image
%sampler_ptr = OpTypePointer UniformConstant %smp
%color_ptr = OpTypePointer Output %v4f
%zero = OpConstant %uint 0
%half = OpConstant %float 0.5
%uv = OpConstantComposite %v2f %half %half
%images = OpVariable %images_ptr UniformConstant
%samplers = OpVariable %samplers_ptr UniformConstant
%foreign_samplers = OpVariable %samplers_ptr UniformConstant
%color = OpVariable %color_ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
%foreign_zero = OpInBoundsAccessChain %samplers_ptr %foreign_samplers
%foreign_copy = OpCopyObject %samplers_ptr %foreign_zero
%image_access = OpAccessChain %image_ptr %images %zero
%sampler_access = OpAccessChain %sampler_ptr %samplers %zero
%foreign_access = OpAccessChain %sampler_ptr %foreign_copy %zero
%i = OpLoad %image %image_access
%s = OpLoad %smp %sampler_access
%f = OpLoad %smp %foreign_access
%local_texture = OpSampledImage %sampled %i %s
%foreign_texture = OpSampledImage %sampled %i %f
%a = OpImageSampleImplicitLod %v4f %local_texture %uv
%b = OpImageSampleImplicitLod %v4f %foreign_texture %uv
%rgba = OpFAdd %v4f %a %b
OpStore %color %rgba
OpReturn
OpFunctionEnd
"""


ALIAS_CASES = [
    pytest.param("samplers", 1, (), id="direct"),
    pytest.param("samplers", 1, ("OpCopyObject",), id="sampler-copy"),
    pytest.param("samplers", 1, ("OpCopyObject", "OpCopyObject"), id="sampler-copy-chain"),
    pytest.param("images", 3, ("OpCopyObject",), id="image-copy"),
    pytest.param("samplers", 1, ("OpAccessChain",), id="sampler-zero-chain"),
    pytest.param("samplers", 1, ("OpInBoundsAccessChain",), id="sampler-zero-inbounds"),
    pytest.param("images", 3, ("OpAccessChain",), id="image-zero-chain"),
    pytest.param("images", 3, ("OpInBoundsAccessChain",), id="image-zero-inbounds"),
    pytest.param("samplers", 1, ("OpCopyObject", "OpAccessChain", "OpInBoundsAccessChain", "OpCopyObject"),
                 id="sampler-mixed-chain"),
    pytest.param("images", 3, ("OpInBoundsAccessChain", "OpCopyObject", "OpAccessChain"),
                 id="image-mixed-chain"),
]


def descriptor_alias_fragment(resource, binding, operations):
    assembly = SAMPLE_FRAGMENT
    if resource == "images":
        assembly = assembly.replace("OpDecorate %images Binding 0", f"OpDecorate %images Binding {binding}")
    base = f"%{resource}"
    for i, op in enumerate(operations):
        copy = f"%copy{i}"
        assembly = assembly.replace(
            "OpDecorate %images DescriptorSet 0",
            f'OpName {copy} "copy{i}"\nOpDecorate %images DescriptorSet 0',
        ).replace(
            f" {base} %zero",
            f" {copy} %zero",
        ).replace(
            "%foreign_zero = OpInBoundsAccessChain",
            f"{copy} = {op} %{resource}_ptr {base}\n%foreign_zero = OpInBoundsAccessChain",
        )
        base = copy
    return assembly


def assert_descriptor_aliases(text, resource, binding, operations):
    # OpName keeps these IDs stable after the CLI's spirv-dis round trip.
    definitions = {parts[0]: parts[2:] for parts in map(str.split, text.splitlines())
                   if len(parts) > 2 and parts[1] == "="}
    base = f"%{resource}"
    local_type = definitions[base][1]
    array = definitions[definitions[local_type][2]]
    assert array[0] == "OpTypeArray"
    assert f"OpDecorate {base} Binding {m.metal_ids(m.capacities())[binding]}" in text
    for i, op in enumerate(operations):
        copy = f"%copy{i}"
        assert definitions[copy] == [op, local_type, base]
        base = copy
    indexed = definitions["%sampler_access" if resource == "samplers" else "%image_access"]
    assert indexed[0] == "OpAccessChain"
    assert indexed[2] == base
    assert definitions[indexed[1]] == ["OpTypePointer", "UniformConstant", array[1]]
    foreign_type = definitions["%foreign_samplers"][1]
    assert definitions[definitions[foreign_type][2]][0] == "OpTypeRuntimeArray"
    assert definitions["%foreign_zero"] == ["OpInBoundsAccessChain", foreign_type, "%foreign_samplers"]
    assert definitions["%foreign_copy"] == ["OpCopyObject", foreign_type, "%foreign_zero"]
    assert definitions["%sampler_access"][1] == definitions["%foreign_access"][1]


@pytest.mark.parametrize("resource,binding,operations", ALIAS_CASES)
def test_rewrite_retypes_only_pointer_preserving_descriptor_aliases(resource, binding, operations):
    assembly = descriptor_alias_fragment(resource, binding, operations)
    assert_descriptor_aliases(m.rewrite(assembly, m.capacities()), resource, binding, operations)


@pytest.mark.skipif(not all(shutil.which(tool) for tool in ("spirv-as", "spirv-dis", "spirv-val")),
                    reason="needs SPIRV-Tools on PATH")
@pytest.mark.parametrize("resource,binding,operations", ALIAS_CASES)
def test_cli_retypes_descriptor_array_aliases(tmp_path, resource, binding, operations):
    assembly = descriptor_alias_fragment(resource, binding, operations)
    src, dst = tmp_path / "input.spv", tmp_path / "metal.spv"
    target = ["--target-env", "vulkan1.2"]
    subprocess.run(["spirv-as", *target, "-", "-o", str(src)], input=assembly, text=True, check=True)
    subprocess.run(["spirv-val", *target, str(src)], check=True)
    subprocess.run([sys.executable, "-I", m.__file__, str(src), str(dst)], check=True)
    subprocess.run(["spirv-val", *target, str(dst)], check=True)
    text = subprocess.check_output(["spirv-dis", str(dst)], text=True)
    assert_descriptor_aliases(text, resource, binding, operations)

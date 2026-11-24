"""Split the #670 shared comparison/ordinary sampler, compile it and check the GPU readback.

Usage: compare_sampler.py <spirv-as> <spirv-val> <spirv-cross> [<dxc> <offscreen_check> hardware|warp]

Without the last three arguments only the SPIR-V and HLSL stages and the rejection controls run.
"""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from spirv_sampler_split import split_comparison_sampler
from spirv_sampler_views import split_sampler_views

def substitute(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


ORDINARY_BINDING = 5
SHARED = Path(__file__).with_name("shared_sampler.spvasm").read_text(encoding="utf-8")
# The same module with arrayed (Texture2DArray) images and a float3 coordinate (u, v, layer 0).
SHARED_ARRAY = Path(__file__).with_name("shared_sampler_array.spvasm").read_text(encoding="utf-8")
# The arrayed module reading layer 1 instead of layer 0 (the coordinate's third component becomes 1).
SHARED_ARRAY_LAYER1 = SHARED_ARRAY.replace(
    "%15 = OpConstantComposite %v3float %float_0_25 %float_0_25 %float_0",
    "%float_1 = OpConstant %float 1\n         %15 = OpConstantComposite %v3float %float_0_25 %float_0_25 %float_1")
# The base module with valueTexture as a Texture3D: the depth texture stays 2D (comparison on 3D does not
# exist), and the ordinary sample reads a float3 coordinate (0.25, 0.25, 0.75), the second of two slices.
SHARED_VOLUME = SHARED
for old, new in (
        ("%type_sampled_image = OpTypeSampledImage %type_2d_image\n",
         "%type_sampled_image = OpTypeSampledImage %type_2d_image\n"
         "%type_3d_image = OpTypeImage %float 3D 2 0 0 1 Unknown\n"
         "%_ptr_UniformConstant_type_3d_image = OpTypePointer UniformConstant %type_3d_image\n"
         "%type_sampled_image_3d = OpTypeSampledImage %type_3d_image\n"
         "%v3float = OpTypeVector %float 3\n"
         "%uv3 = OpConstantComposite %v3float %float_0_25 %float_0_25 %float_0_75\n"),
        ("%valueTexture = OpVariable %_ptr_UniformConstant_type_2d_image UniformConstant",
         "%valueTexture = OpVariable %_ptr_UniformConstant_type_3d_image UniformConstant"),
        ("%40 = OpLoad %type_2d_image %valueTexture", "%40 = OpLoad %type_3d_image %valueTexture"),
        ("%42 = OpSampledImage %type_sampled_image %40 %41", "%42 = OpSampledImage %type_sampled_image_3d %40 %41"),
        ("%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0",
         "%43 = OpImageSampleExplicitLod %v4float %42 %uv3 Lod %float_0")):
    SHARED_VOLUME = substitute(SHARED_VOLUME, old, new)
# Slice 1 of the 3D texture holds 0.3125 at its top-left texel (slice 0 holds 0.375, as the 2D case does).
EXPECTED_VOLUME = [0x3f800000, 0x00000000, 0x3ea00000] + [0xc5c5c5c5] * 13
# Two mixed samplers (cmpA at binding 2, cmpB at 6): word 0 SampleCmp(cmpA, ref 0.25 vs 0.5) = 1, word 1
# SampleCmp(cmpB, ref 0.875 vs 0.75) = 0, word 2 ordinary A 0.375, word 3 ordinary B 0.875; 12 guards.
SHARED_MULTI = Path(__file__).with_name("shared_sampler_multi.spvasm").read_text(encoding="utf-8")
MULTI_BINDINGS = [5, 7]
EXPECTED_MULTI = [0x3f800000, 0x00000000, 0x3ec00000, 0x3f600000] + [0xc5c5c5c5] * 12
# 0.25 <= 0.5 passes, 0.75 <= 0.5 fails, ordinary sample of 0.375; the other 13 words are guards.
EXPECTED = [0x3f800000, 0x00000000, 0x3ec00000] + [0xc5c5c5c5] * 13
# Layer 1 of the arrayed textures: 0.25 <= 0.875 and 0.75 <= 0.875 both pass, ordinary sample of 0.3125.
EXPECTED_LAYER1 = [0x3f800000, 0x3f800000, 0x3ea00000] + [0xc5c5c5c5] * 13


def rejected(assembly, binding, reason):
    try:
        split_comparison_sampler(assembly, binding)
    except ValueError as error:
        assert reason in str(error), error
    else:
        raise AssertionError(f"accepted invalid input: {reason}")


def multi_controls(directory):
    lowered, mappings = split_comparison_sampler(SHARED_MULTI, MULTI_BINDINGS)
    assert mappings == [
        {"comparison_sampler": "%cmpA", "plain_sampler": "%x2w_plain_sampler_0", "descriptor_set": 0,
         "comparison_binding": 2, "ordinary_binding": 5},
        {"comparison_sampler": "%cmpB", "plain_sampler": "%x2w_plain_sampler_1", "descriptor_set": 0,
         "comparison_binding": 6, "ordinary_binding": 7}], mappings
    lines = lowered.splitlines()
    # Each lowered sampler is its own variable at its own binding, listed in the interface, and each
    # ordinary sample loads its own variable; the comparison loads are untouched.
    for plain, binding, comparison in (("%x2w_plain_sampler_0", 5, "%cmpA"), ("%x2w_plain_sampler_1", 7, "%cmpB")):
        assert f"{plain} = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant" in lines
        assert f"OpDecorate {plain} DescriptorSet 0" in lines and f"OpDecorate {plain} Binding {binding}" in lines
        assert f"{plain}_load_0 = OpLoad %type_sampler {plain}" in lines
        assert plain in next(line for line in lines if line.startswith("OpEntryPoint")).split()
        assert sum(line.endswith(f"%type_sampler {comparison}") for line in lines) == 2, comparison
    # Declaration order decides which sampler takes which binding.
    swapped, swapped_mappings = split_comparison_sampler(SHARED_MULTI, [7, 5])
    assert [m["ordinary_binding"] for m in swapped_mappings] == [7, 5]
    assert "OpDecorate %x2w_plain_sampler_0 Binding 7" in swapped.splitlines()
    # A one-element sequence uses the multi-sampler names; a plain int keeps the single-sampler names.
    single_list, single_mappings = split_comparison_sampler(SHARED, [ORDINARY_BINDING])
    assert single_mappings[0]["plain_sampler"] == "%x2w_plain_sampler_0"
    assert "%x2w_plain_sampler_0_load_0 = OpLoad %type_sampler %x2w_plain_sampler_0" in single_list.splitlines()
    # Free bindings below, between and above the occupied ones are accepted (3 is free in this module).
    assert [m["ordinary_binding"] for m in split_comparison_sampler(SHARED_MULTI, [3, 8])[1]] == [3, 8]

    # Rejections: each must fail with its own message and leave the input module untouched.
    rejected(SHARED_MULTI, 5, "found 2 samplers with mixed uses and was given 1 ordinary bindings")
    rejected(SHARED_MULTI, [5], "found 2 samplers with mixed uses and was given 1 ordinary bindings")
    rejected(SHARED_MULTI, [5, 7, 8], "found 2 samplers with mixed uses and was given 3 ordinary bindings")
    rejected(SHARED, [5, 7], "found 1 samplers with mixed uses and was given 2 ordinary bindings")
    rejected(SHARED_MULTI, [5, 5], "Ordinary sampler bindings must be distinct")
    rejected(SHARED_MULTI, [5, -1], "Ordinary sampler binding must be nonnegative")
    # Collisions with each existing variable's binding, for either sampler (0, 1, 2, 4, 6 are taken).
    for binding in (0, 1, 2, 4, 6):
        rejected(SHARED_MULTI, [binding, 7], f"binding {binding} is occupied in set 0")
        rejected(SHARED_MULTI, [5, binding], f"binding {binding} is occupied in set 0")
    # Reserved IDs: the multi-sampler names are reserved too, in any position.
    for name in ("%x2w_plain_sampler_1", "%x2w_plain_sampler_1_load_0", "%x2w_plain_sampler"):
        rejected(substitute(SHARED_MULTI, "%float_0 = OpConstant %float 0",
                            f"%float_0 = OpConstant %float 0\n{name} = OpConstant %float 0"),
                 MULTI_BINDINGS, "Reserved sampler split ID is already in use")
    # The narrow profile applies to every mixed sampler: a nonzero LOD on cmpB's ordinary sample is refused.
    rejected(substitute(SHARED_MULTI, "%49 = OpImageSampleExplicitLod %v4float %48 %uv_b Lod %float_0",
                        "%49 = OpImageSampleExplicitLod %v4float %48 %uv_b Lod %float_0_25"),
             MULTI_BINDINGS, "requires a constant LOD of zero")
    rejected(substitute(SHARED_MULTI, "%cmpA %cmpB %results", "%cmpA %results"),
             MULTI_BINDINGS, "%cmpB: sampler must be listed in the entry-point interface")
    # Occupancy is per descriptor set: binding 6 is taken in set 0 but free for a sampler moved to set 1.
    other_set = substitute(SHARED_MULTI, "OpDecorate %cmpB DescriptorSet 0", "OpDecorate %cmpB DescriptorSet 1")
    moved = split_comparison_sampler(other_set, [6, 2])[1]
    assert [(m["descriptor_set"], m["ordinary_binding"]) for m in moved] == [(0, 6), (1, 2)], moved
    rejected(other_set, [2, 7], "binding 2 is occupied in set 0")
    rejected(other_set, [5, 6], "binding 6 is occupied in set 1 by %cmpB")
    # Unmixed samplers in the same module are left alone: one mixed sampler plus one comparison-only sampler.
    comparison_only = substitute(SHARED_MULTI, "%49 = OpImageSampleExplicitLod %v4float %48 %uv_b Lod %float_0",
                                 "%49 = OpImageSampleDrefExplicitLod %float %48 %uv_b %float_0_875 Lod %float_0")
    one_text, one = split_comparison_sampler(comparison_only, [5])
    assert [m["comparison_sampler"] for m in one] == ["%cmpA"] and "%x2w_plain_sampler_1" not in one_text, one
    assert split_comparison_sampler(comparison_only, 5)[1]["ordinary_binding"] == 5

    tool = [sys.executable, "-B", str(ROOT / "tools" / "spirv_sampler_split.py")]
    source = directory / "shared_multi.spvasm"
    source.write_text(SHARED_MULTI, encoding="utf-8")
    target = directory / "lowered_multi_cli.spvasm"
    run = subprocess.run([*tool, source, target, "--ordinary-binding", "5", "--ordinary-binding", "7"],
                         capture_output=True, text=True, check=True)
    assert run.stdout.splitlines() == [
        "comparison_sampler=%cmpA plain_sampler=%x2w_plain_sampler_0 descriptor_set=0 comparison_binding=2 "
        "ordinary_binding=5",
        "comparison_sampler=%cmpB plain_sampler=%x2w_plain_sampler_1 descriptor_set=0 comparison_binding=6 "
        "ordinary_binding=7"], run.stdout
    assert target.read_text(encoding="utf-8") == lowered
    for arguments, reason in ((["--ordinary-binding", "5"], "found 2 samplers"),
                              (["--ordinary-binding", "5", "--ordinary-binding", "5"], "must be distinct"),
                              (["--ordinary-binding", "5", "--ordinary-binding", "6"], "binding 6 is occupied")):
        failed = subprocess.run([*tool, source, directory / "refused.spvasm", *arguments], capture_output=True, text=True)
        assert failed.returncode == 1 and reason in failed.stderr, (arguments, failed)
        assert not (directory / "refused.spvasm").exists()
    return lowered, swapped


def controls(directory):
    lowered, mapping = split_comparison_sampler(SHARED, ORDINARY_BINDING)
    assert mapping == {"descriptor_set": 0, "comparison_binding": 2, "ordinary_binding": ORDINARY_BINDING}, mapping
    # A Texture3D value image with a float3 coordinate splits like the 2D module: same mapping, only the
    # ordinary load moves, the comparison sampler keeps its two loads and the 2D depth texture is untouched.
    volume_lowered, volume_mapping = split_comparison_sampler(SHARED_VOLUME, ORDINARY_BINDING)
    assert volume_mapping == mapping, volume_mapping
    assert "%x2w_plain_sampler_0 = OpLoad %type_sampler %x2w_plain_sampler" in volume_lowered.splitlines()
    assert "OpTypeImage %float 3D 2 0 0 1 Unknown" in volume_lowered
    # The arrayed variant splits to the same mapping and only the ordinary load moves.
    array_lowered, array_mapping = split_comparison_sampler(SHARED_ARRAY, ORDINARY_BINDING)
    assert array_mapping == mapping, array_mapping
    assert "%x2w_plain_sampler_0 = OpLoad %type_sampler %x2w_plain_sampler" in array_lowered.splitlines()
    assert "OpTypeImage %float 2D 2 1 0 1 Unknown" in array_lowered
    # Modules without a mixed standalone sampler come back byte-for-byte, whatever else they contain.
    separated = substitute(SHARED, "%41 = OpLoad %type_sampler %cmpSampler", "%41 = OpLoad %type_sampler %plainSampler")
    separated_lod = substitute(separated, "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0",
                               "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0_25")
    table = Path(__file__).with_name("sampler.spvasm").read_text(encoding="utf-8")  # #676 indexed table
    for name, module in (("already separated", separated), ("unrelated nonzero LOD", separated_lod),
                         ("indexed sampler table", table), ("split sampler views", split_sampler_views(table))):
        assert split_comparison_sampler(module, ORDINARY_BINDING) == (module, None), name

    # An unrelated ordinary sample at nonzero LOD on another sampler does not block the split.
    unrelated = substitute(SHARED, "%44 = OpCompositeExtract %float %43 0",
                           "%44 = OpCompositeExtract %float %43 0\n"
                           "%u1 = OpLoad %type_sampler %plainSampler\n"
                           "%u2 = OpSampledImage %type_sampled_image %40 %u1\n"
                           "%u3 = OpImageSampleExplicitLod %v4float %u2 %15 Lod %float_0_25")
    split, mapping = split_comparison_sampler(unrelated, ORDINARY_BINDING)
    assert mapping["ordinary_binding"] == ORDINARY_BINDING
    assert "%u2 = OpSampledImage %type_sampled_image %40 %u1" in split

    # Unsupported uses of the sampler being split are still rejected.
    for binding in (2, 3, 4):
        rejected(SHARED, binding, f"binding {binding} is occupied in set 0")
    # A binding declared without a DescriptorSet decoration still occupies set 0.
    rejected(substitute(SHARED, "OpDecorate %plainSampler DescriptorSet 0\n", ""), 3,
             "binding 3 is occupied in set 0 by %plainSampler")
    # The ordinary sample may not reuse the comparison's sampled image.
    rejected(substitute(SHARED, "%43 = OpImageSampleExplicitLod %v4float %42 ",
                        "%43 = OpImageSampleExplicitLod %v4float %34 "),
             ORDINARY_BINDING, "one sampled image is used for both sampling kinds")
    rejected(substitute(SHARED, "%float_0_75 Lod %float_0", "%float_0_75 Lod %float_0_25"),
             ORDINARY_BINDING, "requires a constant LOD of zero")
    rejected(substitute(SHARED, "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0",
                        "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0_25"),
             ORDINARY_BINDING, "requires a constant LOD of zero")
    rejected(substitute(SHARED, "%33 = OpLoad %type_sampler %cmpSampler",
                        "%copy = OpCopyObject %_ptr_UniformConstant_type_sampler %cmpSampler\n"
                        "%33 = OpLoad %type_sampler %cmpSampler"),
             ORDINARY_BINDING, "only direct loads are supported")
    rejected(substitute(SHARED, "%33 = OpLoad %type_sampler %cmpSampler",
                        "%33 = OpLoad %type_sampler %cmpSampler\n"
                        "%loaded_copy = OpCopyObject %type_sampler %33"),
             ORDINARY_BINDING, "loaded sampler feeds OpCopyObject")

    # The remaining profile diagnostics.
    rejected(SHARED, -1, "Ordinary sampler binding must be nonnegative")
    rejected(substitute(SHARED, "OpEntryPoint GLCompute", "OpEntryPoint Fragment"),
             ORDINARY_BINDING, "requires a single GLCompute entry point")
    # Multisampled images, and 3D images that are multisampled or arrayed, stay outside the profile.
    profile = ("only float32 Texture1D, Texture2D, TextureCube and non-arrayed Texture3D sampling "
               "(including the array forms of the first three) is supported")
    for image in ("OpTypeImage %float 2D 2 0 1 1 Unknown",):
        rejected(substitute(SHARED, "OpTypeImage %float 2D 2 0 0 1 Unknown", image),
                 ORDINARY_BINDING, profile)
    for image in ("OpTypeImage %float 3D 2 0 1 1 Unknown", "OpTypeImage %float 3D 2 1 0 1 Unknown"):
        rejected(substitute(SHARED_VOLUME, "OpTypeImage %float 3D 2 0 0 1 Unknown", image),
                 ORDINARY_BINDING, profile)
    # Depth comparison on a 3D image is refused (the depth texture becomes 3D here), as is a nonzero LOD.
    rejected(substitute(SHARED, "OpTypeImage %float 2D 2 0 0 1 Unknown", "OpTypeImage %float 3D 2 0 0 1 Unknown"),
             ORDINARY_BINDING, "depth-comparison sampling is not supported on Texture3D")
    rejected(substitute(SHARED_VOLUME, "%43 = OpImageSampleExplicitLod %v4float %42 %uv3 Lod %float_0",
                        "%43 = OpImageSampleExplicitLod %v4float %42 %uv3 Lod %float_0_25"),
             ORDINARY_BINDING, "requires a constant LOD of zero")
    rejected(substitute(SHARED_ARRAY, "OpTypeImage %float 2D 2 1 0 1 Unknown",
                        "OpTypeImage %float 2D 2 1 1 1 Unknown"),
             ORDINARY_BINDING, profile)
    rejected(substitute(SHARED, " %cmpSampler %plainSampler %results", " %plainSampler %results"),
             ORDINARY_BINDING, "sampler must be listed in the entry-point interface")
    rejected(substitute(SHARED, "%float_0 = OpConstant %float 0",
                        "%float_0 = OpConstant %float 0\n"
                        "%x2w_plain_sampler_0 = OpConstant %float 0"),
             ORDINARY_BINDING, "Reserved sampler split ID is already in use")

    # Occupancy is per descriptor set: binding 3 is taken in set 0 and free in set 1.
    other_set = substitute(SHARED, "OpDecorate %cmpSampler DescriptorSet 0",
                           "OpDecorate %cmpSampler DescriptorSet 1")
    other_lowered, other_mapping = split_comparison_sampler(other_set, 3)
    assert other_mapping == {"descriptor_set": 1, "comparison_binding": 2, "ordinary_binding": 3}, other_mapping
    assert "OpDecorate %x2w_plain_sampler DescriptorSet 1" in other_lowered.splitlines()
    assert "OpDecorate %x2w_plain_sampler Binding 3" in other_lowered.splitlines()

    tool = [sys.executable, "-B", str(ROOT / "tools" / "spirv_sampler_split.py")]
    source = directory / "shared.spvasm"
    source.write_text(SHARED, encoding="utf-8")
    target = directory / "lowered.spvasm"
    run = subprocess.run([*tool, source, target, "--ordinary-binding", str(ORDINARY_BINDING)],
                         capture_output=True, text=True, check=True)
    assert run.stdout.split() == ["descriptor_set=0", "comparison_binding=2", f"ordinary_binding={ORDINARY_BINDING}"]
    assert target.read_text(encoding="utf-8") == lowered
    occupied = subprocess.run([*tool, source, directory / "occupied.spvasm", "--ordinary-binding", "3"],
                              capture_output=True, text=True)
    assert occupied.returncode == 1 and "occupied" in occupied.stderr, occupied
    assert not (directory / "occupied.spvasm").exists()
    return lowered


def emit(directory, assembler, validator, cross):
    lowered = controls(directory)
    multi_lowered, multi_swapped = multi_controls(directory)
    outputs = {}
    array_lowered = split_comparison_sampler(SHARED_ARRAY, ORDINARY_BINDING)[0]
    assert SHARED_ARRAY_LAYER1 != SHARED_ARRAY
    layer1_lowered, layer1_mapping = split_comparison_sampler(SHARED_ARRAY_LAYER1, ORDINARY_BINDING)
    assert layer1_mapping["ordinary_binding"] == ORDINARY_BINDING, layer1_mapping
    volume_lowered = split_comparison_sampler(SHARED_VOLUME, ORDINARY_BINDING)[0]
    for name, assembly in (("shared", SHARED), ("lowered", lowered),
                           ("shared_volume", SHARED_VOLUME), ("lowered_volume", volume_lowered),
                           ("shared_array", SHARED_ARRAY), ("lowered_array", array_lowered),
                           ("lowered_array_layer1", layer1_lowered),
                           ("shared_multi", SHARED_MULTI), ("lowered_multi", multi_lowered),
                           ("lowered_multi_swapped", multi_swapped)):
        text = directory / f"{name}.spvasm"
        text.write_text(assembly, encoding="utf-8")
        module = directory / f"{name}.spv"
        subprocess.run([assembler, "--target-env", "vulkan1.2", text, "-o", module], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
        hlsl = directory / f"{name}.hlsl"
        subprocess.run([cross, module, "--hlsl", "--shader-model", "60", "--output", hlsl], check=True)
        outputs[name] = hlsl
    source = outputs["lowered"].read_text(encoding="utf-8")
    comparison = re.search(r"SamplerComparisonState\s+(\w+)\s*:\s*register\(s2\b", source)
    ordinary = re.search(rf"SamplerState\s+(\w+)\s*:\s*register\(s{ORDINARY_BINDING}\b", source)
    assert comparison and ordinary, source
    assert source.count(f"SampleCmpLevelZero({comparison[1]},") == 2, source
    assert source.count(f"SampleLevel({ordinary[1]},") == 1, source
    array_source = outputs["lowered_array"].read_text(encoding="utf-8")
    comparison = re.search(r"SamplerComparisonState\s+(\w+)\s*:\s*register\(s2\b", array_source)
    ordinary = re.search(rf"SamplerState\s+(\w+)\s*:\s*register\(s{ORDINARY_BINDING}\b", array_source)
    assert comparison and ordinary and "Texture2DArray" in array_source, array_source
    assert array_source.count(f"SampleCmpLevelZero({comparison[1]},") == 2, array_source
    assert array_source.count(f"SampleLevel({ordinary[1]},") == 1, array_source
    volume_source = outputs["lowered_volume"].read_text(encoding="utf-8")
    comparison = re.search(r"SamplerComparisonState\s+(\w+)\s*:\s*register\(s2\b", volume_source)
    ordinary = re.search(rf"SamplerState\s+(\w+)\s*:\s*register\(s{ORDINARY_BINDING}\b", volume_source)
    assert comparison and ordinary and "Texture3D<" in volume_source, volume_source
    assert volume_source.count(f"SampleCmpLevelZero({comparison[1]},") == 2, volume_source
    assert volume_source.count(f"SampleLevel({ordinary[1]}, float3(0.25f, 0.25f, 0.75f)") == 1, volume_source
    multi_source = outputs["lowered_multi"].read_text(encoding="utf-8")
    for register, kind in ((2, "SamplerComparisonState"), (6, "SamplerComparisonState"), (5, "SamplerState"),
                           (7, "SamplerState")):
        assert re.search(rf"{kind}\s+\w+\s*:\s*register\(s{register}\b", multi_source), (register, multi_source)
    assert len(re.findall(r"SamplerState\s+\w+\s*:\s*register", multi_source)) == 2, multi_source
    assert multi_source.count("SampleCmpLevelZero(") == 2 and multi_source.count("SampleLevel(") == 2, multi_source
    print(f"shared and lowered SPIR-V validated; comparison s2, ordinary s{ORDINARY_BINDING}; controls passed",
          flush=True)
    return outputs


def compile_dxil(directory, dxc, outputs):
    # The unsplit module reproduces the original DXC rejection.
    shared = subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", directory / "shared.dxil", outputs["shared"]],
                            capture_output=True, text=True)
    if shared.returncode == 0 or "no matching member function for call to 'SampleLevel'" not in shared.stderr:
        raise AssertionError(f"Expected the shared-sampler DXC diagnostic\n{shared.stdout}\n{shared.stderr}")
    print("shared-sampler HLSL: DXC rejected SampleLevel as expected", flush=True)
    # The unsplit arrayed module is rejected the same way.
    shared = subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", directory / "shared_array.dxil",
                             outputs["shared_array"]], capture_output=True, text=True)
    if shared.returncode == 0 or "no matching member function for call to 'SampleLevel'" not in shared.stderr:
        raise AssertionError(f"Expected the arrayed shared-sampler DXC diagnostic\n{shared.stdout}\n{shared.stderr}")
    print("arrayed shared-sampler HLSL: DXC rejected SampleLevel as expected", flush=True)
    shared = subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", directory / "shared_multi.dxil",
                             outputs["shared_multi"]], capture_output=True, text=True)
    if shared.returncode == 0 or "no matching member function for call to 'SampleLevel'" not in shared.stderr:
        raise AssertionError(f"Expected the two-sampler shared DXC diagnostic\n{shared.stdout}\n{shared.stderr}")
    print("two-sampler shared HLSL: DXC rejected SampleLevel as expected", flush=True)
    shared = subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", directory / "shared_volume.dxil",
                             outputs["shared_volume"]], capture_output=True, text=True)
    if shared.returncode == 0 or "no matching member function for call to 'SampleLevel'" not in shared.stderr:
        raise AssertionError(f"Expected the Texture3D shared-sampler DXC diagnostic\n{shared.stdout}\n{shared.stderr}")
    print("Texture3D shared-sampler HLSL: DXC rejected SampleLevel as expected", flush=True)
    dxil = {}
    for name in ("lowered", "lowered_volume", "lowered_array", "lowered_array_layer1", "lowered_multi", "lowered_multi_swapped"):
        dxil[name] = directory / f"{name}.dxil"
        subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", dxil[name], outputs[name]], check=True)
    print("DXC compiled the lowered module and both arrayed lowered modules (layers 0 and 1)", flush=True)
    return dxil


def with_words(base=None, **words):
    reference = (base or EXPECTED).copy()
    for index, value in words.items():
        reference[int(index[1:])] = value
    return reference


# Texture3D value texture (two 2x2 slices, mip 1 of 0.625): slice 1's top-left is 0.3125, slice 0's 0.375.
VOLUME_DRAWS = [
    ("volume slice 1", "lowered_volume", "volume", EXPECTED_VOLUME, []),
    # The slice 0 value in the reference: only the ordinary word differs.
    ("volume slice 0 reference", "lowered_volume", "volume", EXPECTED, ["word=2 expected=3ec00000 actual=3ea00000"]),
    ("volume comparison descriptor", "lowered_volume", "volume-comparison-greater",
     with_words(EXPECTED_VOLUME, w0=0x00000000, w1=0x3f800000), []),
    ("volume ordinary descriptor vs baseline", "lowered_volume", "volume-ordinary-min-lod1", EXPECTED_VOLUME,
     ["word=2 expected=3ea00000 actual=3f200000"]),
    ("volume ordinary descriptor", "lowered_volume", "volume-ordinary-min-lod1",
     with_words(EXPECTED_VOLUME, w2=0x3f200000), []),
]

MULTI = "lowered_multi"
# (name, module, descriptor variant, reference, mismatching words reported on stderr); the two-sampler
# cases pass both ordinary registers to the host.
MULTI_DRAWS = [
    ("multi baseline", MULTI, "multi", EXPECTED_MULTI, []),
    ("multi wrong reference", MULTI, "multi", with_words(EXPECTED_MULTI, w1=0x3f800000),
     ["word=1 expected=3f800000 actual=00000000"]),
    # One static sampler changes at a time; each moves only the word it drives.
    ("multi comparison A vs baseline", MULTI, "multi-comparison-a-greater", EXPECTED_MULTI,
     ["word=0 expected=3f800000 actual=00000000"]),
    ("multi comparison A", MULTI, "multi-comparison-a-greater", with_words(EXPECTED_MULTI, w0=0x00000000), []),
    ("multi comparison B vs baseline", MULTI, "multi-comparison-b-greater", EXPECTED_MULTI,
     ["word=1 expected=00000000 actual=3f800000"]),
    ("multi comparison B", MULTI, "multi-comparison-b-greater", with_words(EXPECTED_MULTI, w1=0x3f800000), []),
    ("multi ordinary A vs baseline", MULTI, "multi-ordinary-a-min-lod1", EXPECTED_MULTI,
     ["word=2 expected=3ec00000 actual=3f200000"]),
    ("multi ordinary A", MULTI, "multi-ordinary-a-min-lod1", with_words(EXPECTED_MULTI, w2=0x3f200000), []),
    ("multi ordinary B vs baseline", MULTI, "multi-ordinary-b-min-lod1", EXPECTED_MULTI,
     ["word=3 expected=3f600000 actual=3f200000"]),
    ("multi ordinary B", MULTI, "multi-ordinary-b-min-lod1", with_words(EXPECTED_MULTI, w3=0x3f200000), []),
    # Bindings given in the other order: register 5 now serves cmpB's ordinary sample, so clamping the
    # first register changes word 3, not word 2.
    ("multi swapped bindings", "lowered_multi_swapped", "multi-ordinary-a-min-lod1",
     with_words(EXPECTED_MULTI, w3=0x3f200000), []),
    ("multi swapped bindings vs baseline", "lowered_multi_swapped", "multi-ordinary-a-min-lod1", EXPECTED_MULTI,
     ["word=3 expected=3f600000 actual=3f200000"]),
]


# (name, module, descriptor variant, reference, mismatching words reported on stderr)
DRAWS = [
    ("baseline", "lowered", "baseline", EXPECTED, []),
    # Wrong reference: claim the failing comparison passed.
    ("wrong reference", "lowered", "baseline", with_words(w1=0x3f800000),
     ["word=1 expected=3f800000 actual=00000000"]),
    # s2 switched to GREATER: only the two comparison words change, against the baseline reference.
    ("comparison descriptor vs baseline", "lowered", "comparison-greater", EXPECTED,
     ["word=0 expected=3f800000 actual=00000000", "word=1 expected=00000000 actual=3f800000"]),
    ("comparison descriptor", "lowered", "comparison-greater", with_words(w0=0x00000000, w1=0x3f800000), []),
    # Ordinary sampler clamped to MinLOD 1: only the ordinary word changes, to t1's 1x1 mip (0.625).
    ("ordinary descriptor vs baseline", "lowered", "ordinary-min-lod1", EXPECTED,
     ["word=2 expected=3ec00000 actual=3f200000"]),
    ("ordinary descriptor", "lowered", "ordinary-min-lod1", with_words(w2=0x3f200000), []),
    # Texture2DArray: two-layer t0 and t1 whose layer 1 differs from layer 0 in each sampled texel.
    ("array layer 0", "lowered_array", "array", EXPECTED, []),
    ("array layer 1", "lowered_array_layer1", "array", EXPECTED_LAYER1, []),
    # Layer 1 against layer 0's reference: the second comparison and the ordinary sample change.
    ("array layer 1 vs layer 0", "lowered_array_layer1", "array", EXPECTED,
     ["word=1 expected=00000000 actual=3f800000", "word=2 expected=3ec00000 actual=3ea00000"]),
    # On layer 1, s2 switched to GREATER changes only the comparison words (0.875 is not below 0.25 or 0.75).
    ("array comparison descriptor vs layer 1", "lowered_array_layer1", "array-comparison-greater", EXPECTED_LAYER1,
     ["word=0 expected=3f800000 actual=00000000", "word=1 expected=3f800000 actual=00000000"]),
    ("array comparison descriptor", "lowered_array_layer1", "array-comparison-greater",
     with_words(EXPECTED_LAYER1, w0=0x00000000, w1=0x00000000), []),
    # On layer 1, the ordinary sampler at MinLOD 1 reads layer 1's 1x1 mip (0.5625), not layer 0's (0.625).
    ("array ordinary descriptor vs layer 1", "lowered_array_layer1", "array-ordinary-min-lod1", EXPECTED_LAYER1,
     ["word=2 expected=3ea00000 actual=3f100000"]),
    ("array ordinary descriptor", "lowered_array_layer1", "array-ordinary-min-lod1",
     with_words(EXPECTED_LAYER1, w2=0x3f100000), []),
]


def draw(directory, dxil, renderer, adapter):
    reference = directory / "reference.bin"
    for name, module, variant, expected, mismatches in [*DRAWS, *VOLUME_DRAWS, *MULTI_DRAWS]:
        registers = ",".join(map(str, MULTI_BINDINGS)) if variant.startswith("multi") else str(ORDINARY_BINDING)
        reference.write_bytes(struct.pack("<16I", *expected))
        run = subprocess.run([renderer, adapter, "--spirv-compare-sampler", dxil[module], registers,
                              reference, variant], capture_output=True, text=True)
        print(f"[{name}]", run.stdout, run.stderr, flush=True)
        if run.returncode == 77 and name == "baseline":
            return 77
        reported = [line for line in run.stderr.splitlines() if line.startswith("word=")]
        if (run.returncode != int(bool(mismatches)) or reported != mismatches
                or f"descriptors={variant} " not in run.stdout
                or f"compared_words=16 mismatches={len(mismatches)}" not in run.stdout):
            raise RuntimeError(f"Unexpected {name} readback")
    print("wrong reference fails only at word 1; s2 changes only words 0-1; "
          f"s{ORDINARY_BINDING} changes only word 2", flush=True)
    print("Texture2DArray: layer 0 and layer 1 each read their own texels (layer 1 vs layer 0 differs at words 1-2); "
          f"on layer 1, s2 changes only words 0-1 and s{ORDINARY_BINDING} only word 2 (layer 1's mip 1)", flush=True)
    print("Texture3D: the ordinary sample reads slice 1 (a slice 0 reference differs at word 2 only); "
          "s2 changes only words 0-1 and the ordinary sampler's MinLOD 1 only word 2", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (4, 7):
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        outputs = emit(directory, *sys.argv[1:4])
        if len(sys.argv) == 7:
            dxil = compile_dxil(directory, sys.argv[4], outputs)
            sys.exit(draw(directory, dxil, *sys.argv[5:]))

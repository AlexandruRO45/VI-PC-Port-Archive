"""Validate the original and split sampler module, then optionally run the D3D12 readback."""
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from spirv_sampler_views import split_sampler_views


def check_sampler_rewrite(source):
    # Use the fragment variant: its ordinary sample is the implicit-LOD form.
    sample = "%raw = OpImageSampleImplicitLod %vec4 %si %uv"
    comparison = "%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod"
    second = comparison.replace("%cmp =", "%cmp2 =")
    depth_gather = "%cmp = OpImageDrefGather %vec4 %si %uv %half"
    ordinary_gather = "%raw = OpImageGather %vec4 %si %uv %zero"
    for variant, ordinary, comparisons in (
        (source.replace(comparison, comparison + "\n" + second), sample, (comparison, second)),
        (source.replace(comparison, depth_gather).replace(
            "%cw = OpBitcast %uint %cmp",
            "%depth = OpCompositeExtract %float %cmp 0\n%cw = OpBitcast %uint %depth"),
         sample, (depth_gather,)),
        (source.replace(sample, ordinary_gather), ordinary_gather, (comparison,)),
    ):
        original_lines = variant.splitlines()
        rewritten_lines = split_sampler_views(variant).splitlines()
        assert ordinary in rewritten_lines, ordinary
        for instruction in comparisons:
            name = f"%bridge_cmp_{original_lines.index(instruction)}"
            assert f"{name}_ptr = OpAccessChain %_ptr_UC_smp %bridge_cmp %i" in rewritten_lines, name
            expected = instruction.replace(" %si ", f" {name}_image ")
            assert expected in rewritten_lines, expected
            assert instruction not in rewritten_lines, instruction

    binding = "OpDecorate %samplers Binding 1"
    table = "%samplers = OpVariable %_ptr_UC_rt_smp UniformConstant"
    foreign = source.replace(' "main" %texture %samplers %out',
                             ' "main" %texture %samplers %out %foreign')
    foreign = foreign.replace(binding, binding + "\nOpDecorate %foreign DescriptorSet 0\n"
                              "OpDecorate %foreign Binding 2")
    foreign = foreign.replace(table, table + "\n%foreign = OpVariable %_ptr_UC_rt_smp UniformConstant")
    foreign = foreign.replace(comparison,
                              "%foreign_p = OpAccessChain %_ptr_UC_smp %foreign %i\n"
                              "%foreign_s = OpLoad %smp %foreign_p\n"
                              "%foreign_si = OpSampledImage %pair %im %foreign_s\n"
                              + comparison.replace(" %si ", " %foreign_si "))
    for unsupported, reason in (
        (source.replace(binding, binding + "\n%group = OpDecorationGroup\n"
                        "OpDecorate %group RelaxedPrecision\nOpGroupDecorate %group %raw"),
         "unsupported indirect descriptor decorations"),
        (source.replace("%out", "%bridge_cmp"),
         "comparison sampler view requires free descriptor set 3 and bridge_cmp IDs"),
        (foreign, "unsupported depth-reference sampler provenance"),
        (source.replace("OpDecorate %samplers DescriptorSet 0", "OpDecorate %samplers DescriptorSet 4"),
         "expected undecorated sampler table at set 0 binding 1"),
        (source.replace(binding, binding + "\nOpDecorate %si NonUniformEXT"),
         "unsupported decorated sampler path"),
    ):
        try:
            split_sampler_views(unsupported)
        except ValueError as error:
            assert str(error) == reason, error
        else:
            raise AssertionError(f"accepted invalid sampler profile: {reason}")


def emit(folder, assembler, validator, cross, pixel=False):
    original = Path(__file__).with_name("sampler.spvasm")
    if pixel:
        # Same resource flow with implicit LOD in the fragment stage.
        text = original.read_text().replace('GLCompute', 'Fragment').replace(' %out %tid', ' %out')
        text = text.replace('LocalSize 1 1 1', 'OriginUpperLeft')
        text = text.replace('OpDecorate %tid BuiltIn WorkgroupId', '').replace('%tid = OpVariable %input Input', '')
        text = text.replace('%xyzval = OpLoad %xyz %tid', '').replace('OpCompositeExtract %uint %xyzval 0', 'OpCopyObject %uint %zero')
        text = text.replace('OpImageSampleExplicitLod %vec4 %si %uv Lod %lod', 'OpImageSampleImplicitLod %vec4 %si %uv')
        original = folder / "pixel.spvasm"
        original.write_text(text)
    split = folder / "split.spvasm"
    source = original.read_text()
    if pixel:
        check_sampler_rewrite(source)
    for unsupported in (source.replace("DescriptorSet 0", "DescriptorSet 3", 1),
                        source.replace("OpDecorate %out Binding 0", "OpDecorate %p NonUniformEXT")):
        try:
            split_sampler_views(unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an unsupported descriptor layout")
    split.write_text(split_sampler_views(source))
    module = folder / "shader.spv"
    for source in (original, split):
        subprocess.run([assembler, "--target-env", "vulkan1.2", source, "-o", module], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
    hlsl = folder / "shader.hlsl"
    subprocess.run([cross, module, "--hlsl", "--shader-model", "60", "--output", hlsl], check=True)
    text = hlsl.read_text()
    for token in ("SamplerState ", "SamplerComparisonState ", (".Sample(" if pixel else ".SampleLevel("), ".SampleCmpLevelZero("):
        if token not in text:
            raise ValueError(f"missing {token} in emitted HLSL")
    return hlsl


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        pixel = emit(Path(tmp), *sys.argv[1:4], pixel=True)
        if len(sys.argv) > 4:
            subprocess.run([sys.argv[4], "-T", "ps_6_0", "-E", "main", "-Fo", pixel.with_suffix(".dxil"), pixel], check=True)
        hlsl = emit(Path(tmp), *sys.argv[1:4])
        if len(sys.argv) > 4:
            dxil = hlsl.with_suffix(".dxil")
            subprocess.run([sys.argv[4], "-T", "cs_6_0", "-E", "main", "-Fo", dxil, hlsl], check=True)
            sys.exit(subprocess.run([sys.argv[5], sys.argv[6], "--spirv-sampler-views", dxil]).returncode)

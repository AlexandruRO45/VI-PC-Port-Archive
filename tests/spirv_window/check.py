"""Validate window and sampler lowering, composed ballot lowering, and rejection paths."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import spirv_address_window as window
from spirv_sampler_split import split_comparison_sampler


def validate_sampler(directory, name, source):
    assembly = directory / f"{name}.spvasm"
    binary = directory / f"{name}.spv"
    assembly.write_text(source, encoding="utf-8")
    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", assembly, "-o", binary], check=True)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", binary], check=True)


def check_sampler_dimensions(directory):
    shared = Path(__file__).with_name("shared_sampler.spvasm").read_text(encoding="utf-8")
    for dimension, arrayed, width, capability, texture in (
        ("1D", 0, 1, "Sampled1D", "Texture1D"),
        ("1D", 1, 2, "Sampled1D", "Texture1DArray"),
        ("Cube", 0, 3, None, "TextureCube"),
        ("Cube", 1, 4, "SampledCubeArray", "TextureCubeArray"),
    ):
        image_type = f"OpTypeImage %float {dimension} 2 {arrayed} 0 1 Unknown"
        source = shared.replace("OpTypeImage %float 2D 2 0 0 1 Unknown", image_type)
        if capability:
            source = source.replace("OpCapability Shader", f"OpCapability Shader\nOpCapability {capability}")
        if width == 1:
            source = source.replace(" %15 %float_", " %float_0_25 %float_")
            source = source.replace(" %15 Lod", " %float_0_25 Lod")
        else:
            coordinate = " ".join(["%float_0_25"] * width)
            source = source.replace("OpTypeVector %float 2", f"OpTypeVector %float {width}")
            source = source.replace("OpConstantComposite %v2float %float_0_25 %float_0_25",
                                    f"OpConstantComposite %v2float {coordinate}")
            if width == 4:
                source = source.replace("%v4float = OpTypeVector %float 4", "")
                source = source.replace("%v2float", "%v4float")

        validate_sampler(directory, texture, source)
        lowered, mapping = split_comparison_sampler(source, 5)
        assert mapping == {"descriptor_set": 0, "comparison_binding": 2, "ordinary_binding": 5}, texture
        lines = {line.strip() for line in lowered.splitlines()}
        assert f"%type_2d_image = {image_type}" in lines, texture
        assert "%33 = OpLoad %type_sampler %cmpSampler" in lines, texture
        assert "%34 = OpSampledImage %type_sampled_image %32 %33" in lines, texture
        assert "%37 = OpSampledImage %type_sampled_image %32 %33" in lines, texture
        assert "%x2w_plain_sampler_0 = OpLoad %type_sampler %x2w_plain_sampler" in lines, texture
        assert "%42 = OpSampledImage %type_sampled_image %40 %x2w_plain_sampler_0" in lines, texture
        assert "OpDecorate %x2w_plain_sampler Binding 5" in lines, texture
        validate_sampler(directory, f"{texture}_split", lowered)
        print(f"{texture}: shared and split SPIR-V validated")

fixture = Path(__file__).with_name("subtract.spvasm")
original = fixture.read_text(encoding="utf-8")
adapted = window.lower(original)
assert "OpMemoryModel Logical GLSL450" in adapted
assert "OpConvertUToPtr" not in adapted and "OpUGreaterThanEqual" in adapted
# The range check applies to the indexed address, with a strict bound.
assert "%x2w_address_p0_range = OpULessThan %bool %x2w_address_p0_address %x2w_end" in adapted.splitlines()

# A direct word cast is checked from its own address.
direct = original.replace(
    "%v1 = OpConvertUToPtr %_ptr_PSB_arr %a1\n"
    "%p1 = OpAccessChain %_ptr_PSB_uint %v1 %c0",
    "%p1 = OpConvertUToPtr %_ptr_PSB_uint %a1",
)
assert direct != original
assert "%x2w_address_p1_low = OpBitwiseAnd %ulong %a1 %x2w_mask" in window.lower(direct).splitlines()

# Add a ballot read whose bit controls the value written through the physical address.
combined = (original
    .replace("OpCapability Shader\n", "OpCapability Shader\nOpCapability GroupNonUniformBallot\n", 1)
    .replace("%main = OpFunction",
             "%ballot_type = OpTypeVector %uint 4\n"
             "%ballot_mask = OpConstantComposite %ballot_type %c1 %c2 %c3 %c4\n"
             "%main = OpFunction", 1)
    .replace("OpStore %p2 %d Aligned 4",
             "%ballot_bit = OpGroupNonUniformBallotBitExtract %bool %c3 %ballot_mask %c0\n"
             "%selected = OpSelect %uint %ballot_bit %d %c0\n"
             "OpStore %p2 %selected Aligned 4", 1))
assert "OpGroupNonUniformBallotBitExtract" in combined

for source, reason in (
    (original.replace("%d ", "%x2w_d "), "identifier prefix"),
    (combined.replace("%ballot_mask ", "%x2w_ballot_c0 "), "ballot lowering ID"),
    (original.replace("OpDecorate %UserData Block", "OpDecorate %user_data DescriptorSet 2\nOpDecorate %UserData Block"),
     "descriptor sets"),
    (original.replace("OpStore %p2 %d", "%leak = OpConvertPtrToU %ulong %p2\nOpStore %p2 %d"),
     "physical word pointer"),
    (original.replace("OpTypePointer PhysicalStorageBuffer %uint", "OpTypePointer PhysicalStorageBuffer %ulong"),
     "scalar layout"),
    (original.replace("OpDecorate %UserData Block",
                      "OpDecorate %user_data DescriptorSet 1\nOpDecorate %UserData Block"),
     "descriptor sets"),
    (original.replace('OpEntryPoint GLCompute %main "main" %user_data',
                      'OpEntryPoint GLCompute %main "main" %user_data\n'
                      'OpEntryPoint GLCompute %main "other" %user_data'),
     "expected one shader entry point"),
    (original.replace("OpDecorate %rt_uint ArrayStride 4", "OpDecorate %rt_uint ArrayStride 8"),
     "scalar layout"),
    (original.replace("%main = OpFunction",
                      "%unexpected_pc = OpTypePointer PushConstant %uint\n%main = OpFunction"),
     "unrecognized push-constant declaration"),
):
    try:
        window.lower(source)
    except ValueError as error:
        assert reason in str(error), error
    else:
        raise AssertionError(f"accepted invalid profile: {reason}")

output = ROOT / "out"
output.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=output) as directory:
    check_sampler_dimensions(Path(directory))
    target = Path(directory) / "window.spv"
    tool = [sys.executable, "-B", str(ROOT / "tools" / "spirv_address_window.py")]
    subprocess.run([*tool, fixture, target], check=True)
    valid = target.read_bytes()
    assert valid[:4] == bytes.fromhex("03022307")
    broken = Path(directory) / "broken.spvasm"
    broken.write_text(original.replace("%b = OpLoad %uint %p1", "%b = OpLoad %uint %undefined"), encoding="utf-8")
    failed = subprocess.run([*tool, broken, target], capture_output=True)
    assert failed.returncode and target.read_bytes() == valid

    # This input name previously overlapped the temporaries for %ballot_bit.
    renamed = combined.replace("%p2 ", "%ballot_ballot_bit ")
    assert renamed != combined
    for label, source in (("combined", combined), ("renamed_pointer", renamed), ("direct", direct)):
        mixed = Path(directory) / f"{label}.spvasm"
        mixed.write_text(source, encoding="utf-8")
        mixed_target = Path(directory) / f"{label}.spv"
        subprocess.run([*tool, mixed, mixed_target], check=True)
        assert mixed_target.read_bytes()[:4] == bytes.fromhex("03022307")
    lowered = window.lower(combined)
    assert "OpMemoryModel Logical GLSL450" in lowered
    assert "OpGroupNonUniformBallotBitExtract" not in lowered
    assert "OpConvertUToPtr" not in lowered
print("physical, logical and combined SPIR-V validated; rejection cases passed")

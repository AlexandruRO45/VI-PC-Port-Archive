"""Pre-expanded full-screen triangle for ps_ij_computed, including on GPUs without a geometry stage."""
from pathlib import Path
import subprocess

NAME = "ps_ij_computed"
VERTEX_NAME = NAME + ".ij_vs"


def build_vertex(spirv, entry):
    """Build the test producer for the fragment's explicit flat-input interface, if used."""
    root = Path(spirv)
    binary = root / (VERTEX_NAME + ".spv")
    locations = entry.get("computed_ij_flat_inputs")
    if not locations:
        binary.unlink(missing_ok=True)
        return
    bary = entry["bary_fallback_input"]["location"]
    if (set(locations) != {"0"} or len(locations["0"]) != 3
            or set(entry["inputs"]) != set(locations["0"])):
        raise ValueError("ps_ij_computed fixture must read only the three flat vertex values of attr0")

    outputs = [f"%attr{i}" for i in range(3)]
    code = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        f'OpEntryPoint Vertex %main "main" %index %position %bary {" ".join(outputs)}',
        "OpDecorate %index BuiltIn VertexIndex",
        "OpDecorate %position BuiltIn Position",
        f"OpDecorate %bary Location {bary}",
        *[f"OpDecorate {name} Location {loc}" for name, loc in zip(outputs, locations["0"])],
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%int = OpTypeInt 32 1",
        "%float = OpTypeFloat 32",
        "%vec3 = OpTypeVector %float 3",
        "%vec4 = OpTypeVector %float 4",
        "%in_int = OpTypePointer Input %int",
        "%out_vec3 = OpTypePointer Output %vec3",
        "%out_vec4 = OpTypePointer Output %vec4",
        *[f"%i{i} = OpConstant %int {i}" for i in range(3)],
        "%zero = OpConstant %float 0",
        "%one = OpConstant %float 1",
        "%two = OpConstant %float 2",
        "%minus_one = OpConstant %float -1",
        "%three = OpConstant %float 3",
        # Same raw param0 and triangle order as vs_ids.asm, with the three values exported at each corner.
        "%attr_value0 = OpConstantComposite %vec4 %zero %zero %zero %one",
        "%attr_value1 = OpConstantComposite %vec4 %two %zero %zero %one",
        "%attr_value2 = OpConstantComposite %vec4 %zero %two %zero %one",
        "%index = OpVariable %in_int Input",
        "%position = OpVariable %out_vec4 Output",
        "%bary = OpVariable %out_vec3 Output",
        *[f"{name} = OpVariable %out_vec4 Output" for name in outputs],
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%vertex = OpLoad %int %index",
        *[f"%is{i} = OpIEqual %bool %vertex %i{i}" for i in range(3)],
        *[f"%weight{i} = OpSelect %float %is{i} %one %zero" for i in range(3)],
        "%weights = OpCompositeConstruct %vec3 %weight0 %weight1 %weight2",
        "%x = OpSelect %float %is1 %three %minus_one",
        "%y = OpSelect %float %is2 %three %minus_one",
        "%pos = OpCompositeConstruct %vec4 %x %y %zero %one",
        "OpStore %position %pos",
        "OpStore %bary %weights",
        *[f"OpStore {name} %attr_value{i}" for i, name in enumerate(outputs)],
        "OpReturn",
        "OpFunctionEnd",
    ]
    asm = binary.with_suffix(".spvasm")
    asm.write_text("\n".join(code) + "\n", encoding="utf-8", newline="\n")
    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", asm, "-o", binary], check=True)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", binary], check=True)
    return binary

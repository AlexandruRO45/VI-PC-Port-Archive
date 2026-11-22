#!/usr/bin/env python3
"""Pass-through geometry shaders that feed rdna2spirv.py --bary-fallback pixel shaders.

On a device without VK_KHR_fragment_shader_barycentric, `rdna2spirv.py --bary-fallback` declares the raw barycentric
pair a pixel shader reads as an ordinary vec3 input, at the location its index.json entry gives as
"bary_fallback_input". This module writes the stage that fills it: a geometry shader that copies Position and the
pixel shader's vec4 inputs from each vertex unchanged and adds (1,0,0), (0,1,0), (0,0,1) at the triangle's three
vertices. Default (perspective) interpolation of those gives the perspective-correct barycentrics, the value
BaryCoordKHR has, so the pixel shader's .y/.z are AMD's i/j. When the index also lists
"computed_ij_flat_inputs", the helper forwards all three raw values of each such input to every output vertex.

    python -I tools/bary_fallback_gs.py <spirv dir>      # <name>.bary_gs.spv per such PS in index.json

    python -I tools/bary_fallback_gs.py <spirv dir> --pair VS PS

The paired helper forwards a VS ClipDistance or CullDistance array. The suffix identifies the kind and length.
Mixed clip/cull layouts are rejected. Standard library only; build_for_index assembles and validates with spirv-as / spirv-val from PATH.
"""
import argparse
import json
import os
import subprocess
import sys

SUFFIX = ".bary_gs"   # <ps name>.bary_gs.spvasm / .spv next to the pixel shader


def gs_spvasm(locations, bary_location, clip_count: int = 0, cull_count: int = 0, *, computed_ij=None):
    """Triangle passthrough with optional one-hot barycentrics and raw per-vertex flat attributes."""
    locations = sorted(set(int(n) for n in locations))
    computed_ij = {int(n): tuple(int(loc) for loc in ls) for n, ls in (computed_ij or {}).items()}
    flat_outputs = [(attr, v, loc) for attr, ls in sorted(computed_ij.items()) for v, loc in enumerate(ls)]
    flat_locations = {loc for _, _, loc in flat_outputs}
    if (any(len(ls) != 3 for ls in computed_ij.values()) or len(flat_locations) != len(flat_outputs)
            or flat_locations.intersection(locations)):
        raise ValueError("computed I/J needs three distinct locations clear of pass-through outputs per attribute")
    if bary_location in locations or bary_location in flat_locations:
        raise ValueError(f"bary location {bary_location} collides with another output location")
    sources = sorted(set(locations) | computed_ij.keys())
    ins = [f"%in{n}" for n in sources]
    outs = [f"%out{n}" for n in locations]
    flat = [f"%flat{loc}" for _, _, loc in flat_outputs]
    bary = bary_location is not None
    if type(clip_count) is not int or not 0 <= clip_count <= 8:
        raise ValueError("clip distance count must be an integer from 0 to 8")
    if type(cull_count) is not int or not 0 <= cull_count <= 8:
        raise ValueError("cull distance count must be an integer from 0 to 8")
    if clip_count and cull_count:
        raise ValueError("paired helper cannot forward mixed ClipDistance and CullDistance")
    count = clip_count or cull_count
    kind = "cull" if cull_count else "clip"
    builtin = kind.title() + "Distance"
    distance_iface = [f"%in_{kind}", f"%out_{kind}"] if count else []
    iface = " ".join(["%in_pos", "%out_pos", *ins, *outs, *flat,
                      *(["%out_bary"] if bary else []), *distance_iface])
    lines = [
        "OpCapability Geometry",
        *([f"OpCapability {builtin}"] if count else []),
        "OpMemoryModel Logical GLSL450",
        f'OpEntryPoint Geometry %main "main" {iface}',
        "OpExecutionMode %main Triangles",
        "OpExecutionMode %main Invocations 1",
        "OpExecutionMode %main OutputTriangleStrip",
        "OpExecutionMode %main OutputVertices 3",
        "OpDecorate %in_pos BuiltIn Position",
        "OpDecorate %out_pos BuiltIn Position",
        *([f"OpDecorate %in_{kind} BuiltIn {builtin}", f"OpDecorate %out_{kind} BuiltIn {builtin}"]
          if count else []),
        *[f"OpDecorate %in{n} Location {n}" for n in sources],
        *[f"OpDecorate %out{n} Location {n}" for n in locations],
        *[f"OpDecorate %flat{loc} Location {loc}" for _, _, loc in flat_outputs],
        *([f"OpDecorate %out_bary Location {int(bary_location)}"] if bary else []),
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%int = OpTypeInt 32 1",
        "%float = OpTypeFloat 32",
        "%v3float = OpTypeVector %float 3",
        "%v4float = OpTypeVector %float 4",
        "%uint_3 = OpConstant %uint 3",
        "%arr = OpTypeArray %v4float %uint_3",
        "%ptr_in_arr = OpTypePointer Input %arr",
        "%ptr_in_v4 = OpTypePointer Input %v4float",
        "%ptr_out_v4 = OpTypePointer Output %v4float",
        "%ptr_out_v3 = OpTypePointer Output %v3float",
        *([f"%{kind}_count = OpConstant %uint {count}",
           f"%{kind} = OpTypeArray %float %{kind}_count",
           f"%triangle_{kind} = OpTypeArray %{kind} %uint_3",
           f"%ptr_in_triangle_{kind} = OpTypePointer Input %triangle_{kind}",
           f"%ptr_in_{kind} = OpTypePointer Input %{kind}",
           f"%ptr_out_{kind} = OpTypePointer Output %{kind}"] if count else []),
        "%f0 = OpConstant %float 0",
        "%f1 = OpConstant %float 1",
        "%bary0 = OpConstantComposite %v3float %f1 %f0 %f0",
        "%bary1 = OpConstantComposite %v3float %f0 %f1 %f0",
        "%bary2 = OpConstantComposite %v3float %f0 %f0 %f1",
        *[f"%int_{v} = OpConstant %int {v}" for v in range(3)],
        "%in_pos = OpVariable %ptr_in_arr Input",
        "%out_pos = OpVariable %ptr_out_v4 Output",
        *[f"{v} = OpVariable %ptr_in_arr Input" for v in ins],
        *[f"{v} = OpVariable %ptr_out_v4 Output" for v in outs + flat],
        *(["%out_bary = OpVariable %ptr_out_v3 Output"] if bary else []),
        *([f"%in_{kind} = OpVariable %ptr_in_triangle_{kind} Input", f"%out_{kind} = OpVariable %ptr_out_{kind} Output"]
          if count else []),
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
    ]
    for attr, v, loc in flat_outputs:
        lines += [f"%flat_ptr{loc} = OpAccessChain %ptr_in_v4 %in{attr} %int_{v}",
                  f"%flat_value{loc} = OpLoad %v4float %flat_ptr{loc}"]
    for v in range(3):
        for src, dst in (("%in_pos", "%out_pos"), *((f"%in{n}", f"%out{n}") for n in locations)):
            p, x = f"%p{v}_{src[1:]}", f"%x{v}_{src[1:]}"
            lines += [f"{p} = OpAccessChain %ptr_in_v4 {src} %int_{v}", f"{x} = OpLoad %v4float {p}",
                      f"OpStore {dst} {x}"]
        if count:
            lines += [f"%{kind}_ptr{v} = OpAccessChain %ptr_in_{kind} %in_{kind} %int_{v}",
                      f"%distances{v} = OpLoad %{kind} %{kind}_ptr{v}", f"OpStore %out_{kind} %distances{v}"]
        lines.extend(f"OpStore %flat{loc} %flat_value{loc}" for _, _, loc in flat_outputs)
        if bary:
            lines.append(f"OpStore %out_bary %bary{v}")
        lines.append("OpEmitVertex")
    lines += ["OpEndPrimitive", "OpReturn", "OpFunctionEnd"]
    return "\n".join(lines) + "\n"


def write_for_index(spirv_dir, pair: tuple[str, str] | None = None):
    """Write helpers for all fallback PS entries, or one distance-layout-specific helper for an explicit (VS, PS) pair."""
    index_path = os.path.join(spirv_dir, "index.json")
    if pair is None and not os.path.isfile(index_path):
        return []
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    clip_count = cull_count = 0
    names = sorted(index)
    suffix = ""
    if pair is not None:
        vertex, pixel = pair
        producer = index.get(vertex, {})
        consumer = index.get(pixel, {})
        fallback = consumer.get("bary_fallback_input") or consumer.get("computed_ij_flat_inputs")
        if producer.get("stage") != "vs" or consumer.get("stage") != "ps" or not fallback:
            raise ValueError("--pair requires a translated VS and a barycentric-fallback PS from the index")
        flat_attrs = {int(n) for n in (consumer.get("computed_ij_flat_inputs") or {})}
        if not flat_attrs.issubset(producer.get("outputs", [])):
            raise ValueError("VS does not export all computed I/J attributes required by the PS")
        caps = set(producer.get("capabilities", []))
        unsupported = caps & {"ShaderLayer", "ShaderViewportIndex"}
        if unsupported:
            raise ValueError("paired helper cannot forward " + ", ".join(sorted(unsupported)))
        def distance_count(kind):
            if kind.title() + "Distance" not in caps:
                return 0
            field = f"{kind}_distance_count"
            count = producer.get(field)
            if type(count) is not int or not 1 <= count <= 8:
                raise ValueError(f"missing or invalid VS {field}; regenerate the translation index")
            return count

        clip_count = distance_count("clip")
        cull_count = distance_count("cull")
        suffix = f".cull{cull_count}" if cull_count else f".clip{clip_count}"
        names = [pixel]
    written = []
    for name in names:
        meta = index[name]
        if not isinstance(meta, dict):
            continue
        bary = (meta.get("bary_fallback_input") or {}).get("location")
        flat = meta.get("computed_ij_flat_inputs") or {}
        if bary is None and not flat:
            continue
        locations = set(meta.get("inputs", [])) - {loc for ls in flat.values() for loc in ls}
        path = os.path.join(spirv_dir, name + SUFFIX + suffix + ".spvasm")
        shader = gs_spvasm(locations, bary, clip_count, cull_count, computed_ij=flat)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(shader)
        written.append(path)
    return written


def build_for_index(spirv_dir, pair: tuple[str, str] | None = None):
    """write_for_index, then assemble each file to <name>.bary_gs.spv and validate it (spirv-as / spirv-val on PATH,
    Vulkan 1.2 as for the translator). Returns the .spv paths; raises RuntimeError with the tool output on failure."""
    built = []
    for asm in write_for_index(spirv_dir, pair):
        spv = asm[: -len(".spvasm")] + ".spv"
        for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", spv],
                    ["spirv-val", "--target-env", "vulkan1.2", spv]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"{cmd[0]} failed for {asm}:\n{r.stdout}{r.stderr}")
        built.append(spv)
    return built


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spirv_dir")
    parser.add_argument("--pair", nargs=2, metavar=("VS", "PS"), help="build for these index entries")
    args = parser.parse_args(argv[1:])
    if not os.path.isdir(args.spirv_dir):
        parser.exit(1, f"{args.spirv_dir}: not a directory\n")
    try:
        paths = build_for_index(args.spirv_dir, tuple(args.pair) if args.pair else None)
    except (OSError, ValueError, RuntimeError) as err:
        parser.exit(1, f"{err}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    main(sys.argv)

#!/usr/bin/env python3
"""Vertex shaders that draw what rdna2spirv.py --vs-compute compute modules wrote (issue #94, atom 2).

A tier-C wave64 VS built with --vs-compute runs as a compute prepass: workgroup (x, y) is the 64-lane wave of
vertices x * 64 .. x * 64 + 63 of instance y, and each invocation writes its vertex's position and params to slot
y * row + x * 64 + i of a buffer (rdna2spirv.vs_compute_layout, index.json "vs_compute"), where row is
NumWorkgroups.x * 64. The draw that follows uses the vertex shader written here with the same vertex and instance
ranges: vertex v of instance n (zero-based, VertexIndex - BaseVertex and InstanceIndex - BaseInstance, the v0 and v3
the original VS would have seen) reads slot n * row + v and exports its position and params at the locations the
original VS used, so the pixel shader is unchanged. The buffer's device address is in push-constant dwords 36..37 and
row in dword 38, the same tail the compute module reads (where dword 38 is the vertex count).

    python -I tools/vs_compute_vs.py <spirv dir>    # <name>.vs_compute_draw.spv per "vs_compute" VS in index.json

Standard library only; build_for_index assembles and validates with spirv-as / spirv-val from PATH.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

SUFFIX = ".vs_compute_draw"   # <vs name>.vs_compute_draw.spvasm / .spv next to <vs name>.vs_compute.spv
USER_SGPRS = 32


def vs_spvasm(layout):
    """SPIR-V assembly of the vertex shader for one --vs-compute layout (index.json "vs_compute")."""
    params = sorted((int(k), int(v)) for k, v in layout["params"].items())
    stride = int(layout["vertex_stride"])
    if stride != 16 * (1 + len(params)) or int(layout.get("position", 0)) != 0:
        raise ValueError("vs_compute_vs: unexpected layout (position at 0, one vec4 per param)")
    consts, ulongs, body, n = {}, {}, [], [0]

    def c(v):
        consts.setdefault(int(v), f"%c{int(v)}")
        return consts[int(v)]

    def cl(v):
        ulongs.setdefault(int(v), f"%cl{int(v)}")
        return ulongs[int(v)]

    def tmp(p="t"):
        n[0] += 1
        return f"%{p}{n[0]}"

    def load_vec4(base64, off):
        """vec4 of the four uints at slot byte offset `off` from the ulong address `base64`, as floats."""
        comps = []
        for i in range(4):
            a, p, w, f = tmp("a"), tmp("p"), tmp("w"), tmp("f")
            body.extend([f"{a} = OpIAdd %ulong {base64} {cl(off + 4 * i)}",
                         f"{p} = OpConvertUToPtr %_ptr_PSB_uint {a}", f"{w} = OpLoad %uint {p} Aligned 4",
                         f"{f} = OpBitcast %float {w}"])
            comps.append(f)
        v = tmp("v")
        body.append(f"{v} = OpCompositeConstruct %v4float {' '.join(comps)}")
        return v

    for k in (5, 6, 7):
        c(k)
    body += ["%vi = OpLoad %int %vertex_index", "%bv = OpLoad %int %base_vertex", "%vz = OpISub %int %vi %bv",
             "%ii = OpLoad %int %instance_index", "%bi = OpLoad %int %base_instance", "%iz = OpISub %int %ii %bi",
             "%v = OpBitcast %uint %vz", "%inst = OpBitcast %uint %iz",
             f"%rowp = OpAccessChain %_ptr_PC_uint %pc {c(7)}", "%row = OpLoad %uint %rowp",
             "%irow = OpIMul %uint %inst %row", "%slot = OpIAdd %uint %irow %v",
             f"%alo_p = OpAccessChain %_ptr_PC_uint %pc {c(5)}", f"%ahi_p = OpAccessChain %_ptr_PC_uint %pc {c(6)}",
             "%alo = OpLoad %uint %alo_p", "%ahi = OpLoad %uint %ahi_p",
             "%alo64 = OpUConvert %ulong %alo", "%ahi64 = OpUConvert %ulong %ahi",
             f"%ahis = OpShiftLeftLogical %ulong %ahi64 {cl(32)}", "%addr = OpBitwiseOr %ulong %ahis %alo64",
             "%slot64 = OpUConvert %ulong %slot", f"%soff = OpIMul %ulong %slot64 {cl(stride)}",
             "%vbase = OpIAdd %ulong %addr %soff"]
    pos = load_vec4("%vbase", 0)
    body.append(f"OpStore %out_pos {pos}")
    for loc, off in params:
        body.append(f"OpStore %out{loc} {load_vec4('%vbase', off)}")
    body += ["OpReturn", "OpFunctionEnd"]

    outs = ["%out_pos"] + [f"%out{loc}" for loc, _ in params]
    head = ["OpCapability Shader", "OpCapability Int64", "OpCapability PhysicalStorageBufferAddresses",
            "OpCapability DrawParameters", 'OpExtension "SPV_KHR_physical_storage_buffer"',
            "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
            'OpEntryPoint Vertex %main "main" %vertex_index %instance_index %base_vertex %base_instance %pc '
            + " ".join(outs),
            "OpDecorate %vertex_index BuiltIn VertexIndex", "OpDecorate %instance_index BuiltIn InstanceIndex",
            "OpDecorate %base_vertex BuiltIn BaseVertex", "OpDecorate %base_instance BuiltIn BaseInstance",
            "OpDecorate %out_pos BuiltIn Position"]
    head += [f"OpDecorate %out{loc} Location {loc}" for loc, _ in params]
    head += ["OpDecorate %arr_ud ArrayStride 4", "OpDecorate %PC Block", "OpMemberDecorate %PC 0 Offset 0"]
    head += [f"OpMemberDecorate %PC {k} Offset {4 * USER_SGPRS + 4 * (k - 1)}" for k in range(1, 8)]
    types = ["%void = OpTypeVoid", "%fn = OpTypeFunction %void", "%uint = OpTypeInt 32 0", "%int = OpTypeInt 32 1",
             "%ulong = OpTypeInt 64 0", "%float = OpTypeFloat 32", "%v4float = OpTypeVector %float 4"]
    types += [f"{name} = OpConstant %uint {v}" for v, name in sorted(consts.items())]
    types += [f"{name} = OpConstant %ulong {v}" for v, name in sorted(ulongs.items())]
    types += [f"%n_ud = OpConstant %uint {USER_SGPRS}", "%arr_ud = OpTypeArray %uint %n_ud",
              "%PC = OpTypeStruct %arr_ud %uint %uint %uint %uint %uint %uint %uint",
              "%_ptr_PC = OpTypePointer PushConstant %PC", "%_ptr_PC_uint = OpTypePointer PushConstant %uint",
              "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint",
              "%_ptr_In_int = OpTypePointer Input %int", "%_ptr_Out_v4 = OpTypePointer Output %v4float"]
    gvars = ["%vertex_index = OpVariable %_ptr_In_int Input", "%instance_index = OpVariable %_ptr_In_int Input",
             "%base_vertex = OpVariable %_ptr_In_int Input", "%base_instance = OpVariable %_ptr_In_int Input",
             "%pc = OpVariable %_ptr_PC PushConstant", "%out_pos = OpVariable %_ptr_Out_v4 Output"]
    gvars += [f"%out{loc} = OpVariable %_ptr_Out_v4 Output" for loc, _ in params]
    fn = ["%main = OpFunction %void None %fn", "%entry = OpLabel"]
    return "\n".join(head + types + gvars + fn + body) + "\n"


def write_for_index(spirv_dir):
    """Write <name>.vs_compute_draw.spvasm for every index.json entry with a "vs_compute" layout. Returns the names."""
    with open(os.path.join(spirv_dir, "index.json"), encoding="utf-8-sig") as f:
        index = json.load(f)
    names = []
    for name, entry in sorted(index.items()):
        if isinstance(entry, dict) and isinstance(entry.get("vs_compute"), dict):
            with open(os.path.join(spirv_dir, name + SUFFIX + ".spvasm"), "w", encoding="utf-8", newline="\n") as f:
                f.write(vs_spvasm(entry["vs_compute"]))
            names.append(name)
    return names


def tool_path(name):
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(name)
    return found


def build_for_index(spirv_dir):
    """write_for_index, then assemble and validate each module. Returns (built, failed)."""
    built, failed = [], []
    for name in write_for_index(spirv_dir):
        sa = os.path.join(spirv_dir, name + SUFFIX + ".spvasm")
        sv = os.path.join(spirv_dir, name + SUFFIX + ".spv")
        a = subprocess.run([tool_path("spirv-as"), "--target-env", "vulkan1.3", sa, "-o", sv],
                           capture_output=True, text=True)
        v = subprocess.run([tool_path("spirv-val"), "--target-env", "vulkan1.3", sv], capture_output=True, text=True) \
            if a.returncode == 0 else a
        (built if v.returncode == 0 else failed).append(name)
    return built, failed


def main(argv):
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print("usage: python -I tools/vs_compute_vs.py <spirv dir>", file=sys.stderr)
        return 2
    try:
        built, failed = build_for_index(argv[0])
    except FileNotFoundError as e:
        print(f"ERROR: {e} not found; install SPIRV-Tools (spirv-as, spirv-val) and put it on PATH", file=sys.stderr)
        return 2
    print(f"{len(built)} --vs-compute draw shader(s) built" + (f", {len(failed)} failed: {', '.join(failed)}"
                                                              if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

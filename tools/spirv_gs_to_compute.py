#!/usr/bin/env python3
"""Lower a translated geometry shader to a compute "capture" pass, for GPUs without a geometry stage (SHADER-159).

Apple GPUs (Metal, MoltenVK) have no geometry shaders. A legacy GS can still run there as a compute pass that
writes the vertices it emits into a buffer, followed by an ordinary draw of that buffer. This rewrites a module made
by tools/rdna2spirv.py (stage `Geometry`) into a GLCompute module with one invocation per input primitive:

- the per-vertex inputs (`BuiltIn Position`, `Location N`: Input arrays of vec4) are read from the input buffer
  (set 1, binding 0), `{ uint primitive_count; uint pad[3]; vec4 data[]; }`: vertex v of primitive p has its
  inputs at `data[(p * VERTS_IN + v) * NUM_INPUTS + slot]`;
- the vec4 outputs become private variables, and every `OpEmitVertex` copies them to the record buffer
  (set 1, binding 1), `{ vec4 rec[]; }`: vertex n of primitive p at `rec[(p * MAX_VERTS + n) * NUM_OUTPUTS + slot]`;
- the meta buffer (set 1, binding 2), `{ uint meta[]; }`, gets `meta[p * (MAX_VERTS + 1)]` = the vertices
  primitive p emitted and `meta[p * (MAX_VERTS + 1) + 1 + n]` = 1 when `OpEndPrimitive` ends a strip after vertex n
  (0 otherwise).

Slots are ordered Position first, then by location; `--layout` writes them with the sizes. `PrimitiveId` reads give
p, `InvocationId` reads 0 (only `Invocations 1` is supported). The workgroup is 64x1x1; invocations past
`primitive_count` return at once. Push constants and set 0 are untouched. The draw afterwards turns each strip into
triangles (or lines, points) in emission order.

--vs writes the vertex shader that feeds the GS as the compute pass that fills the input buffer (vs_capture), and
--draw-vs the vertex shader that draws the records from vertex attributes (draw_vs). vk_harness runs the three
passes where the device has no geometry stage (VKH_GS_COMPUTE; scripts/run_gpu_tests.py's gs_tri).

Usage: spirv_gs_to_compute.py in.spv out.spv [--layout out.json] [--vs vs.spv vs_out.spv] [--draw-vs draw.spv]
Needs spirv-dis and spirv-as (SPIRV-Tools) on PATH, or SPIRV_DIS / SPIRV_AS.
"""
import json
import os
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path

LOCAL_SIZE = 64
IN_VERTS = {"InputPoints": 1, "InputLines": 2, "InputLinesAdjacency": 4, "Triangles": 3,
            "InputTrianglesAdjacency": 6}
OUT_TOPOLOGY = {"OutputPoints": "points", "OutputLineStrip": "line_strip", "OutputTriangleStrip": "triangle_strip"}
KEEP_BUILTINS = ("SubgroupLocalInvocationId", "SubgroupSize", "NumSubgroups", "SubgroupId")


class Unsupported(Exception):
    pass


def parse(text):
    """defs: id -> (opcode, operands) for '%id = OpX ...' lines; deco: id -> its OpDecorate operand strings."""
    defs, deco = {}, {}
    for ln in text.splitlines():
        m = re.match(r"\s*(%\w+) = (Op\w+)(.*)$", ln)
        if m:
            defs[m.group(1)] = (m.group(2), m.group(3).split())
        m = re.match(r"\s*OpDecorate (%\w+) (.*)$", ln)
        if m:
            deco.setdefault(m.group(1), []).append(m.group(2))
    return defs, deco


def find_type(defs, op, operands):
    return next((r for r, d in defs.items() if d == (op, operands)), None)


def layout_of(text):
    """The shader's inputs, outputs and sizes; raises Unsupported for what the rewrite can't handle."""
    defs, deco = parse(text)
    ep = re.search(r"^\s*OpEntryPoint Geometry (%\w+) ", text, re.M)
    if not ep:
        raise Unsupported("not a geometry shader")
    main = ep.group(1)
    modes = re.findall(rf"^\s*OpExecutionMode {re.escape(main)} (.*)$", text, re.M)
    modes = [m.split() for m in modes]
    verts_in = next((IN_VERTS[m[0]] for m in modes if m[0] in IN_VERTS), None)
    topology = next((OUT_TOPOLOGY[m[0]] for m in modes if m[0] in OUT_TOPOLOGY), None)
    max_verts = next((int(m[1]) for m in modes if m[0] == "OutputVertices"), None)
    invocations = next((int(m[1]) for m in modes if m[0] == "Invocations"), 1)
    if None in (verts_in, topology, max_verts):
        raise Unsupported(f"execution modes {modes}")
    if invocations != 1:
        raise Unsupported(f"Invocations {invocations}")
    f32 = find_type(defs, "OpTypeFloat", ["32"])
    v4 = find_type(defs, "OpTypeVector", [f32, "4"])

    def slot(v):
        ds = deco.get(v, [])
        if "BuiltIn Position" in ds:
            return (-1, "position")
        loc = next((int(d.split()[1]) for d in ds if d.startswith("Location ")), None)
        if loc is None:
            raise Unsupported(f"{v}: per-vertex variable without Position or Location ({ds})")
        return (loc, f"location {loc}")

    inputs, outputs, builtins = [], [], {}
    for v, (op, ops) in defs.items():
        if op != "OpVariable" or ops[1:2] not in (["Input"], ["Output"]):
            continue
        bi = next((d.split()[1] for d in deco.get(v, []) if d.startswith("BuiltIn ")), None)
        if bi in KEEP_BUILTINS or bi in ("PrimitiveId", "InvocationId"):
            builtins[v] = bi
            continue
        pointee = defs[ops[0]][1][1]
        if ops[1] == "Input":
            t = defs.get(pointee, (None, []))
            if t[0] != "OpTypeArray" or t[1][0] != v4:
                raise Unsupported(f"{v}: input isn't an array of vec4")
            inputs.append(v)
        elif pointee == v4:
            outputs.append(v)
        else:
            raise Unsupported(f"{v}: output isn't a vec4")
    inputs.sort(key=slot)
    outputs.sort(key=slot)
    return {"main": main, "verts_in": verts_in, "max_verts": max_verts, "topology": topology,
            "inputs": [slot(v)[1] for v in inputs], "outputs": [slot(v)[1] for v in outputs],
            "input_vars": inputs, "output_vars": outputs, "builtins": builtins}


def rewrite(text):
    """Return (compute module text, layout dict for --layout)."""
    lay = layout_of(text)
    defs, _ = parse(text)
    main, ins, outs, builtins = lay["main"], lay["input_vars"], lay["output_vars"], lay["builtins"]
    uint = find_type(defs, "OpTypeInt", ["32", "0"])
    f32 = find_type(defs, "OpTypeFloat", ["32"])
    v4 = find_type(defs, "OpTypeVector", [f32, "4"])
    if not (uint and v4):
        raise Unsupported("module has no uint or vec4 type")
    v3u = find_type(defs, "OpTypeVector", [uint, "3"]) or "%gsc_v3u"
    boolt = find_type(defs, "OpTypeBool", []) or "%gsc_bool"
    float_ptrs = {r for r, (op, ops) in defs.items() if op == "OpTypePointer" and ops[1] == f32}
    nin, nout, mv, vin = len(ins), len(outs), lay["max_verts"], lay["verts_in"]

    def k(v):
        return f"%gsc_k{v}"

    decos = ["OpDecorate %gsc_gid BuiltIn GlobalInvocationId",
             "OpDecorate %gsc_rt_v4 ArrayStride 16", "OpDecorate %gsc_rt_u ArrayStride 4",
             "OpDecorate %gsc_pad ArrayStride 4",
             "OpDecorate %gsc_in_t Block", "OpMemberDecorate %gsc_in_t 0 Offset 0",
             "OpMemberDecorate %gsc_in_t 1 Offset 4", "OpMemberDecorate %gsc_in_t 2 Offset 16",
             "OpDecorate %gsc_rec_t Block", "OpMemberDecorate %gsc_rec_t 0 Offset 0",
             "OpDecorate %gsc_meta_t Block", "OpMemberDecorate %gsc_meta_t 0 Offset 0",
             "OpDecorate %gsc_in DescriptorSet 1", "OpDecorate %gsc_in Binding 0", "OpDecorate %gsc_in NonWritable",
             "OpDecorate %gsc_rec DescriptorSet 1", "OpDecorate %gsc_rec Binding 1",
             "OpDecorate %gsc_meta DescriptorSet 1", "OpDecorate %gsc_meta Binding 2"]
    types = ([f"{v3u} = OpTypeVector {uint} 3"] if v3u == "%gsc_v3u" else []) + \
            ([f"{boolt} = OpTypeBool"] if boolt == "%gsc_bool" else [])
    types += [f"{k(v)} = OpConstant {uint} {v}" for v in sorted({0, 1, 2, 3, vin, nin, nout, mv, mv + 1}
                                                              | set(range(max(nin, nout))))]
    types += [f"%gsc_ptr_in_v3u = OpTypePointer Input {v3u}",
              f"%gsc_rt_v4 = OpTypeRuntimeArray {v4}", f"%gsc_rt_u = OpTypeRuntimeArray {uint}",
              f"%gsc_pad = OpTypeArray {uint} {k(3)}",
              f"%gsc_in_t = OpTypeStruct {uint} %gsc_pad %gsc_rt_v4",
              "%gsc_rec_t = OpTypeStruct %gsc_rt_v4", "%gsc_meta_t = OpTypeStruct %gsc_rt_u",
              "%gsc_ptr_in_t = OpTypePointer StorageBuffer %gsc_in_t",
              "%gsc_ptr_rec_t = OpTypePointer StorageBuffer %gsc_rec_t",
              "%gsc_ptr_meta_t = OpTypePointer StorageBuffer %gsc_meta_t",
              f"%gsc_ptr_sb_v4 = OpTypePointer StorageBuffer {v4}",
              f"%gsc_ptr_sb_u = OpTypePointer StorageBuffer {uint}",
              f"%gsc_ptr_sb_f = OpTypePointer StorageBuffer {f32}",
              f"%gsc_ptr_pv_v4 = OpTypePointer Private {v4}", f"%gsc_ptr_pv_f = OpTypePointer Private {f32}",
              f"%gsc_ptr_pv_u = OpTypePointer Private {uint}",
              "%gsc_gid = OpVariable %gsc_ptr_in_v3u Input",
              "%gsc_in = OpVariable %gsc_ptr_in_t StorageBuffer",
              "%gsc_rec = OpVariable %gsc_ptr_rec_t StorageBuffer",
              "%gsc_meta = OpVariable %gsc_ptr_meta_t StorageBuffer",
              f"%gsc_n = OpVariable %gsc_ptr_pv_u Private {k(0)}"]
    out_vars = {v: f"%gsc_out{i}" for i, v in enumerate(outs)}
    types += [f"{nv} = OpVariable %gsc_ptr_pv_v4 Private" for nv in out_vars.values()]
    in_slot = {v: i for i, v in enumerate(ins)}
    gone = set(ins) | set(outs) | {v for v, b in builtins.items() if b in ("PrimitiveId", "InvocationId")}
    new_globals = ["%gsc_gid", "%gsc_in", "%gsc_rec", "%gsc_meta", "%gsc_n"] + list(out_vars.values())

    count = 0

    def tmp():
        nonlocal count
        count += 1
        return f"%gsc_t{count}"

    out = []
    entry_label = None
    state = "header"        # header -> body (in main, entry block) -> after the entry check
    for ln in text.splitlines():
        s = ln.strip()
        if s == "OpCapability Geometry":
            continue
        m = re.match(r"OpEntryPoint Geometry (%\w+) (\"[^\"]*\")(.*)$", s)
        if m:
            iface = [v for v in m.group(3).split() if v not in gone] + new_globals
            out += [f"OpEntryPoint GLCompute {m.group(1)} {m.group(2)} {' '.join(iface)}",
                    f"OpExecutionMode {m.group(1)} LocalSize {LOCAL_SIZE} 1 1"]
            continue
        if re.match(rf"OpExecutionMode {re.escape(main)} ", s):
            continue
        m = re.match(r"Op(?:Decorate|MemberDecorate|Name) (%\w+)\b", s)
        if m and m.group(1) in gone:
            continue
        m = re.match(r"(%\w+) = (Op\w+)", s)
        if decos and m and m.group(2).startswith("OpType"):
            out += decos            # decorations end where the first type begins
            decos = []
        if m and m.group(2) == "OpVariable" and m.group(1) in gone:
            continue
        if m and m.group(2) == "OpFunction":
            if types:
                out += types
                types = []
            if m.group(1) == main:
                state = "main_entry"
            out.append(ln)
            continue
        if s == "OpFunctionEnd":
            state = "other"
        if state == "main_entry" and re.match(r"%\w+ = OpLabel$", s):
            state = "main_vars"
            entry_label = s.split()[0]
            out.append(ln)
            continue
        if state == "main_vars" and not re.match(r"%\w+ = OpVariable ", s):
            # after the entry block's variables: p = GlobalInvocationId.x; return when p >= primitive_count
            state = "main"
            out += [f"%gsc_gidv = OpLoad {v3u} %gsc_gid", f"%gsc_prim = OpCompositeExtract {uint} %gsc_gidv 0",
                    f"%gsc_cntp = OpAccessChain %gsc_ptr_sb_u %gsc_in {k(0)}", f"%gsc_cnt = OpLoad {uint} %gsc_cntp",
                    f"%gsc_past = OpUGreaterThanEqual {boolt} %gsc_prim %gsc_cnt",
                    "OpSelectionMerge %gsc_body None", "OpBranchConditional %gsc_past %gsc_ret %gsc_body",
                    "%gsc_ret = OpLabel", "OpReturn", "%gsc_body = OpLabel",
                    f"%gsc_inbase = OpIMul {uint} %gsc_prim {k(vin)}",
                    f"%gsc_recbase = OpIMul {uint} %gsc_prim {k(mv)}",
                    f"%gsc_metabase = OpIMul {uint} %gsc_prim {k(mv + 1)}"]
        if state not in ("main", "main_vars"):
            out.append(ln)
            continue
        # ---- the body of main ----
        # The primitive-count check splits the entry block: what followed its variables now sits in %gsc_body, so
        # an OpPhi naming the entry label as a predecessor names %gsc_body instead.
        m = re.match(r"(%\w+ = OpPhi %\w+) (.*)$", s)
        if m:
            ops = m.group(2).split()
            ops[1::2] = ["%gsc_body" if lbl == entry_label else lbl for lbl in ops[1::2]]
            s = f"{m.group(1)} {' '.join(ops)}"
            ln = s
        m = re.match(r"(%\w+) = Op(?:InBounds)?AccessChain (%\w+) (%\w+) (.*)$", s)
        if m and m.group(3) in in_slot:
            idx = m.group(4).split()
            vtx = idx[0]
            if defs.get(vtx, (None, [None]))[1][:1] != [uint]:
                vtx = tmp()
                out.append(f"{vtx} = OpBitcast {uint} {idx[0]}")
            a, b, c = tmp(), tmp(), tmp()
            ptr = "%gsc_ptr_sb_f" if m.group(2) in float_ptrs else "%gsc_ptr_sb_v4"
            out += [f"{a} = OpIAdd {uint} %gsc_inbase {vtx}", f"{b} = OpIMul {uint} {a} {k(nin)}",
                    f"{c} = OpIAdd {uint} {b} {k(in_slot[m.group(3)])}",
                    " ".join([f"{m.group(1)} = OpAccessChain {ptr} %gsc_in {k(2)} {c}"] + idx[1:])]
            continue
        if m and m.group(3) in out_vars:
            ptr = "%gsc_ptr_pv_f" if m.group(2) in float_ptrs else "%gsc_ptr_pv_v4"
            out.append(f"{m.group(1)} = OpAccessChain {ptr} {out_vars[m.group(3)]} {m.group(4)}")
            continue
        m = re.match(r"(%\w+) = OpLoad (%\w+) (%\w+)$", s)
        if m and builtins.get(m.group(3)) in ("PrimitiveId", "InvocationId"):
            val = "%gsc_prim" if builtins[m.group(3)] == "PrimitiveId" else k(0)
            out.append(f"{m.group(1)} = OpBitcast {m.group(2)} {val}" if m.group(2) != uint
                       else f"{m.group(1)} = OpCopyObject {uint} {val}")
            continue
        for v in list(in_slot) + list(out_vars):
            if re.search(rf"{re.escape(v)}\b", s):
                m2 = re.match(rf"OpStore {re.escape(v)} (%\w+)$", s)
                if v in out_vars and m2:
                    s = f"OpStore {out_vars[v]} {m2.group(1)}"
                    break
                raise Unsupported(f"unhandled use of {v}: {s}")
        if s == "OpEmitVertex":
            n, a = tmp(), tmp()
            out += [f"{n} = OpLoad {uint} %gsc_n", f"{a} = OpIAdd {uint} %gsc_recbase {n}"]
            for i, nv in enumerate(out_vars.values()):
                b, c, val, p = tmp(), tmp(), tmp(), tmp()
                out += [f"{b} = OpIMul {uint} {a} {k(nout)}", f"{c} = OpIAdd {uint} {b} {k(i)}",
                        f"{val} = OpLoad {v4} {nv}", f"{p} = OpAccessChain %gsc_ptr_sb_v4 %gsc_rec {k(0)} {c}",
                        f"OpStore {p} {val}"]
            d, e, q, n1 = tmp(), tmp(), tmp(), tmp()
            out += [f"{d} = OpIAdd {uint} %gsc_metabase {n}", f"{e} = OpIAdd {uint} {d} {k(1)}",
                    f"{q} = OpAccessChain %gsc_ptr_sb_u %gsc_meta {k(0)} {e}", f"OpStore {q} {k(0)}",
                    f"{n1} = OpIAdd {uint} {n} {k(1)}", f"OpStore %gsc_n {n1}"]
            continue
        if s == "OpEndPrimitive":
            # flag the last emitted vertex: meta[base + 1 + (n - 1)]; with none emitted this writes the count slot,
            # which the final store overwrites
            n, d, q = tmp(), tmp(), tmp()
            out += [f"{n} = OpLoad {uint} %gsc_n", f"{d} = OpIAdd {uint} %gsc_metabase {n}",
                    f"{q} = OpAccessChain %gsc_ptr_sb_u %gsc_meta {k(0)} {d}", f"OpStore {q} {k(1)}"]
            continue
        if s == "OpReturn":
            n, q = tmp(), tmp()
            out += [f"{n} = OpLoad {uint} %gsc_n",
                    f"{q} = OpAccessChain %gsc_ptr_sb_u %gsc_meta {k(0)} %gsc_metabase", f"OpStore {q} {n}"]
        if s in ("OpKill", "OpTerminateInvocation", "OpReturnValue"):
            raise Unsupported(s)
        out.append(s)
    public = {key: lay[key] for key in ("verts_in", "max_verts", "topology", "inputs", "outputs")}
    public["local_size"] = LOCAL_SIZE
    return "\n".join(out) + "\n", public


ZERO_VERTEX_BUILTINS = ("InstanceIndex", "BaseVertex", "BaseInstance", "DrawIndex")   # one draw, one instance, base 0


def vs_capture(text, lay):
    """The vertex shader feeding the GS, rewritten into the compute pass that fills the GS's input buffer.

    One invocation per vertex of a non-indexed list draw (workgroup LOCAL_SIZE x 1 x 1): vertex v reads VertexIndex v
    (InstanceIndex, BaseVertex, BaseInstance and DrawIndex read 0) and returns at once when v >= primitive_count *
    verts_in. At every return from the entry point, the outputs the GS reads (`lay`, rewrite()'s layout) go to
    `data[v * NUM_INPUTS + slot]` of the input buffer (set 1, binding 0), which is where the GS pass reads vertex v of
    primitive v / verts_in. The other outputs are kept as private variables and dropped. A vertex shader with vertex
    attributes (Input variables other than these builtins) or an output block is not supported.
    """
    defs, deco = parse(text)
    ep = re.search(r"^\s*OpEntryPoint Vertex (%\w+) ", text, re.M)
    if not ep:
        raise Unsupported("not a vertex shader")
    main = ep.group(1)
    uint = find_type(defs, "OpTypeInt", ["32", "0"])
    f32 = find_type(defs, "OpTypeFloat", ["32"])
    v4 = find_type(defs, "OpTypeVector", [f32, "4"])
    if not v4:
        raise Unsupported("module has no vec4 type")
    uint = uint or "%gsc_uint"
    v3u = find_type(defs, "OpTypeVector", [uint, "3"]) or "%gsc_v3u"
    boolt = find_type(defs, "OpTypeBool", []) or "%gsc_bool"
    vertex_vars, outputs, out_vars = {}, {}, set()
    for v, (op, ops) in defs.items():
        if op != "OpVariable" or ops[1:2] not in (["Input"], ["Output"]):
            continue
        ds = deco.get(v, [])
        bi = next((d.split()[1] for d in ds if d.startswith("BuiltIn ")), None)
        pointee = defs[ops[0]][1][1]
        if ops[1] == "Input":
            if bi in ("VertexIndex",) + ZERO_VERTEX_BUILTINS:
                vertex_vars[v] = bi
            elif bi not in KEEP_BUILTINS:
                raise Unsupported(f"{v}: vertex input {bi or ds} (vertex attributes are not fed to the capture pass)")
            continue
        if "Block" in deco.get(pointee, []):
            raise Unsupported(f"{v}: output block")
        out_vars.add(v)
        loc = next((int(d.split()[1]) for d in ds if d.startswith("Location ")), None)
        name = "position" if bi == "Position" else f"location {loc}" if loc is not None else None
        if name and pointee == v4:
            outputs[name] = v
    missing = [n for n in lay["inputs"] if n not in outputs]
    if missing:
        raise Unsupported(f"the vertex shader writes no vec4 for the GS input(s) {missing}")
    nin, vin = len(lay["inputs"]), lay["verts_in"]

    def k(v):
        return f"%gsc_k{v}"

    decos = ["OpDecorate %gsc_gid BuiltIn GlobalInvocationId",
             "OpDecorate %gsc_rt_v4 ArrayStride 16", "OpDecorate %gsc_pad ArrayStride 4",
             "OpDecorate %gsc_in_t Block", "OpMemberDecorate %gsc_in_t 0 Offset 0",
             "OpMemberDecorate %gsc_in_t 1 Offset 4", "OpMemberDecorate %gsc_in_t 2 Offset 16",
             "OpDecorate %gsc_in DescriptorSet 1", "OpDecorate %gsc_in Binding 0"]
    types = ([f"{uint} = OpTypeInt 32 0"] if uint == "%gsc_uint" else []) + \
            ([f"{v3u} = OpTypeVector {uint} 3"] if v3u == "%gsc_v3u" else []) + \
            ([f"{boolt} = OpTypeBool"] if boolt == "%gsc_bool" else [])
    types += [f"{k(v)} = OpConstant {uint} {v}" for v in sorted({0, 2, 3, vin, nin} | set(range(nin)))]
    types += [f"%gsc_ptr_in_v3u = OpTypePointer Input {v3u}",
              f"%gsc_rt_v4 = OpTypeRuntimeArray {v4}", f"%gsc_pad = OpTypeArray {uint} {k(3)}",
              f"%gsc_in_t = OpTypeStruct {uint} %gsc_pad %gsc_rt_v4",
              "%gsc_ptr_in_t = OpTypePointer StorageBuffer %gsc_in_t",
              f"%gsc_ptr_sb_v4 = OpTypePointer StorageBuffer {v4}",
              f"%gsc_ptr_sb_u = OpTypePointer StorageBuffer {uint}",
              "%gsc_gid = OpVariable %gsc_ptr_in_v3u Input", "%gsc_in = OpVariable %gsc_ptr_in_t StorageBuffer"]
    count = 0

    def tmp():
        nonlocal count
        count += 1
        return f"%gsc_t{count}"

    out = []
    entry_label = None
    state = "header"
    for ln in text.splitlines():
        s = ln.strip()
        if s == "OpCapability DrawParameters" or s == 'OpExtension "SPV_KHR_shader_draw_parameters"':
            continue        # BaseVertex/BaseInstance/DrawIndex are constants now
        m = re.match(r"OpEntryPoint Vertex (%\w+) (\"[^\"]*\")(.*)$", s)
        if m:
            iface = [v for v in m.group(3).split() if v not in vertex_vars] + ["%gsc_gid", "%gsc_in"]
            out += [f"OpEntryPoint GLCompute {m.group(1)} {m.group(2)} {' '.join(iface)}",
                    f"OpExecutionMode {m.group(1)} LocalSize {LOCAL_SIZE} 1 1"]
            continue
        m = re.match(r"OpDecorate (%\w+)\b", s)
        if m and (m.group(1) in vertex_vars or m.group(1) in out_vars):
            continue        # builtins that are gone, and the outputs' BuiltIn/Location/interpolation decorations
        m = re.match(r"Op(?:MemberDecorate|Name) (%\w+)\b", s)
        if m and m.group(1) in vertex_vars:
            continue
        m = re.match(r"(%\w+) = (Op\w+)", s)
        if decos and m and m.group(2).startswith("OpType"):
            out += decos
            decos = []
        if m and m.group(2) == "OpVariable" and m.group(1) in vertex_vars:
            continue
        if m and m.group(2) == "OpTypePointer" and s.split()[3] == "Output":
            s = re.sub(r"OpTypePointer Output ", "OpTypePointer Private ", s)
            out.append(s)
            continue
        if m and m.group(2) == "OpVariable" and m.group(1) in out_vars:
            out.append(re.sub(r" Output\b", " Private", s, count=1))
            continue
        if m and m.group(2) == "OpFunction":
            if types:
                out += types
                types = []
            state = "main_entry" if m.group(1) == main else "other"
            out.append(ln)
            continue
        if s == "OpFunctionEnd":
            state = "other"
        if state == "main_entry" and re.match(r"%\w+ = OpLabel$", s):
            state = "main_vars"
            entry_label = s.split()[0]
            out.append(ln)
            continue
        if state == "main_vars" and not re.match(r"%\w+ = OpVariable ", s):
            # after the entry block's variables: v = GlobalInvocationId.x; return when v >= primitive_count * verts_in
            state = "main"
            out += [f"%gsc_gidv = OpLoad {v3u} %gsc_gid", f"%gsc_vtx = OpCompositeExtract {uint} %gsc_gidv 0",
                    f"%gsc_cntp = OpAccessChain %gsc_ptr_sb_u %gsc_in {k(0)}", f"%gsc_cnt = OpLoad {uint} %gsc_cntp",
                    f"%gsc_total = OpIMul {uint} %gsc_cnt {k(vin)}",
                    f"%gsc_past = OpUGreaterThanEqual {boolt} %gsc_vtx %gsc_total",
                    "OpSelectionMerge %gsc_body None", "OpBranchConditional %gsc_past %gsc_ret %gsc_body",
                    "%gsc_ret = OpLabel", "OpReturn", "%gsc_body = OpLabel",
                    f"%gsc_inbase = OpIMul {uint} %gsc_vtx {k(nin)}"]
        if state not in ("main", "main_vars"):
            if any(re.search(rf"{re.escape(v)}\b", s) for v in vertex_vars) and not s.startswith("OpEntryPoint"):
                raise Unsupported(f"vertex builtin read outside the entry point: {s}")
            out.append(ln)
            continue
        m = re.match(r"(%\w+ = OpPhi %\w+) (.*)$", s)
        if m:
            ops = m.group(2).split()
            ops[1::2] = ["%gsc_body" if lbl == entry_label else lbl for lbl in ops[1::2]]
            s = f"{m.group(1)} {' '.join(ops)}"
        m = re.match(r"(%\w+) = OpLoad (%\w+) (%\w+)$", s)
        if m and m.group(3) in vertex_vars:
            val = "%gsc_vtx" if vertex_vars[m.group(3)] == "VertexIndex" else k(0)
            out.append(f"{m.group(1)} = OpBitcast {m.group(2)} {val}" if m.group(2) != uint
                       else f"{m.group(1)} = OpCopyObject {uint} {val}")
            continue
        if any(re.search(rf"{re.escape(v)}\b", s) for v in vertex_vars):
            raise Unsupported(f"unhandled use of a vertex builtin: {s}")
        if s == "OpReturn":
            for i, name in enumerate(lay["inputs"]):
                val, a, p = tmp(), tmp(), tmp()
                out += [f"{val} = OpLoad {v4} {outputs[name]}", f"{a} = OpIAdd {uint} %gsc_inbase {k(i)}",
                        f"{p} = OpAccessChain %gsc_ptr_sb_v4 %gsc_in {k(2)} {a}", f"OpStore {p} {val}"]
        if s in ("OpKill", "OpTerminateInvocation", "OpReturnValue"):
            raise Unsupported(s)
        out.append(s)
    return "\n".join(out) + "\n"


def draw_vs(lay):
    """The vertex shader that draws the captured vertices: vertex attribute `location i` (a vec4) is output slot i of
    the layout, written to the same output (BuiltIn Position or Location N) the geometry shader wrote it to."""
    outs = lay["outputs"]
    names = [f"%in{i}" for i in range(len(outs))] + [f"%out{i}" for i in range(len(outs))]
    text = ["OpCapability Shader", "OpMemoryModel Logical GLSL450",
            f"OpEntryPoint Vertex %main \"main\" {' '.join(names)}"]
    for i, slot in enumerate(outs):
        text.append(f"OpDecorate %in{i} Location {i}")
        text.append(f"OpDecorate %out{i} BuiltIn Position" if slot == "position"
                    else f"OpDecorate %out{i} Location {int(slot.split()[1])}")
    text += ["%void = OpTypeVoid", "%fn = OpTypeFunction %void", "%float = OpTypeFloat 32",
             "%v4 = OpTypeVector %float 4", "%ptr_in = OpTypePointer Input %v4", "%ptr_out = OpTypePointer Output %v4"]
    text += [f"%in{i} = OpVariable %ptr_in Input" for i in range(len(outs))]
    text += [f"%out{i} = OpVariable %ptr_out Output" for i in range(len(outs))]
    text += ["%main = OpFunction %void None %fn", "%entry = OpLabel"]
    for i in range(len(outs)):
        text += [f"%v{i} = OpLoad %v4 %in{i}", f"OpStore %out{i} %v{i}"]
    text += ["OpReturn", "OpFunctionEnd"]
    return "\n".join(text) + "\n"


USAGE = "\n".join(__doc__.strip().splitlines()[-2:])


def main(argv):
    args = argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    lay_path = None
    if "--layout" in args:
        i = args.index("--layout")
        lay_path = args[i + 1]
        del args[i:i + 2]
    vs_paths = draw_path = None
    if "--vs" in args:
        i = args.index("--vs")
        vs_paths = args[i + 1:i + 3]
        del args[i:i + 3]
    if "--draw-vs" in args:
        i = args.index("--draw-vs")
        draw_path = args[i + 1:i + 2]
        del args[i:i + 2]
        draw_path = draw_path[0] if draw_path else None
    if len(args) != 2 or (vs_paths is not None and len(vs_paths) != 2) or ("--draw-vs" in argv and not draw_path):
        print(USAGE, file=sys.stderr)
        return 2
    paths = [("input", args[0]), ("SPIR-V output", args[1])]
    if lay_path:
        paths.append(("--layout output", lay_path))
    if vs_paths:
        paths += [("--vs input", vs_paths[0]), ("--vs output", vs_paths[1])]
    if draw_path:
        paths.append(("--draw-vs output", draw_path))
    try:
        resolved = [(name, Path(path).resolve()) for name, path in paths]
        for (left_name, left), (right_name, right) in combinations(resolved, 2):
            if left == right or (left.exists() and right.exists() and left.samefile(right)):
                print(f"{left_name} and {right_name} refer to the same file: {left}", file=sys.stderr)
                return 2
    except (OSError, RuntimeError, ValueError) as e:
        print(f"cannot check input/output paths: {e}", file=sys.stderr)
        return 2
    dis = os.environ.get("SPIRV_DIS", "spirv-dis")
    asm = os.environ.get("SPIRV_AS", "spirv-as")
    text = subprocess.run([dis, "--raw-id", args[0]], check=True, capture_output=True, text=True).stdout
    try:
        new, lay = rewrite(text)
    except Unsupported as e:
        print(f"{args[0]}: not supported: {e}", file=sys.stderr)
        return 1
    vs_text = vs_new = None
    if vs_paths:
        vs_text = subprocess.run([dis, "--raw-id", vs_paths[0]], check=True, capture_output=True, text=True).stdout
        try:
            vs_new = vs_capture(vs_text, lay)
        except Unsupported as e:
            print(f"{vs_paths[0]}: not supported: {e}", file=sys.stderr)
            return 1

    def assemble(source_text, module_text, path):
        ver = re.search(r"; Version: (\d+)\.(\d+)", source_text)
        target = f"spv{ver.group(1)}.{ver.group(2)}" if ver else "vulkan1.2"
        subprocess.run([asm, "--target-env", target, "-", "-o", path], input=module_text, check=True, text=True)

    assemble(text, new, args[1])
    if vs_paths:
        assemble(vs_text, vs_new, vs_paths[1])
    if draw_path:
        assemble(text, draw_vs(lay), draw_path)
    if lay_path:
        with open(lay_path, "w", newline="\n") as f:
            json.dump(lay, f, indent=1)
    print(json.dumps(lay))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv))

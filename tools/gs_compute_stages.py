#!/usr/bin/env python3
"""The two stages that draw what a --gs-compute geometry shader captured (SHADER-1092, the SHADER-159 follow-up).

On a device without the Vulkan geometry stage, `rdna2spirv.py --gs-compute` publishes the capture module that
`tools/spirv_gs_to_compute.py` makes (records + meta buffers, one invocation per input primitive). These are the
other two modules, generated per GS from its index.json "gs_compute" layout, with the strip-to-triangle decode
inside the vertex shader as straight-line selects (there is no index pass):

- `<name>.gs_scatter.spv` builds the capture pass's input buffer: it reads the per-vertex outputs of the real
  vertex shader (a linear buffer in the GS's input slot order: position first, then locations ascending, one
  vec4 per slot, num_inputs vec4s per vertex) and the draw's index buffer, and regroups them into the
  primitive-major form the capture pass reads. Push-constant tail members (after the pipeline's own 32 dwords
  of user SGPRs and its heap and code base):
      1..2   linear VS-output buffer
      3..4   index buffer; when member 8 is 2 (no index buffer) this only needs to be readable, so the
             caller can pass the linear buffer's address
      5..6   capture input buffer ({ uint primitive_count; uint pad[3]; vec4 data[]; })
      7      index count (the vertex count when member 8 is 2); primitive_count = member 7 / verts_in
      8      0: R16_UINT indices, 1: R32_UINT indices, 2: no index buffer
      9      the base: first vertex when member 8 is 2, base vertex otherwise (wraps as u32, as Vulkan does)
  Invocation 0 writes the count, 1..3 clear the pad, and invocation i >= 4 copies data dword i - 4.

- `<name>.gs_vs.spv` is the vertex shader of the record-driven draw, on the pattern of tools/mesh_compute_vs.py:
  vertex v is corner c of out-triangle t of primitive p, the draw's worst case being every primitive's
  max(1, max_verts - 2) strip triangles, drawn as a triangle list. The decode walks the meta buffer's end
  flags in emission order with selects, and the vertices past the last end flag form the invocation's trailing
  strip, as a geometry shader's implicit OpEndPrimitive does. A triangle past the primitive's emitted count (or
  a primitive past the count) gets a position outside the clip volume; the corner's record feeds the outputs at
  the GS's own output slots, so the pixel shader is unchanged. Push-constant tail members 1..5: record buffer
  address, meta buffer address, primitive count.

    python3 -I tools/gs_compute_stages.py <spirv dir> [name ...]    # per index entry with "gs_compute"

Standard library only; assembles and validates with spirv-as / spirv-val from PATH.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

USER_SGPRS = 32  # dwords 0..31 are the pipeline's user SGPRs; 32..33 its heap and code base, as mesh_compute_vs.py


class Unsupported(Exception):
    pass


class _E:
    """Append-only expression builder: every helper emits ops and returns a fresh SSA name."""

    def __init__(self):
        self.consts, self.ulongs, self.body, self.n = {}, {}, [], [0]

    def tmp(self, p="t"):
        self.n[0] += 1
        return f"%{p}{self.n[0]}"

    def c(self, v):
        self.consts.setdefault(int(v), f"%c{int(v)}")
        return self.consts[int(v)]

    def cl(self, v):
        self.ulongs.setdefault(int(v), f"%cl{int(v)}")
        return self.ulongs[int(v)]

    # -- push constants and addresses
    def load_pc(self, k):
        """Tail member k (1-based): the uint at push-constant dword 36 + k - 1, past the user SGPRs (0..31,
        member 0) and the heap and code base (dwords 32..35, struct members 1..4)."""
        p, v = self.tmp("p"), self.tmp("w")
        self.body += [f"{p} = OpAccessChain %_ptr_PC_uint %pc {self.c(4 + k)}", f"{v} = OpLoad %uint {p}"]
        return v

    def bda(self, lo, hi):
        """The 64-bit address from two push-constant dwords."""
        l, h, s, a = self.tmp("l"), self.tmp("h"), self.tmp("s"), self.tmp("a")
        self.body += [f"{l} = OpUConvert %ulong {lo}", f"{h} = OpUConvert %ulong {hi}",
                      f"{s} = OpShiftLeftLogical %ulong {h} {self.cl(32)}",
                      f"{a} = OpBitwiseOr %ulong {s} {l}"]
        return a

    # -- PhysicalStorageBuffer access; indices are dwords unless the name says bytes
    def dw(self, addr, idx):
        """The byte address of dword `idx` within the buffer at `addr`."""
        return self.add64(addr, self.u64(self.mul(idx, self.c(4))))

    def load_word(self, addr, off):
        """uint at byte offset `off` (an int) from the ulong `addr`."""
        w, p, v = self.tmp("w"), self.tmp("p"), self.tmp("v")
        self.body += [f"{w} = OpIAdd %ulong {addr} {self.cl(off)}" if off else f"{w} = OpCopyObject %ulong {addr}",
                      f"{p} = OpConvertUToPtr %_ptr_PSB_uint {w}", f"{v} = OpLoad %uint {p} Aligned 4"]
        return v

    def store_word(self, addr, off, val):
        w, p = self.tmp("w"), self.tmp("p")
        self.body += [f"{w} = OpIAdd %ulong {addr} {self.cl(off)}" if off else f"{w} = OpCopyObject %ulong {addr}",
                      f"{p} = OpConvertUToPtr %_ptr_PSB_uint {w}", f"OpStore {p} {val} Aligned 4"]

    def load_dword(self, addr, idx):
        return self.load_word(self.dw(addr, idx), 0)

    def store_dword(self, addr, idx, val):
        self.store_word(self.dw(addr, idx), 0, val)

    # -- integer / boolean expression helpers
    def op(self, name, ty, *vals):
        r = self.tmp(name[:1])
        self.body.append(f"{r} = Op{name} {ty} {' '.join(str(v) for v in vals)}")
        return r

    def add(self, a, b):
        return self.op("IAdd", "%uint", a, b)

    def add64(self, a, b):
        return self.op("IAdd", "%ulong", a, b)

    def sub(self, a, b):
        return self.op("ISub", "%uint", a, b)

    def mul(self, a, b):
        return self.op("IMul", "%uint", a, b)

    def udiv(self, a, b):
        return self.op("UDiv", "%uint", a, b)

    def umod(self, a, b):
        return self.op("UMod", "%uint", a, b)

    def shl(self, a, b):
        return self.op("ShiftLeftLogical", "%uint", a, b)

    def shr(self, a, b):
        return self.op("ShiftRightLogical", "%uint", a, b)

    def and_(self, a, b):
        return self.op("BitwiseAnd", "%uint", a, b)

    def and64(self, a, b):
        return self.op("BitwiseAnd", "%ulong", a, b)

    def eq(self, a, b):
        return self.op("IEqual", "%bool", a, b)

    def ne(self, a, b):
        return self.op("INotEqual", "%bool", a, b)

    def lt(self, a, b):
        return self.op("ULessThan", "%bool", a, b)

    def ge(self, a, b):
        return self.op("UGreaterThanEqual", "%bool", a, b)

    def land(self, a, b):
        return self.op("LogicalAnd", "%bool", a, b)

    def lor(self, a, b):
        return self.op("LogicalOr", "%bool", a, b)

    def lnot(self, a):
        return self.op("LogicalNot", "%bool", a)

    def sel_u(self, cond, a, b):
        return self.op("Select", "%uint", cond, a, b)

    def sel_f4(self, cond, a, b):
        return self.op("Select", "%v4float", cond, a, b)

    def u64(self, a):
        return self.op("UConvert", "%ulong", a)


def _prologue(entry, pc_tail, io_vars, extra_decos=()):
    """Decorations and the entry point; the caller appends types, globals and the function body.
    %pc holds the pipeline's 32 user-SGPR dwords (member 0), its heap and code base (members 1..4, dwords
    32..35) and `pc_tail` tail dwords from dword 36, as tools/mesh_compute_vs.py's block does."""
    decos = ["OpDecorate %PC Block", "OpMemberDecorate %PC 0 Offset 0", "OpDecorate %arr_ud ArrayStride 4"]
    decos += [f"OpMemberDecorate %PC {m} Offset {128 + 4 * (m - 1)}" for m in range(1, 5)]
    decos += [f"OpMemberDecorate %PC {4 + k} Offset {4 * USER_SGPRS + 16 + 4 * (k - 1)}"
              for k in range(1, pc_tail + 1)]
    head = ["OpCapability Shader", "OpCapability Int64", "OpCapability PhysicalStorageBufferAddresses",
            'OpExtension "SPV_KHR_physical_storage_buffer"', "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
            f'OpEntryPoint {entry} %main "main" {" ".join(io_vars)} %pc']
    if entry == "GLCompute":
        head.append("OpExecutionMode %main LocalSize 64 1 1")
    return head + list(extra_decos) + decos


def _types_and_globals(x, mode, pc_tail, extra_globals=()):
    """The shared types, the constant pools and the module-scope variables.
    mode "%gid" is the scatter (compute); mode "%vertex_index" is the draw vertex shader."""
    types = ["%void = OpTypeVoid", "%fn = OpTypeFunction %void", "%bool = OpTypeBool",
             "%uint = OpTypeInt 32 0", "%int = OpTypeInt 32 1", "%ulong = OpTypeInt 64 0"]
    if mode == "%gid":
        types += ["%v3u = OpTypeVector %uint 3", "%_ptr_In_v3u = OpTypePointer Input %v3u"]
    else:
        types += ["%float = OpTypeFloat 32", "%v4float = OpTypeVector %float 4",
                  "%_ptr_In_int = OpTypePointer Input %int", "%_ptr_Out_v4 = OpTypePointer Output %v4float",
                  "%f1 = OpConstant %float 1", "%f2 = OpConstant %float 2",
                  "%outside = OpConstantComposite %v4float %f2 %f2 %f2 %f1", "%false = OpConstantFalse %bool"]
    types += [f"{n} = OpConstant %uint {v}" for v, n in sorted(x.consts.items())]
    types += [f"{n} = OpConstant %ulong {v}" for v, n in sorted(x.ulongs.items())]
    types += [f"%n_ud = OpConstant %uint {USER_SGPRS}", "%arr_ud = OpTypeArray %uint %n_ud",
              "%PC = OpTypeStruct %arr_ud %uint %uint %uint %uint " + " ".join(["%uint"] * pc_tail),
              "%_ptr_PC = OpTypePointer PushConstant %PC", "%_ptr_PC_uint = OpTypePointer PushConstant %uint",
              "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint"]
    gvars = []
    if mode == "%gid":
        gvars.append("%gid = OpVariable %_ptr_In_v3u Input")
    else:
        gvars.append("%vertex_index = OpVariable %_ptr_In_int Input")
    gvars.append("%pc = OpVariable %_ptr_PC PushConstant")
    return types + gvars + list(extra_globals)


def scatter_module(lay):
    """The regroup pass: linear VS outputs + indices -> the capture pass's primitive-major input buffer."""
    nin, vin = int(lay["num_inputs"]), int(lay["verts_in"])
    x = _E()
    per = 4 * nin * vin  # data dwords per primitive
    x.body += ["%gidv = OpLoad %v3u %gid", "%gidi = OpCompositeExtract %uint %gidv 0"]
    # Every push-constant load happens before the first branch, so it dominates all its uses.
    lin = x.bda(x.load_pc(1), x.load_pc(2))
    idx = x.bda(x.load_pc(3), x.load_pc(4))
    in_addr = x.bda(x.load_pc(5), x.load_pc(6))
    cnt = x.load_pc(7)
    flag = x.load_pc(8)
    first = x.load_pc(9)
    prims = x.udiv(cnt, x.c(vin))
    # The header words: invocation 0 writes the primitive count, 1..3 clear the pad.
    head_c = x.lt("%gidi", x.c(4))
    x.body += ["OpSelectionMerge %head_done None", f"OpBranchConditional {head_c} %head_do %head_done",
               "%head_do = OpLabel"]
    x.store_word(x.add64(in_addr, x.u64(x.mul("%gidi", x.c(4)))), 0,
                 x.sel_u(x.eq("%gidi", x.c(0)), prims, x.c(0)))
    x.body += ["OpBranch %head_done", "%head_done = OpLabel"]
    # Data dword i = gid - 4: input vertex pv = i / (4 * nin) of primitive pv / vin, slot and component.
    i = x.sub("%gidi", x.c(4))
    words = x.mul(prims, x.c(per))
    x.body += ["OpSelectionMerge %data_done None",
               f"OpBranchConditional {x.lt(i, words)} %data_do %data_done", "%data_do = OpLabel"]
    pv = x.udiv(i, x.c(4 * nin))
    p = x.udiv(pv, x.c(vin))
    v = x.umod(pv, x.c(vin))
    slot_comp = x.umod(i, x.c(4 * nin))
    slot = x.udiv(slot_comp, x.c(4))
    comp = x.umod(i, x.c(4))
    # The source vertex id: the u16 and u32 reads are both done, and member 3..4 only has to be readable
    # when member 8 is 2, so the flag only picks the result (no path reads an unusable address).
    ivc = x.add(x.mul(p, x.c(vin)), v)
    byte = x.mul(ivc, x.c(2))
    wv16 = x.load_word(x.add64(idx, x.and64(x.u64(byte), x.cl(4294967292))), 0)
    v16 = x.and_(x.shr(wv16, x.shl(x.umod(byte, x.c(2)), x.c(3))), x.c(65535))
    v32 = x.load_dword(idx, x.mul(ivc, x.c(4)))
    vid = x.sel_u(x.eq(flag, x.c(1)), v32, v16)
    vid = x.sel_u(x.eq(flag, x.c(2)), ivc, vid)
    vid = x.add(vid, first)
    # linear[(vid * nin + slot) * 4 + comp] -> input data dword (pv * nin + slot) * 4 + comp.
    lin_dw = x.add(x.mul(x.add(x.mul(vid, x.c(nin)), slot), x.c(4)), comp)
    src = x.load_dword(lin, lin_dw)
    in_dw = x.add(x.mul(x.add(x.mul(pv, x.c(nin)), slot), x.c(4)), comp)
    x.store_dword(in_addr, x.add(in_dw, x.c(4)), src)  # + 4 dwords: past the count and the pad
    x.body += ["OpBranch %data_done", "%data_done = OpLabel", "OpReturn", "OpFunctionEnd"]
    head = _prologue("GLCompute", 9, ("%gid",), ("OpDecorate %gid BuiltIn GlobalInvocationId",))
    tail = _types_and_globals(x, "%gid", 9)
    return "\n".join(head + tail + ["%main = OpFunction %void None %fn", "%entry = OpLabel"] + x.body) + "\n"


def draw_vs_module(lay):
    """The vertex shader of the record-driven draw, the strip-to-triangle decode inside as straight-line selects."""
    nout, mv, topo = len(lay["outputs"]), int(lay["max_verts"]), lay["topology"]
    if topo != "triangle_strip":
        raise Unsupported(f"gs_compute_stages: topology {topo!r} is not wired up (the corpus emits strips)")
    if not lay["outputs"] or lay["outputs"][0] != "position":
        raise Unsupported("gs_compute_stages: the lowering orders the position output first")
    out_locs = [int(o.split()[1]) for o in lay["outputs"] if o != "position"]
    x = _E()
    per = 3 * (mv - 2 if mv >= 3 else 1)
    x.body += ["%vid = OpLoad %int %vertex_index", "%vidu = OpBitcast %uint %vid",
               f"%p = OpUDiv %uint %vidu {x.c(per)}", f"%rem = OpUMod %uint %vidu {x.c(per)}",
               f"%t = OpUDiv %uint %rem {x.c(3)}", f"%corner = OpUMod %uint %rem {x.c(3)}"]
    prims = x.load_pc(5)
    rec = x.bda(x.load_pc(1), x.load_pc(2))
    meta = x.bda(x.load_pc(3), x.load_pc(4))
    p_in = x.lt("%p", prims)
    p_safe = x.sel_u(p_in, "%p", x.c(0))  # a primitive past the count reads primitive 0, clipped out below
    base = x.mul(p_safe, x.c(mv + 1))
    count = x.load_dword(meta, base)
    # meta[base] = the emitted count; meta[base + 1 + n] = 1 when a strip ends after vertex n. The flags of
    # vertices past the count are stale buffer content, so only a flag below the count ends a segment.
    seg, before, found, f_seg, f_i = x.c(0), x.c(0), None, x.c(0), x.c(0)
    for n in range(mv):
        flag = x.load_dword(meta, x.add(base, x.c(n + 1)))
        end = x.land(x.lt(x.c(n), count), x.ne(flag, x.c(0)))
        seg_end = x.sel_u(end, x.c(n + 1), seg)
        seg_len = x.sub(seg_end, seg)
        seg_tris = x.sel_u(x.ge(seg_len, x.c(3)), x.sub(seg_len, x.c(2)), x.c(0))
        here = x.land(x.land(x.ge("%t", before), x.lt("%t", x.add(before, seg_tris))),
                      x.lnot(found or "%false"))
        f_seg = x.sel_u(here, seg, f_seg)
        f_i = x.sel_u(here, x.sub("%t", before), f_i)
        found = x.lor(found or "%false", here)
        before = x.add(before, seg_tris)
        seg = x.sel_u(end, x.c(n + 1), seg)
    # The vertices past the last end flag form the invocation's trailing strip: an implicit OpEndPrimitive.
    tail_len = x.sub(count, seg)
    tail_tris = x.sel_u(x.ge(tail_len, x.c(3)), x.sub(tail_len, x.c(2)), x.c(0))
    here = x.land(x.land(x.ge("%t", before), x.lt("%t", x.add(before, tail_tris))),
                  x.lnot(found or "%false"))
    f_seg = x.sel_u(here, seg, f_seg)
    f_i = x.sel_u(here, x.sub("%t", before), f_i)
    found = x.lor(found or "%false", here)
    valid = x.land(p_in, found)
    # Corner -> emitted-vertex id: corner 0 is f_seg + f_i; odd strip triangles flip the winding.
    c0 = x.add(f_seg, f_i)
    a1 = x.add(f_seg, x.add(f_i, x.c(1)))
    a2 = x.add(a1, x.c(1))
    i_even = x.eq(x.and_(f_i, x.c(1)), x.c(0))
    id1 = x.sel_u(i_even, a1, a2)
    id2 = x.sel_u(i_even, a2, a1)
    vid = x.sel_u(x.eq("%corner", x.c(1)), id1, c0)
    vid = x.sel_u(x.eq("%corner", x.c(2)), id2, vid)
    vid = x.sel_u(valid, vid, x.c(0))
    # The corner's record: rec[(p * max_verts + vid) * num_outputs + slot], the lowering's layout.
    rb = x.mul(x.add(x.mul(p_safe, x.c(mv)), vid), x.c(nout))
    pos_addr = x.add64(rec, x.u64(x.mul(rb, x.c(16))))
    posw = [x.load_word(pos_addr, 4 * k) for k in range(4)]
    x.body += [f"%pf{k} = OpBitcast %float {posw[k]}" for k in range(4)]
    pos = x.tmp("pos")
    x.body.append(f"{pos} = OpCompositeConstruct %v4float %pf0 %pf1 %pf2 %pf3")
    pos_out = x.sel_f4(valid, pos, "%outside")
    x.body.append(f"OpStore %out_pos {pos_out}")
    for s, loc in enumerate(out_locs, start=1):
        addr = x.add64(rec, x.u64(x.mul(x.add(rb, x.c(s)), x.c(16))))
        words = [x.load_word(addr, 4 * k) for k in range(4)]
        vec = x.tmp("ov")
        x.body += [f"%of{k}_{loc} = OpBitcast %float {words[k]}" for k in range(4)]
        x.body += [f"{vec} = OpCompositeConstruct %v4float %of0_{loc} %of1_{loc} %of2_{loc} %of3_{loc}",
                   f"OpStore %out{loc} {vec}"]
    x.body += ["OpReturn", "OpFunctionEnd"]
    io = ["%vertex_index", "%out_pos"] + [f"%out{l}" for l in out_locs]
    decos = ["OpDecorate %vertex_index BuiltIn VertexIndex", "OpDecorate %out_pos BuiltIn Position"] + \
            [f"OpDecorate %out{l} Location {l}" for l in out_locs]
    head = _prologue("Vertex", 5, tuple(io), tuple(decos))
    gvars = ["%out_pos = OpVariable %_ptr_Out_v4 Output"] + \
            [f"%out{l} = OpVariable %_ptr_Out_v4 Output" for l in out_locs]
    tail = _types_and_globals(x, "%vertex_index", 5, extra_globals=gvars)
    return "\n".join(head + tail + ["%main = OpFunction %void None %fn", "%entry = OpLabel"] + x.body) + "\n"


def build_for_index(spirv_dir, names=None):
    """Write <name>.gs_scatter.spv and <name>.gs_vs.spv for every index entry with a "gs_compute" layout."""
    with open(os.path.join(spirv_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    written = []
    for name, entry in sorted(index.items()):
        lay = entry.get("gs_compute")
        if not lay or (names and name not in names):
            continue
        for suffix, module in (("gs_scatter", scatter_module), ("gs_vs", draw_vs_module)):
            try:
                text = module(lay)
            except Unsupported as e:
                print(f"WARNING: {name}: {e}", file=sys.stderr)
                continue
            Path(os.path.join(spirv_dir, f"{name}.{suffix}.spvasm")).write_text(
                text, encoding="utf-8", newline="\n")
            spv_path = os.path.join(spirv_dir, f"{name}.{suffix}.spv")
            r = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-o", spv_path, "-"],
                               input=text.encode(), capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"{name}.{suffix}: spirv-as rejected the module:\n"
                                   f"{r.stderr.decode(errors='replace')[:500]}")
            r = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", spv_path], capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"{name}.{suffix}: spirv-val rejected the module:\n"
                                   f"{r.stderr.decode(errors='replace')[:500]}")
            written.append(os.path.basename(spv_path))
            print(f"{name}.{suffix}.spv assembled and validated")
    return written


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 0 if argv else 2
    spirv_dir = argv[0]
    if not os.path.isfile(os.path.join(spirv_dir, "index.json")):
        print(f"ERROR: {spirv_dir}/index.json not found", file=sys.stderr)
        return 2
    build_for_index(spirv_dir, set(argv[1:]) or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

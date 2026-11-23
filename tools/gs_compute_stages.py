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
import re
import subprocess
import sys
from pathlib import Path

USER_SGPRS = 32  # dwords 0..31 are the pipeline's user SGPRs; 32..33 its heap and code base, as mesh_compute_vs.py

INPUT_SLOTS_MAX = 6   # vs_input: the vertex-buffer table's slot count (3 tail dwords each, as an array member)
VS_INPUT_SLOTS = 9    # vs_input: 9 scalar tail dwords past the 144-byte block, then the per-slot array


class Unsupported(Exception):
    pass


class _E:
    """Append-only expression builder: every helper emits ops and returns a fresh SSA name.

    The type ids and the constant-name prefixes are parameters: the generated-from-scratch modules use the
    defaults, the rewriters of existing modules point them at the module's own ids so no name collides."""

    def __init__(self):
        self.consts, self.ulongs, self.body, self.n = {}, {}, [], [0]
        self.cpre, self.clpre = "%c", "%cl"
        self.t_uint, self.t_bool, self.t_ulong = "%uint", "%bool", "%ulong"
        self.t_psb_u, self.t_psb_v4 = "%_ptr_PSB_uint", None   # a PhysicalStorageBuffer pointer type id

    def tmp(self, p="t"):
        self.n[0] += 1
        return f"%{p}{self.n[0]}"

    def c(self, v):
        self.consts.setdefault(int(v), f"{self.cpre}{int(v)}")
        return self.consts[int(v)]

    def cl(self, v):
        self.ulongs.setdefault(int(v), f"{self.clpre}{int(v)}")
        return self.ulongs[int(v)]

    # -- push constants and addresses
    def load_pc(self, k):
        """Parameter k (1-based). "s" mode: the uint at s0 + k - 1, the draw's own user data, as the D3D12
        layer's flavors read them. "tail" mode: push-constant dword 36 + k - 1, past the user SGPRs (0..31,
        member 0) and the heap and code base (dwords 32..35, struct members 1..4)."""
        p, v = self.tmp("p"), self.tmp("w")
        if getattr(self, "pc_mode", "tail") == "s":
            self.body += [f"{p} = OpAccessChain %_ptr_PC_uint %pc {self.c(0)} {self.c(k - 1)}",
                          f"{v} = OpLoad %uint {p}"]
        else:
            self.body += [f"{p} = OpAccessChain %_ptr_PC_uint %pc {self.c(4 + k)}", f"{v} = OpLoad %uint {p}"]
        return v

    def bda(self, lo, hi):
        """The 64-bit address from two push-constant dwords."""
        l, h, s, a = self.tmp("l"), self.tmp("h"), self.tmp("s"), self.tmp("a")
        self.body += [f"{l} = OpUConvert {self.t_ulong} {lo}", f"{h} = OpUConvert {self.t_ulong} {hi}",
                      f"{s} = OpShiftLeftLogical {self.t_ulong} {h} {self.cl(32)}",
                      f"{a} = OpBitwiseOr {self.t_ulong} {s} {l}"]
        return a

    # -- PhysicalStorageBuffer access; indices are dwords unless the name says bytes
    def dw(self, addr, idx):
        """The byte address of dword `idx` (an int or a value) within the buffer at `addr`."""
        idx = self.c(idx) if isinstance(idx, int) else idx
        return self.add64(addr, self.u64(self.mul(idx, self.c(4))))

    def load_word(self, addr, off):
        """uint at byte offset `off` (an int) from the ulong `addr`."""
        w, p, v = self.tmp("w"), self.tmp("p"), self.tmp("v")
        self.body += [f"{w} = OpIAdd {self.t_ulong} {addr} {self.cl(off)}" if off
                      else f"{w} = OpCopyObject {self.t_ulong} {addr}",
                      f"{p} = OpConvertUToPtr {self.t_psb_u} {w}", f"{v} = OpLoad {self.t_uint} {p} Aligned 4"]
        return v

    def store_word(self, addr, off, val):
        w, p = self.tmp("w"), self.tmp("p")
        self.body += [f"{w} = OpIAdd {self.t_ulong} {addr} {self.cl(off)}" if off
                      else f"{w} = OpCopyObject {self.t_ulong} {addr}",
                      f"{p} = OpConvertUToPtr {self.t_psb_u} {w}", f"OpStore {p} {val} Aligned 4"]

    def store_vec4(self, addr, off, val):
        """A whole vec4 store at byte offset `off`; t_psb_v4 must name a vec4 PhysicalStorageBuffer pointer."""
        w, p = self.tmp("w"), self.tmp("p")
        self.body += [f"{w} = OpIAdd {self.t_ulong} {addr} {self.cl(off)}" if off
                      else f"{w} = OpCopyObject {self.t_ulong} {addr}",
                      f"{p} = OpConvertUToPtr {self.t_psb_v4} {w}", f"OpStore {p} {val} Aligned 16"]

    def load_dword(self, addr, idx):
        return self.load_word(self.dw(addr, idx), 0)

    def store_dword(self, addr, idx, val):
        self.store_word(self.dw(addr, idx), 0, val)

    # -- integer / boolean expression helpers
    def op(self, name, ty, *vals):
        r = self.tmp(name[:1])
        self.body.append(f"{r} = Op{name} {ty} {' '.join(self.c(v) if isinstance(v, int) else str(v)
                                                          for v in vals)}")
        return r

    def add(self, a, b):
        return self.op("IAdd", self.t_uint, a, b)

    def add64(self, a, b):
        return self.op("IAdd", self.t_ulong, a, b)

    def sub(self, a, b):
        return self.op("ISub", self.t_uint, a, b)

    def mul(self, a, b):
        return self.op("IMul", self.t_uint, a, b)

    def udiv(self, a, b):
        return self.op("UDiv", self.t_uint, a, b)

    def umod(self, a, b):
        return self.op("UMod", self.t_uint, a, b)

    def shl(self, a, b):
        return self.op("ShiftLeftLogical", self.t_uint, a, b)

    def shr(self, a, b):
        return self.op("ShiftRightLogical", self.t_uint, a, b)

    def and_(self, a, b):
        return self.op("BitwiseAnd", self.t_uint, a, b)

    def and64(self, a, b):
        return self.op("BitwiseAnd", self.t_ulong, a, b)

    def eq(self, a, b):
        return self.op("IEqual", self.t_bool, a, b)

    def ne(self, a, b):
        return self.op("INotEqual", self.t_bool, a, b)

    def lt(self, a, b):
        return self.op("ULessThan", self.t_bool, a, b)

    def ge(self, a, b):
        return self.op("UGreaterThanEqual", self.t_bool, a, b)

    def land(self, a, b):
        return self.op("LogicalAnd", self.t_bool, a, b)

    def lor(self, a, b):
        return self.op("LogicalOr", self.t_bool, a, b)

    def lnot(self, a):
        return self.op("LogicalNot", self.t_bool, a)

    def sel_u(self, cond, a, b):
        return self.op("Select", self.t_uint, cond, a, b)

    def sel_f4(self, cond, a, b):
        return self.op("Select", "%v4float", cond, a, b)

    def u64(self, a):
        return self.op("UConvert", self.t_ulong, a)


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


def _layer_prologue(entry, io_vars, extra_decos=()):
    """The layer flavor's head: the standard translated-module push-constant block (s0..s31, heap, code),
    144 bytes; the draw's own user data carries the stage's parameters in s0..s4."""
    decos = ["OpDecorate %PC Block", "OpMemberDecorate %PC 0 Offset 0", "OpDecorate %arr_ud ArrayStride 4",
             "OpMemberDecorate %PC 1 Offset 128", "OpMemberDecorate %PC 2 Offset 132",
             "OpMemberDecorate %PC 3 Offset 136", "OpMemberDecorate %PC 4 Offset 140"]
    head = ["OpCapability Shader", "OpCapability Int64", "OpCapability PhysicalStorageBufferAddresses",
            'OpExtension "SPV_KHR_physical_storage_buffer"', "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
            f'OpEntryPoint {entry} %main "main" {" ".join(io_vars)} %pc']
    return head + list(extra_decos) + decos


def _layer_types(x, extra_globals=()):
    """The layer flavor's types and globals: the standard block plus this generator's own."""
    types = ["%void = OpTypeVoid", "%fn = OpTypeFunction %void", "%bool = OpTypeBool",
             "%uint = OpTypeInt 32 0", "%int = OpTypeInt 32 1", "%ulong = OpTypeInt 64 0",
             "%float = OpTypeFloat 32", "%v4float = OpTypeVector %float 4",
             "%_ptr_In_int = OpTypePointer Input %int", "%_ptr_Out_v4 = OpTypePointer Output %v4float",
             f"%n_ud = OpConstant %uint {USER_SGPRS}", "%arr_ud = OpTypeArray %uint %n_ud",
             "%PC = OpTypeStruct %arr_ud %uint %uint %uint %uint",
             "%_ptr_PC = OpTypePointer PushConstant %PC", "%_ptr_PC_uint = OpTypePointer PushConstant %uint",
             "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint",
             "%f1 = OpConstant %float 1", "%f2 = OpConstant %float 2",
             "%outside = OpConstantComposite %v4float %f2 %f2 %f2 %f1", "%false = OpConstantFalse %bool"]
    types += [f"{n} = OpConstant %uint {v}" for v, n in sorted(x.consts.items())]
    types += [f"{n} = OpConstant %ulong {v}" for v, n in sorted(x.ulongs.items())]
    gvars = ["%vertex_index = OpVariable %_ptr_In_int Input", "%pc = OpVariable %_ptr_PC PushConstant"]
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
    v16 = x.and_(x.shr(wv16, x.shl(x.umod(byte, x.c(4)), x.c(3))), x.c(65535))
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


def draw_vs_module(lay, pc_mode="tail"):
    """The vertex shader of the record-driven draw, the strip-to-triangle decode inside as straight-line selects."""
    nout, mv, topo = len(lay["outputs"]), int(lay["max_verts"]), lay["topology"]
    if topo != "triangle_strip":
        raise Unsupported(f"gs_compute_stages: topology {topo!r} is not wired up (the corpus emits strips)")
    if not lay["outputs"] or lay["outputs"][0] != "position":
        raise Unsupported("gs_compute_stages: the lowering orders the position output first")
    out_locs = [int(o.split()[1]) for o in lay["outputs"] if o != "position"]
    x = _E()
    x.pc_mode = pc_mode
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
    gvars = ["%out_pos = OpVariable %_ptr_Out_v4 Output"] + \
            [f"%out{l} = OpVariable %_ptr_Out_v4 Output" for l in out_locs]
    if pc_mode == "s":
        head, tail = _layer_prologue("Vertex", tuple(io), tuple(decos)), \
            _layer_types(x, extra_globals=gvars)
    else:
        head, tail = _prologue("Vertex", 5, tuple(io), tuple(decos)), \
            _types_and_globals(x, "%vertex_index", 5, extra_globals=gvars)
    return "\n".join(head + tail + ["%main = OpFunction %void None %fn", "%entry = OpLabel"] + x.body) + "\n"


# ---- the D3D12 layer's flavor (SHADER-1092): addresses as heap-relative console VAs, no descriptor sets ----
#
# The layer cannot bind a descriptor set at dispatch time, so its generated stages address every buffer through
# the push-constant block: the heap base is already members 1..2 of every translated module's %user_data, and the
# console VAs the recorder fills in are added to it. The pairing of a vertex shader with a geometry shader is
# runtime state (a D3D12 PSO names both), so nothing pair-specific is baked into a module: the per-PSO facts ride a
# small parameters block in the arena, the per-draw facts ride push constants.


def _parse_vs(text):
    """The interface facts of a vertex shader's disassembly: types, decorations, variables, constant values."""
    # spirv-dis indents some lines; every pattern tolerates that.
    types = {}
    for m in re.finditer(r"^\s*(%\w+) = (OpType\w+) ?([^\n]*)$", text, re.M):
        types[m.group(1)] = (m.group(2), m.group(3).split())
    const_vals = {m.group(1): int(m.group(3))
                  for m in re.finditer(r"^\s*(%\w+) = OpConstant (%\w+) (\d+)$", text, re.M)}
    locs = {m.group(1): int(m.group(2)) for m in re.finditer(r"^\s*OpDecorate (%\w+) Location (\d+)$", text, re.M)}
    builtins = {m.group(1): m.group(2) for m in re.finditer(r"^\s*OpDecorate (%\w+) BuiltIn (\w+)$", text, re.M)}
    member_builtins = {(m.group(1), int(m.group(2))): m.group(3)
                       for m in re.finditer(r"^\s*OpMemberDecorate (%\w+) (\d+) BuiltIn (\w+)$", text, re.M)}
    variables = [(m.group(1), m.group(2), m.group(3))
                 for m in re.finditer(r"^\s*(%\w+) = OpVariable (%\w+) (Input|Output|PushConstant)$", text, re.M)]

    def pointee(ptr):
        return types[ptr][1][1]

    return types, const_vals, locs, builtins, member_builtins, variables, pointee


def vs_input_module(text):
    """The layer's fused fetch + vertex shader + regroup pass, generated per vertex shader (SHADER-1092).

    One invocation per input-primitive vertex slot (pv = GlobalInvocationId.x): it resolves the slot's source
    vertex through the index buffer, fetches the vertex's attributes from the bound vertex buffers, runs the
    vertex shader's own code unchanged, and stores the outputs the geometry shader consumes straight into the
    capture pass's primitive-major input buffer (invocation 0 writes the count word; the pad is never read).
    The vertex shader's own s0..s31 and heap and code base pass through untouched.

    Push constants, past the module's own block (the block grows by 108 bytes, to 252 inside Vulkan's 256
    minimum): tail 0..1 the parameters block's console VA (heap-relative), 2..3 the capture input buffer's VA,
    4..5 the index buffer's VA (only read when tail 7 is 0 or 1; a non-indexed draw passes any readable VA),
    6 the index count (the vertex count when tail 7 is 2), 7 the kind (0: R16_UINT, 1: R32_UINT, 2:
    non-indexed), 8 the base (first vertex, or base vertex; wraps as u32); then one array member, element
    3s + 0..2: vertex buffer slot s's console VA (lo, hi) and stride. The input layout's slot count is capped
    at INPUT_SLOTS_MAX.

    The parameters block (per pipeline, heap-relative): dword 0 the geometry shader's input slot count (nin),
    1 its verts_in, 2 this shader's output slot count; then per output slot j (position first, then locations
    ascending) the geometry shader input slot that consumes output j, or 0xffffffff when none does (that store
    goes to a scratch vec4 the capture pass never reads, so the module stays free of data-dependent branches);
    then per input location, ascending, its input layout slot and byte offset. The capture input buffer is the
    16-byte header, then the dense rows the capture pass reads (slots*nin vec4s), then one scratch vec4 per
    slot; a scratch store lands at slots*nin + pv, so it never races another invocation's row.
    """
    types, const_vals, locs, builtins, member_builtins, variables, pointee = _parse_vs(text)
    uint = next((i for i, (op, ops) in types.items() if op == "OpTypeInt" and ops[:2] == ["32", "0"]), None)
    f32 = next((i for i, (op, ops) in types.items() if op == "OpTypeFloat" and ops[:1] == ["32"]), None)
    v4f = next((i for i, (op, ops) in types.items() if op == "OpTypeVector" and ops == [f32, "4"]), None)
    boolt = next((i for i, (op, ops) in types.items() if op == "OpTypeBool"), None)
    ulong = next((i for i, (op, ops) in types.items() if op == "OpTypeInt" and ops[:2] == ["64", "0"]), None)
    pcu = next((i for i, (op, ops) in types.items() if op == "OpTypePointer" and ops[:2] == ["PushConstant", uint]),
               None)
    if not (uint and f32 and v4f):
        raise Unsupported("vs_input: module has no uint or float-vec4 type")
    ep = re.search(r'^\s*OpEntryPoint Vertex (%\w+) "main"(.*)$', text, re.M)
    if not ep:
        raise Unsupported("vs_input: no Vertex entry point")
    if "OpMemoryModel Logical GLSL450" not in text:
        raise Unsupported("vs_input: module's memory model is not Logical GLSL450")
    entry = ep.group(1)
    pc_var = next((v for v, ptr, cls in variables if cls == "PushConstant"), None)
    if not pc_var:
        raise Unsupported("vs_input: module has no push-constant block (the translator's UserData contract)")
    pc_struct = pointee(next(ptr for v, ptr, cls in variables if v == pc_var))
    n_members = len(types[pc_struct][1])

    def is_v4(ptr):
        return pointee(ptr) == v4f

    in_vars = {v: locs[v] for v, ptr, cls in variables if cls == "Input" and v in locs and is_v4(ptr)}
    out_vars = {v: locs[v] for v, ptr, cls in variables if cls == "Output" and v in locs and is_v4(ptr)}
    in_builtins = {v: builtins[v] for v, ptr, cls in variables
                   if cls == "Input" and builtins.get(v) in
                   ("VertexIndex", "InstanceIndex", "BaseVertex", "BaseInstance")}
    pos_var = None
    for v, ptr, cls in variables:
        if cls != "Output":
            continue
        if builtins.get(v) == "Position":
            pos_var = v
            break
        if member_builtins.get((pointee(ptr), 0)) == "Position":
            pos_var = v  # the gl_PerVertex-style struct output; member 0 is the position
            break
    if not pos_var:
        raise Unsupported("vs_input: module has no position output")
    for v, ptr, cls in variables:
        if cls == "Input" and v not in in_vars and v not in in_builtins:
            raise Unsupported(f"vs_input: unhandled input variable {v} (iface blocks or non-vec4 attributes)")
    n_in = len(in_vars)
    in_rank = {loc: r for r, loc in enumerate(sorted(in_vars.values()))}
    out_slots = sorted(out_vars.items(), key=lambda kv: kv[1])
    out_slot_of = {v: 1 + i for i, (v, _) in enumerate(out_slots)}  # the position is output slot 0
    n_out = len(out_slots) + 1
    dropped_vars = set(in_vars) | set(out_vars) | set(in_builtins) | {pos_var}

    x = _E()
    x.cpre, x.clpre = "%gsc_k", "%gsc_cl"
    x.t_uint, x.t_bool = uint, boolt or "%gsc_bool"
    x.t_ulong = ulong or "%gsc_ulong"
    x.t_psb_u, x.t_psb_v4 = "%gsc_ptr_psb_u", "%gsc_ptr_psb_v4"
    pcu = pcu or "%gsc_pcu"

    def pc_member_u(m, comp=None):
        p, w = x.tmp("p"), x.tmp("w")
        idx = x.c(m) + (f" {x.c(comp)}" if comp is not None else "")
        x.body += [f"{p} = OpAccessChain {pcu} {pc_var} {idx}", f"{w} = OpLoad {uint} {p}"]
        return w

    def vb_slot(slot, comp):
        a = x.add(x.mul(slot, x.c(3)), x.c(comp))
        p, w = x.tmp("p"), x.tmp("w")
        x.body += [f"{p} = OpAccessChain {pcu} {pc_var} {x.c(n_members + VS_INPUT_SLOTS)} {a}",
                   f"{w} = OpLoad {uint} {p}"]
        return w

    def va64(lo, hi):
        l, h = x.u64(lo), x.u64(hi)
        s = x.op("ShiftLeftLogical", x.t_ulong, h, x.cl(32))
        return x.op("BitwiseOr", x.t_ulong, s, l)

    # The heap member's shape differs by origin: the translator declares two uint members, glslang one uvec2.
    m1 = types[pc_struct][1][1]
    if types[m1][0] == "OpTypeVector":
        heap_lo, heap_hi = pc_member_u(1, 0), pc_member_u(1, 1)
    else:
        heap_lo, heap_hi = pc_member_u(1), pc_member_u(2)
    heap64 = va64(heap_lo, heap_hi)
    params64 = x.add64(heap64, va64(pc_member_u(n_members + 0), pc_member_u(n_members + 1)))
    in64 = x.add64(heap64, va64(pc_member_u(n_members + 2), pc_member_u(n_members + 3)))
    idx64 = x.add64(heap64, va64(pc_member_u(n_members + 4), pc_member_u(n_members + 5)))
    count, flag, first = (pc_member_u(n_members + 6), pc_member_u(n_members + 7), pc_member_u(n_members + 8))
    nin = x.load_dword(params64, 0)
    vin = x.load_dword(params64, 1)
    prims = x.udiv(count, vin)
    slots = x.mul(prims, vin)
    x.body += [f"%gsc_gidv = OpLoad %gsc_v3u %gsc_gid", f"%gsc_pv = OpCompositeExtract {uint} %gsc_gidv 0"]
    # The prologue runs before any of the module's own code; the branches below split only the entry block,
    # which no OpPhi can reference, so the body's control flow is untouched.
    past = x.ge("%gsc_pv", slots)
    x.body += ["OpSelectionMerge %gsc_body None", f"OpBranchConditional {past} %gsc_ret %gsc_body",
               "%gsc_ret = OpLabel", "OpReturn", "%gsc_body = OpLabel"]
    zero = x.eq("%gsc_pv", x.c(0))
    x.body += ["OpSelectionMerge %gsc_cntdone None", f"OpBranchConditional {zero} %gsc_cntdo %gsc_cntdone",
               "%gsc_cntdo = OpLabel"]
    x.store_word(in64, 0, prims)
    x.body += ["OpBranch %gsc_cntdone", "%gsc_cntdone = OpLabel"]
    # The source vertex id: flag 2 is the non-indexed base; 0/1 read u16/u32 indices (a non-indexed draw
    # points the index VA anywhere readable, as tools/gs_compute_stages.py's harness scatter documents).
    byte = x.mul("%gsc_pv", x.c(2))
    waddr = x.add64(idx64, x.and64(x.u64(byte), x.cl(4294967292)))
    wv = x.load_word(waddr, 0)
    v16 = x.and_(x.shr(wv, x.mul(x.umod(byte, x.c(4)), x.c(8))), x.c(65535))
    v32 = x.load_dword(idx64, x.mul("%gsc_pv", x.c(4)))
    vid = x.sel_u(x.eq(flag, x.c(1)), v32, v16)
    vid = x.sel_u(x.eq(flag, x.c(2)), "%gsc_pv", vid)
    vid = x.add(vid, first)

    out_head, out_body = [], []
    pos_targets, dropped_ptrs = set(), set()
    in_main = False
    entry_label_seen = False
    prologue_flushed = False
    last_cap = last_ext = last_memb = -1
    struct_line = f"{pc_struct} = OpTypeStruct " + " ".join(types[pc_struct][1])
    additions_where = None
    for ln in text.splitlines():
        s = ln.strip()
        if not in_main:
            if s == struct_line:
                # The array type and its length constant precede the struct line that names them.
                out_head.append(f"{x.c(3 * INPUT_SLOTS_MAX)} = OpConstant {uint} {3 * INPUT_SLOTS_MAX}")
                out_head.append(f"%gsc_arr18 = OpTypeArray {uint} {x.c(3 * INPUT_SLOTS_MAX)}")
                out_head.append(struct_line + " " + " ".join([uint] * VS_INPUT_SLOTS) + " %gsc_arr18")
                continue
            if s.startswith("OpEntryPoint Vertex "):
                out_head.append(f'OpEntryPoint GLCompute {entry} "main" %gsc_gid {pc_var}')
                out_head.append(f"OpExecutionMode {entry} LocalSize 64 1 1")
                continue
            if s == "OpMemoryModel Logical GLSL450":
                out_head.append("OpMemoryModel PhysicalStorageBuffer64 GLSL450")
                continue
            m = re.match(r"^OpDecorate (%\w+) (?:Location|BuiltIn)", s)
            if m and m.group(1) in dropped_vars:
                continue
            m = re.match(r"^OpName (%\w+)", s)
            if m and m.group(1) in dropped_vars:
                continue
            m = re.match(r"^(%\w+) = OpVariable (%\w+) (?:Input|Output)$", s)
            if m and m.group(1) in dropped_vars:
                continue
            out_head.append(ln)
            if s.startswith("OpCapability"):
                last_cap = len(out_head) - 1
            elif s.startswith("OpExtension"):
                last_ext = len(out_head) - 1
            elif s.startswith(f"OpMemberDecorate {pc_struct} "):
                last_memb = len(out_head) - 1
            elif s.startswith(f"{entry} = OpFunction"):
                in_main = True
            continue
        # ---- main's body ----
        if not entry_label_seen:
            out_body.append(ln)
            if re.match(r"^%\w+ = OpLabel$", s):
                entry_label_seen = True
            continue
        if not prologue_flushed:
            # The function's own OpVariable declarations open the entry block; the prologue follows them.
            if re.match(r"^%\w+ = OpVariable %\w+ Function$", s):
                out_body.append(ln)
                continue
            out_body += x.body
            x.body = []
            prologue_flushed = True
            # this line falls through to the handlers below
        m = re.match(r"^(%\w+) = OpLoad (%\w+) (%\w+)$", s)
        if m and m.group(3) in in_vars:
            rank = in_rank[in_vars[m.group(3)]]
            slot = x.load_dword(params64, 3 + n_out + rank)
            off = x.load_dword(params64, 3 + n_out + n_in + rank)
            vb_lo, vb_hi, stride = vb_slot(slot, 0), vb_slot(slot, 1), vb_slot(slot, 2)
            addr = x.add64(x.add64(heap64, va64(vb_lo, vb_hi)), x.u64(x.add(x.mul(vid, stride), off)))
            words = [x.load_word(addr, 4 * k) for k in range(4)]
            fs = [x.op("Bitcast", f32, w) for w in words]
            out_body += x.body
            x.body = []
            out_body.append(f"{m.group(1)} = OpCompositeConstruct {v4f} " + " ".join(fs))
            continue
        if m and m.group(3) in in_builtins:
            name = in_builtins[m.group(3)]
            src = {"VertexIndex": vid, "BaseVertex": first}.get(name, x.c(0))
            out_body += x.body
            x.body = []
            if m.group(2) == uint:
                out_body.append(f"{m.group(1)} = OpCopyObject {uint} {src}")
            else:
                out_body.append(f"{m.group(1)} = OpBitcast {m.group(2)} {src}")
            continue
        m = re.match(r"^(%\w+) = OpAccessChain (%\w+) (%\w+) (.*)$", s)
        if m and m.group(3) == pos_var:
            idx = m.group(4).split()
            if len(idx) == 1 and const_vals.get(idx[0]) == 0:
                pos_targets.add(m.group(1))
            else:
                dropped_ptrs.add(m.group(1))
            continue
        if m and (m.group(3) in in_vars or m.group(3) in out_vars):
            raise Unsupported(f"vs_input: component access into {m.group(3)} is not supported")
        m = re.match(r"^OpStore (%\w+) (%\w+)$", s)
        if m and (m.group(1) in pos_targets or m.group(1) in out_vars):
            j = 0 if m.group(1) in pos_targets else out_slot_of[m.group(1)]
            # An unconsumed output goes to this slot's own scratch vec4 in the region past the dense rows
            # (slots*nin + pv), never into the next row: a dense-row scratch would race the next slot's own
            # position store within the dispatch.
            perm = x.load_dword(params64, 3 + j)
            dense = x.add(x.mul("%gsc_pv", nin), perm)
            scratch = x.add(x.mul(slots, nin), "%gsc_pv")
            idx = x.sel_u(x.eq(perm, x.c(4294967295)), scratch, dense)
            base = x.add64(in64, x.u64(x.mul(idx, x.c(16))))
            x.store_vec4(base, 16, m.group(2))
            out_body += x.body
            x.body = []
            continue
        if m and m.group(1) in dropped_ptrs:
            continue
        out_body.append(ln)
    # The head still needs: the capability and extension, the block member offsets, and my declarations.
    out_head[last_cap + 1:last_cap + 1] = ["OpCapability PhysicalStorageBufferAddresses", "OpCapability Int64"]
    # The extension line goes after the last extension the module already declares, or right after the
    # two new capabilities when it declares none (both shifted by the splice above).
    out_head.insert(last_ext + 3 if last_ext >= 0 else last_cap + 3,
                    'OpExtension "SPV_KHR_physical_storage_buffer"')
    mem_decos = [f"OpMemberDecorate {pc_struct} {n_members + i} Offset {144 + 4 * i}" for i in range(VS_INPUT_SLOTS)]
    mem_decos += [f"OpMemberDecorate {pc_struct} {n_members + VS_INPUT_SLOTS} Offset {144 + 4 * VS_INPUT_SLOTS}",
                  "OpDecorate %gsc_arr18 ArrayStride 4"]
    where = max(last_memb, 0)
    out_head[where + 1:where + 1] = mem_decos
    extra_types = []
    if boolt is None:
        extra_types.append("%gsc_bool = OpTypeBool")
    if ulong is None:
        extra_types.append("%gsc_ulong = OpTypeInt 64 0")
    if pcu == "%gsc_pcu":
        extra_types.append(f"%gsc_pcu = OpTypePointer PushConstant {uint}")
    extra_types += [f"%gsc_ptr_psb_u = OpTypePointer PhysicalStorageBuffer {uint}",
                    f"%gsc_ptr_psb_v4 = OpTypePointer PhysicalStorageBuffer {v4f}",
                    "%gsc_v3u = OpTypeVector " + f"{uint} 3",
                    "%gsc_ptr_in_v3u = OpTypePointer Input %gsc_v3u"]
    x.consts.pop(3 * INPUT_SLOTS_MAX, None)   # its line sits right before the struct, above
    consts = [f"{n} = OpConstant {x.t_uint} {v}" for v, n in sorted(x.consts.items())]
    uls = [f"{n} = OpConstant {x.t_ulong} {v}" for v, n in sorted(x.ulongs.items())]
    fn_at = next(i for i, l in enumerate(out_head) if l.strip().startswith(f"{entry} = OpFunction"))
    out_head[fn_at:fn_at] = extra_types + consts + uls + ["%gsc_gid = OpVariable %gsc_ptr_in_v3u Input"]
    return "\n".join(out_head + out_body) + "\n"


def build_for_index(spirv_dir, names=None, layer=False):
    """Write the generated stages for every index entry with a "gs_compute" layout: the harness flavor
    (<name>.gs_scatter.spv, <name>.gs_vs.spv) for the vk_harness path, or with layer=True the draw vertex
    shader's layer flavor (<name>.gs_vs_bda.spv), whose parameters ride the draw's own s0..s4."""
    with open(os.path.join(spirv_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    written = []
    for name, entry in sorted(index.items()):
        lay = entry.get("gs_compute")
        if not lay or (names and name not in names):
            continue
        stages = (("gs_scatter", scatter_module), ("gs_vs", draw_vs_module)) if not layer else \
            (("gs_vs_bda", lambda l: draw_vs_module(l, pc_mode="s")),)
        for suffix, module in stages:
            try:
                text = module(lay)
            except Unsupported as e:
                print(f"WARNING: {name}: {e}", file=sys.stderr)
                continue
            written.append(_write_stage(spirv_dir, name, suffix, text))
    return written


def vs_input_for(spirv_dir, name):
    """Write <name>.gs_input.spv, the layer's fused fetch + vertex shader + regroup pass, from the vertex
    shader's own disassembly (<name>.spvasm next to index.json)."""
    src = Path(os.path.join(spirv_dir, f"{name}.spvasm"))
    if not src.is_file():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return None
    text = src.read_text(encoding="utf-8")
    try:
        module = vs_input_module(text)
    except Unsupported as e:
        print(f"WARNING: {name}: {e}", file=sys.stderr)
        return None
    return _write_stage(spirv_dir, name, "gs_input", module, version_of(text))


def version_of(text):
    """The SPIR-V version of a disassembled module, as a spirv-as --target-env value."""
    m = re.search(r"^; Version: (\d+)\.(\d+)$", text, re.M)
    if not m:
        return "vulkan1.2"
    minor = int(m.group(2))
    return {3: "vulkan1.0", 4: "vulkan1.1", 5: "vulkan1.2"}.get(minor, "vulkan1.3")


def _write_stage(spirv_dir, name, suffix, text, target="vulkan1.2"):
    """Assemble and validate one generated stage; raises RuntimeError with the tool's tail on rejection."""
    Path(os.path.join(spirv_dir, f"{name}.{suffix}.spvasm")).write_text(text, encoding="utf-8", newline="\n")
    spv_path = os.path.join(spirv_dir, f"{name}.{suffix}.spv")
    r = subprocess.run(["spirv-as", "--target-env", target, "-o", spv_path, "-"],
                       input=text.encode(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}.{suffix}: spirv-as rejected the module:\n"
                           f"{r.stderr.decode(errors='replace')[:500]}")
    r = subprocess.run(["spirv-val", "--target-env", target, spv_path], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}.{suffix}: spirv-val rejected the module:\n"
                           f"{r.stderr.decode(errors='replace')[:500]}")
    print(f"{name}.{suffix}.spv assembled and validated")
    return os.path.basename(spv_path)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 0 if argv else 2
    layer, vs_input = "--layer" in argv, "--vs-input" in argv
    argv = [a for a in argv if a not in ("--layer", "--vs-input")]
    spirv_dir = argv[0]
    if not os.path.isfile(os.path.join(spirv_dir, "index.json")):
        print(f"ERROR: {spirv_dir}/index.json not found", file=sys.stderr)
        return 2
    if vs_input:
        if len(argv) < 2:
            print("--vs-input needs the vertex shader's index name", file=sys.stderr)
            return 2
        for name in argv[1:]:
            vs_input_for(spirv_dir, name)
        return 0
    build_for_index(spirv_dir, set(argv[1:]) or None, layer=layer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

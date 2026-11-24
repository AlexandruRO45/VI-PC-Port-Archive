"""Optional translation of the pinned package input for comparison with the fixture."""
from pathlib import Path
import hashlib
import re
import sys

def configure(repo, binary, llvm):
    global isa, branch, skip, need, selected
    sys.path.insert(0, str(repo.resolve() / "tools"))
    from pixel_aot import disassemble

    code = binary.read_bytes()
    assert hashlib.sha256(code).hexdigest() == "407c213a5cb6c92c08509f7a8783550d775fe5518399eab3eecf4c736a78e4ed"
    isa = disassemble(code, str(llvm.resolve()), cpu=b"gfx1030", offsets=True)
    ops = dict(isa)
    branch = 0x2410
    mn, offset = ops[branch].split()
    assert mn == "s_cbranch_scc0"
    skip = branch + 4 + 4 * int(offset)
    assert skip == 0x26d8
    assert next(pc for pc, op in isa if branch < pc < skip and op.startswith("ds_write")) == 0x25b0

    # Backward dataflow slice, ending with the SCC used by the branch. Retain two
    # more registers for measurements of the signed footprint comparisons.
    need = {"scc", "s97", "s98"}
    selected = []
    for pc, line in reversed(isa):
        if not 0x22ac <= pc < branch: continue
        op, operands = line.split(" ", 1)
        args = operands.split(", ")
        cmp = op.startswith("s_cmp_")
        dst = "scc" if cmp else args[0]
        src = args if cmp else args[1:]
        sets_scc = op.startswith(("s_and_", "s_add_", "s_sub_", "s_lshl_", "s_bcnt"))
        writes = {dst, "scc"} if sets_scc else {dst}
        if not writes & need: continue
        selected.append((pc,op,dst,src))
        need -= writes
        if op.startswith("v_fmac_"):
            need.add(dst)
        need.update("scc" if t=="src_scc" else t for t in src
                    if re.fullmatch(r"[vs]\d+|src_scc|exec_lo", t))
    assert need == {"v1","v2","s28","s29","s30","s31","exec_lo"}, need

def store_replay():
    # Wave 0 has a zero base (v0 = 192 * (group-thread-index >> 5)).
    # The six rows below label the post-colour-transform source registers.
    # The stores and their immediate offsets come from the pinned input.
    sources = [(11,12,13), (17,18,19), (21,22,23), (25,26,27), (46,47,48), (49,50,52)]
    lines = ["for(uint j=lane;j<576;j+=32) tile[j]=0;",
             "GroupMemoryBarrierWithGroupSync();", "if(scc){", "uint addr0=0,addr1=0,addr54=0,addr55=0;"]
    for row, registers in enumerate(sources):
        for component, register in enumerate(registers):
            lines.append(f"uint tag{register}=1+({row}*32+lane)*3+{component};")
    stores = 0
    for pc, instruction in isa:
        if not branch < pc < skip:
            continue
        op, _, rest = instruction.partition(" ")
        if op in ("ds_write_b32", "ds_write2_b32"):
            operands, _, modifiers = rest.partition(" offset")
            fields = operands.split(", ")
            offsets = dict((key, int(value)) for key, value in
                           re.findall(r"offset(\d*):(\d+)", "offset"+modifiers))
            for n, source in enumerate(fields[1:]):
                byte_offset = offsets.get("", 0) if op == "ds_write_b32" else 4*offsets.get(str(n), 0)
                base = fields[0].replace("v", "addr")
                lines.append(f"tile[({base}+{byte_offset})/4]=tag{source[1:]};")
                stores += 1
        else:
            args = rest.split(", ")
            if args[0] not in ("v0","v1","v54","v55"):
                continue
            operands = [t.replace("v","addr") for t in args[1:]]
            if op == "v_mbcnt_lo_u32_b32" and operands == ["-1","0"]:
                expr = "lane"
            elif op in ("v_add_nc_u32_e32","v_or_b32_e32","v_mul_u32_u24_e32"):
                symbol = {"v_add_nc_u32_e32":"+","v_or_b32_e32":"|","v_mul_u32_u24_e32":"*"}[op]
                expr = symbol.join(operands)
            else:
                raise ValueError((hex(pc), instruction))
            lines.append(f"{args[0].replace('v','addr')}={expr};")
    assert stores == 18
    lines += ["}", "GroupMemoryBarrierWithGroupSync();",
              "for(uint j=lane;j<576;j+=32) result[g.x*576+j]=tile[j];"]
    return lines

def generate(cases, inclusive_x=False, stores=False):
    lines = ["RWStructuredBuffer<uint> result:register(u0);",
             "groupshared uint counts[32], laneWords[32];",
             "[numthreads(32,1,1)] void main(uint lane:SV_GroupIndex,uint3 g:SV_GroupID){",
             "uint width=0,height=0,scaleX=0,scaleY=0; switch(g.x) {"]
    for i,(w,h,x,y) in enumerate(cases):
        lines.append(f"case {i}: width={w};height={h};scaleX={x};scaleY={y};break;")
    lines += ["}", "uint v1=(lane&1)|((lane>>1)&14),v2=(lane>>1)&1;",
              "uint live=(v1<width/2 && v2<height/2); counts[lane]=live;",
              "GroupMemoryBarrierWithGroupSync(); uint pop=0; for(uint i=0;i<32;i++) pop+=counts[i];",
              "uint s28=asuint(scaleX/256.f),s29=asuint(scaleY/256.f),s30=0,s31=0,scc=0;"]
    declared = set(need) | {"scc"}
    for pc,op,dst,src in reversed(selected):
        args = ["scc" if t=="src_scc" else t for t in src]
        a=args[0]; b=args[1] if len(args)>1 else "0"
        if op.startswith(("v_readfirstlane","v_readlane")):
            # Lane zero is always live in this group for the tested dimensions.
            index="0" if op.startswith("v_readfirst") else b
            lines += [f"laneWords[lane]={a};","GroupMemoryBarrierWithGroupSync();"]
            expr=f"laneWords[{index}]"
        elif op.startswith("s_bcnt"): expr="pop"
        elif op.startswith("s_cmp_eq"): expr=f"({a}=={b})"
        elif op.startswith("s_cmp_lg"): expr=f"({a}!={b})"
        elif op.startswith("s_cmp_lt"):
            comparison = "<=" if inclusive_x and b == "33" else "<"
            expr=f"(asint({a}){comparison}asint({b}))"
        elif op.startswith("s_mov"): expr=a
        elif op.startswith("s_and"): expr=f"{a}&{b}"
        elif op.startswith("s_add"): expr=f"{a}+{b}"
        elif op.startswith("s_sub"): expr=f"{a}-{b}"
        elif op.startswith("v_or"): expr=f"{a}|{b}"
        elif op.startswith("v_lshlrev"): expr=f"{b}<<{a}"
        elif op.startswith("v_cvt_f32_u32"): expr=f"asuint(float({a}))"
        elif op.startswith("v_cvt_i32_f32"): expr=f"asuint(int(asfloat({a})))"
        elif op.startswith("v_rndne"): expr=f"asuint(round(asfloat({a})))"
        elif op.startswith("v_add_f32"): expr=f"asuint(asfloat({a})+asfloat({b}))"
        elif op.startswith(("v_fma_","v_fmac_")):
            f=lambda t: f"asfloat({t})" if re.fullmatch(r"[vs]\d+",t) else t
            c=dst if op.startswith("v_fmac") else args[2]
            expr=f"asuint(mad({f(a)},{f(b)},{f(c)}))"
        else: raise ValueError((hex(pc),op))
        lines.append(("" if dst in declared else "uint ")+dst+"="+expr+";")
        declared.add(dst)
        if op.startswith("s_and"): lines.append(f"scc=({dst}!=0);")
        if op.startswith("v_read"): lines.append("GroupMemoryBarrierWithGroupSync();")
    if stores:
        lines.insert(1, "groupshared uint tile[576];")
        lines += store_replay()
    else:
        lines.append("if(lane==0){uint k=g.x*4;result[k]=pop;result[k+1]=s98;result[k+2]=s97;result[k+3]=scc;}")
    lines.append("}")
    return "\n".join(lines)

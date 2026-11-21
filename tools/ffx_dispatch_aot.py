"""Lower the two pinned compute programs for the bounded 8x4 dispatch experiment."""
import argparse
import hashlib
from pathlib import Path
import re
import struct

from pixel_aot import disassemble


DIGEST = {
    "producer": "407c213a5cb6c92c08509f7a8783550d775fe5518399eab3eecf4c736a78e4ed",
    "rcas": "35f9cb0f301653b64a28fd3a0f2a6509abc4a95b565f10d2251711c0ff0c349a",
}


def word(token, packed=False):
    if re.fullmatch(r"[sv]\d+", token):
        return token
    if token == "src_scc":
        return "scc"
    if "." in token:
        fmt = "<e" if packed else "<f"
        value = int.from_bytes(struct.pack(fmt, float(token)), "little")
    else:
        value = int(token, 0) & 0xffffffff
    return f"0x{value:x}u"


def registers(token):
    match = re.fullmatch(r"([vs])\[(\d+):(\d+)\]", token)
    if match:
        prefix, start, end = match.groups()
        return [prefix + str(i) for i in range(int(start), int(end)+1)]
    return re.findall(r"[sv]\d+", token)


def assign(names, values):
    if len(names) != len(values):
        raise ValueError(f"Register count mismatch: {names} {values}")
    return " ".join(f"{n} = {v};" for n, v in zip(names, values))


# Operands remain raw DWORDs across integer, IEEE float and packed-half instructions.
OPS = {
    "mov_b32": "{0}", "movk_i32": "{0}", "add_u32": "{0}+{1}", "add_nc_u32": "{0}+{1}",
    "addk_i32": "{d}+{0}", "sub_i32": "{0}-{1}", "subrev_nc_u32": "{1}-{0}",
    "mul_i32": "{0}*{1}", "mul_lo_u32": "{0}*{1}", "mul_u32_u24": "({0}&0xffffff)*({1}&0xffffff)",
    "add3_u32": "{0}+{1}+{2}", "or_b32": "{0}|{1}", "and_b32": "{0}&{1}",
    "xor_b32": "{0}^{1}", "and_or_b32": "({0}&{1})|{2}",
    "lshl_or_b32": "({0}<<{1})|{2}", "lshl4_add_u32": "({0}<<4)+{1}",
    "lshl_add_u32": "({0}<<{1})+{2}", "lshlrev_b32": "{1}<<{0}", "lshrrev_b32": "{1}>>{0}",
    "lshr_b32": "{0}>>{1}", "lshl_b32": "{0}<<{1}", "brev_b32": "reversebits({0})",
    "max_i32": "max(asint({0}),asint({1}))", "min_i32": "min(asint({0}),asint({1}))",
    "max_u32": "max({0},{1})", "abs_i32": "abs(asint({0}))",
    "med3_i32": "median_i(asint({0}),asint({1}),asint({2}))",
    "bfe_u32": "({0}>>{1}) & ((1u<<{2})-1)", "bfe_i32": "asint({0}<<(32-{1}-{2}))>>(32-{2})",
    "cvt_f32_i32": "asuint(float(asint({0})))", "cvt_f32_u32": "asuint(float({0}))",
    "cvt_i32_f32": "uint(int(asfloat({0})))", "cvt_u32_f32": "uint(asfloat({0}))",
    "cvt_pk_u8_f32": "insert_byte({2},{1},uint(round(clamp(asfloat({0}),0.0,255.0))))",
    "dot4_i32_i8": "idot({0},{1},{2})", "dot4c_i32_i8": "idot({0},{1},{d})",
    "pack_b32_f16": "({0}&65535u)|({1}<<16)",
}

FLOAT = {
    "add_f32": "{0}+{1}", "sub_f32": "{0}-{1}", "mul_f32": "{0}*{1}", "exp_f32": "exp2({0})",
    "log_f32": "log2({0})", "rndne_f32": "round({0})", "rcp_f32": "rcp({0})",
    "min_f32": "min({0},{1})", "max_f32": "max({0},{1})", "med3_f32": "median_f({0},{1},{2})",
    "fma_f32": "mad({0},{1},{2})", "fmamk_f32": "mad({0},{1},{2})", "fmac_f32": "mad({0},{1},{d})",
}


def packed_op(op, dest, inputs, mods):
    if op == "pk_fmac_f16":
        inputs = [*inputs, dest]
    opts = {k: list(map(int, v.split(","))) for k, v in re.findall(r"(\w+):\[([0-1,]+)\]", mods)}
    operations = {"add": "{0}+{1}", "sub": "{0}-{1}", "min": "min({0},{1})",
                  "max": "max({0},{1})", "mul": "{0}*{1}", "fma": "mad({0},{1},{2})",
                  "fmac": "mad({0},{1},{2})"}
    _, operation, kind = op.split("_")
    result = []
    for lane in (0, 1):
        args = []
        select = opts.get(("op_sel", "op_sel_hi")[lane], [lane]*len(inputs))
        negate = opts.get(("neg_lo", "neg_hi")[lane], [0]*len(inputs))
        for token, high, neg in zip(inputs, select, negate):
            arg = f"(({word(token, packed=True)} >> {16*high}) & 65535u)"
            if kind == "f16":
                arg = f"f16tof32({arg})"
            args.append(("-" if neg else "") + arg)
        expr = operations[operation].format(*args)
        if "clamp" in mods:
            expr = f"saturate({expr})"
        result.append(f"f32tof16({expr})" if kind == "f16" else f"(({expr})&65535u)")
    return f"{dest} = {result[0]} | ({result[1]}<<16);"


def alu(mn, tokens, mods):
    op = re.sub(r"_e(?:32|64)$", "", mn[2:])
    dest, *inputs = tokens
    if op.startswith("pk_"):
        return packed_op(op, dest, inputs, mods)
    if op.endswith("_sdwa") or op == "rcp_f16":
        def high(field):
            return int(f"{field}:WORD_1" in mods)
        args = [f"(({reg} >> {16*high(f'src{i}_sel')}) & 65535u)" for i, reg in enumerate(inputs)]
        expr = args[0]
        if op.startswith("rcp"):
            expr = f"f32tof16(rcp(f16tof32({expr})))"
        elif op.startswith("xor"):
            expr += "^" + args[1]
        elif op != "mov_b32_sdwa":
            raise ValueError(op)
        lane = high("dst_sel")
        return f"{dest} = ({dest}&{(65535, 0xffff0000)[1-lane]}u) | (({expr})<<{16*lane});"
    if op == "swap_b32":
        return f"{{ uint t={dest}; {dest}={inputs[0]}; {inputs[0]}=t; }}"
    if op == "fma_mix_f32":
        if inputs[2] != "neg(0)" or "op_sel_hi:[0,1,0]" not in mods:
            raise ValueError("Unexpected mixed-precision operands")
        shift = 16 if "op_sel:[0,1,0]" in mods else 0
        return f"{dest}=asuint(asfloat({inputs[0]})*f16tof32({inputs[1]}>>{shift}));"
    if op in ("mov_b64", "lshrrev_b64"):
        if op == "lshrrev_b64" and inputs[0] != "0":
            raise ValueError("Unsupported 64-bit shift")
        a, b = registers(inputs[-1])
        x, y = registers(dest)
        return f"{{ uint2 t=uint2({a},{b}); {x}=t.x; {y}=t.y; }}"
    args = list(map(word, inputs))
    if op == "dot4_i32_i8" and mods:
        if mods != "op_sel:[0,1,0] op_sel_hi:[1,0,1]":
            raise ValueError(mods)
        args[1] = f"(({args[1]}<<16)|({args[1]}>>16))"
    if mn == "s_bfe_u32":
        args = [args[0], str(int(inputs[1], 0) & 31), str(int(inputs[1], 0) >> 16)]
    if op in FLOAT:
        expr = FLOAT[op].format(*[f"asfloat({a})" for a in args], d=f"asfloat({dest})")
        if mods == "mul:2":
            expr = f"2.0*({expr})"
        return f"{dest} = asuint({expr});"
    return f"{dest} = {OPS[op].format(*args, d=dest)};"


# Constants for a single active 8x4 tile. Earlier graph jobs supply zero scratch,
# two input surfaces and unit exposure. The sharpen flag at b1+0x5c is one.
PARAMS = {0x10: [8,4,8,8], 0x28: [8,4], 0x1fc0: [1.0], 0x1fc8: [4,1.0,1.0,1],
          0x1ff8: [1.0,1.0,0,0], 0x26f0: [1.0,1.0],
          0x178: [0x38003800], 0x180: [1.0], 0x188: [1.0]}


def image_op(name, mn, tokens, mods):
    if mn not in ("image_load", "image_store"):
        raise ValueError(f"Unsupported image access: {mn}")
    data, addr, desc = tokens
    regs = registers(data)
    xy = registers(addr)
    coord = f"uint2({','.join(xy)})" if len(xy) == 2 else f"uint2({xy[0]} & 65535u, {xy[0]}>>16)"
    if mn == "image_load":
        if name == "rcas":
            if desc != "s[24:31]" or "dmask:0x7" not in mods or "d16" not in mods:
                raise ValueError("Unexpected RCAS load")
            return "{ uint3 h=f32tof16(load16(" + coord + ").rgb); " + assign(regs, ["h.x|(h.y<<16)", "h.z"]) + " }"
        source = {"s[32:39]": "historyInput", "s[56:63]": "sceneInput"}[desc]
        return "{ uint3 t=asuint(" + source + ".Load3(12*(" + coord + ".y*8+" + coord + ".x))); " + assign(regs, ["t.x","t.y","t.z"]) + " }"
    value = f"asfloat(uint4({','.join(regs)}))"
    if name == "rcas":
        if desc != "s[16:23]":
            raise ValueError("Unexpected RCAS output")
        return f"finalOutput[{coord}]={value};"
    target = {"s[40:47]": "recurrent", "s[48:55]": "historyOutput", "s[64:71]": "intermediate"}[desc]
    return f"write16({coord},{value});" if target == "intermediate" else f"{target}[{coord}]={value};"


def literal_operands(code, pc, tokens):
    """LLVM prints a 16-bit literal 0x0000ffff as -1. Keep both encoded halves."""
    sources, = struct.unpack_from("<I", code, pc + 4)
    for i in range(1, len(tokens)):
        if (sources >> (9 * (i-1))) & 511 == 255:
            literal, = struct.unpack_from("<I", code, pc + 8)
            tokens[i] = hex(literal)


def translate(name, instructions, machine_code):
    producer = name == "producer"
    signature = "PRODUCER" if producer else "CONSUMER"
    numthreads = 256 if producer else 64
    lines = ['#include "ffx_dispatch_io.hlsli"', f'[RootSignature({signature}_ROOT)]',
             f'[numthreads({numthreads},1,1)] void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {{',
             "uint " + ",".join(f"s{i}=0" for i in range(102)) + ";",
             "uint " + ",".join(f"v{i}=0" for i in range(84)) + ";",
             "uint scc=0; v0=lane;", "s14=group.x; s15=group.y;" if producer else "s8=group.x; s9=group.y;"]
    skip_until = 0
    for pc, instruction in instructions:
        if pc < skip_until:
            continue
        mn, _, rest = instruction.partition(" ")
        # Bracketed image coordinates contain a comma and a space.
        rest = re.sub(r"\[(v\d+), (v\d+)\]", r"[\1,\2]", rest.strip())
        pieces = re.split(r" (?=\w+:|clamp$)", rest, maxsplit=1)
        tokens = pieces[0].split(", ")
        mods = pieces[1] if len(pieces) == 2 else ""
        if mn.startswith("v_pk_") and not mn.startswith("v_pk_fmac_"):
            literal_operands(machine_code, pc, tokens)
        if mn in ("s_endpgm", "s_clause", "s_waitcnt", "s_inst_prefetch") or mn.startswith("s_load_dword"):
            continue  # Descriptor loads map to the D3D12 bindings in image_op.
        if mn.startswith("s_buffer_load"):
            code = assign(registers(tokens[0]), [word(str(x)) for x in PARAMS[pc]])
        elif mn == "buffer_load_dwordx4":
            code = assign(registers(tokens[0]), ["0"]*4)
        elif mn.startswith("image_"):
            code = image_op(name, mn, tokens, mods)
        elif producer and pc == 0xa4:
            # Per-lane form of the two compares, NOR exec mask and branch to endpgm.
            code = "if (v1>=s20 || v2>=s21) return;"
            skip_until = 0xb8
        elif producer and pc in (0x2410, 0x2a3c, 0x40c8, 0x4828, 0x4f84, 0x568c):
            # At most eight lanes survive. The full-wave LDS path is unreachable.
            # With sharpening enabled all four outputs retain their log-domain values.
            if mn != "s_cbranch_scc0":
                raise ValueError("Changed producer branch")
            skip_until = pc + 4 + 4*int(rest)
            code = "scc=0;"
        elif producer and (mn.startswith(("v_readfirstlane", "v_readlane", "s_bcnt")) or
                           (0x231c <= pc < 0x2410 and mn.startswith("s_"))):
            continue  # Scalar wave extents feed only the excluded LDS path.
        elif mn.startswith("s_cmp_"):
            cmp = {"s_cmp_eq_i32": "==", "s_cmp_lg_i32": "!="}[mn]
            code = f"scc = {word(tokens[0])}{cmp}{word(tokens[1])};"
        else:
            code = alu(mn, tokens, mods)
        lines.append("    " + code)
    return "\n".join([*lines, "}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--llvm", required=True)
    args = parser.parse_args()
    for name, digest in DIGEST.items():
        code = (args.directory / (name + ".bin")).read_bytes()
        if hashlib.sha256(code).hexdigest() != digest:
            raise ValueError(f"Unknown {name} machine code")
        ins = disassemble(code, args.llvm, cpu=b"gfx1030" if name == "producer" else b"gfx1013", offsets=True)
        if name == "rcas":
            accesses = [line.split()[0] for _, line in ins if line.startswith("image_")]
            if accesses.count("image_load") != 20 or accesses.count("image_store") != 4 or len(accesses) != 24:
                raise ValueError("Expected twenty integer texture loads and four output stores")
        shader = translate(name, ins, code)
        (args.directory / (name + ".hlsl")).write_text(shader, encoding="ascii")
        print(f"{name} instructions={len(ins)}")

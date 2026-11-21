"""Offline lowering for a straight-line, single-texture gfx1013 pixel shader."""
import argparse
import ctypes as C
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct
import tempfile
import zlib

from resource_inventory import parts


def inflate(data, size, window):
    if not 0 < size <= 16 << 20:
        raise ValueError("Invalid decoded size")
    decoder = zlib.decompressobj(window)
    image = decoder.decompress(data, size + 1)
    if len(image) != size or not decoder.eof:
        raise ValueError("Incomplete or mismatched compressed stream")
    return image


def native_program(archive, record):
    name = record.encode("ascii")
    if not 0 < len(name) < 255 or b"\0" in name:
        raise ValueError("Invalid length-prefixed record name")
    needle = bytes([len(name) + 1]) + name + b"\0"
    with archive.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        pos = data.find(needle)
        end = pos + len(needle)
        if pos < 0 or end + 17 > len(data):
            raise ValueError("Missing or truncated named record")
        if data.find(needle, end) >= 0:
            raise ValueError("Multiple records match the name")
        stored, _, repeat, size = struct.unpack_from("<4I", data, end + 1)
        if stored != repeat or not 0 < stored <= 8 << 20 or end+17+stored > len(data):
            raise ValueError("Invalid record extent")
        blob = inflate(data[end+17:end+17+stored], size, -15)
    program = parts(blob)[b"DXIL"]
    if len(program) < 24 or struct.unpack_from("<H", program, 2)[0] != 0:
        raise ValueError("Expected a pixel shader")
    table = program.index(b"XDAT", 24)
    if table + 12 > len(program):
        raise ValueError("Truncated XDAT header")
    count, = struct.unpack_from("<I", program, table+4)
    if table + 12 + count*12 > len(program):
        raise ValueError("Truncated XDAT table")
    entries = struct.iter_unpack("<3I", program[table+12:table+12+count*12])
    native_entries = [(o, n) for tag, o, n in entries if tag == 13]
    if len(native_entries) != 1:
        raise ValueError("Expected one native program entry")
    offset, length = native_entries[0]
    if offset < table + 12 + count * 12 or offset+length > len(program) or length < 24 or program[offset:offset+4] != b"XBCH":
        raise ValueError("Invalid native program entry")
    size, = struct.unpack_from("<I", program, offset+16)
    image = inflate(program[offset+24:offset+length], size, 15)
    if len(image) < 196 or image[:4] != b"FSSM":
        raise ValueError("Invalid native program header")
    start = struct.unpack_from("<I", image, 192)[0] + 40
    if start < 196 or start % 4 or start + 4 > len(image):
        raise ValueError("Invalid native code offset")
    ends = [i+4 for i in range(start, len(image)-3, 4) if image[i:i+4] == b"\0\0\x81\xbf"]
    if len(ends) != 1:
        raise ValueError("Expected one aligned, terminated program")
    return blob, image[start:ends[0]]


def disassemble(code, library, *, cpu=b"gfx1013", offsets=False):
    llvm = C.CDLL(library)
    for component in ("TargetInfo", "Target", "TargetMC", "AsmPrinter", "Disassembler"):
        getattr(llvm, "LLVMInitializeAMDGPU" + component)()
    pointer, string = C.c_void_p, C.c_char_p
    llvm.LLVMCreateDisasmCPU.argtypes = [string, string, pointer, C.c_int, pointer, pointer]
    llvm.LLVMCreateDisasmCPU.restype = pointer
    llvm.LLVMDisasmInstruction.argtypes = [pointer, pointer, C.c_uint64, C.c_uint64, pointer, C.c_size_t]
    llvm.LLVMDisasmInstruction.restype = C.c_size_t
    llvm.LLVMDisasmDispose.argtypes = [pointer]
    ctx = llvm.LLVMCreateDisasmCPU(b"amdgcn--amdpal", cpu, None, 0, None, None)
    if not ctx:
        raise ValueError(f"LLVM has no {cpu.decode()} disassembler")
    raw = C.create_string_buffer(code)
    offset = 0
    instructions = []
    try:
        while offset < len(code):
            line = C.create_string_buffer(1024)
            size = llvm.LLVMDisasmInstruction(ctx, C.byref(raw, offset), len(code)-offset,
                                              offset, line, len(line))
            if not size:
                raise ValueError(f"Unknown instruction at byte {offset}")
            decoded = line.value.decode("ascii").strip()
            instructions.append((offset, decoded) if offsets else decoded)
            offset += size
    finally:
        llvm.LLVMDisasmDispose(ctx)
    return instructions


PREAMBLE = """Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);
// Finite binary32 to binary16, round toward zero, then widen for the float target.
float export16(float value) {
    uint h = f32tof16(value);
    if (abs(f16tof32(h)) > abs(value)) h--;
    return f16tof32(h);
}
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {"""


def registers(first, last, count):
    first, last = int(first), int(last)
    if last - first + 1 != count:
        raise ValueError("Descriptor or vector width mismatch")
    return range(first, last+1)


ALU = {
    "log": (1, "log2({0})"), "exp": (1, "exp2({0})"), "rcp": (1, "(1.0f / {0})"),
    "max": (2, "max({0}, {1})"), "min": (2, "min({0}, {1})"), "sub": (2, "({0} - {1})"),
    "mul_legacy": (2, "(({0} == 0 || {1} == 0) ? 0.0f : {0} * {1})"),
    "fma": (3, "mad({0}, {1}, {2})"), "fmac": (3, "mad({0}, {1}, {2})"),
}


def fp_operand(token, scalar, vector, pending, sample_pending):
    if re.fullmatch(r"-\d+", token):
        return f"asfloat({int(token) & 0xffffffff}u)"
    if token.startswith("-"):
        return "(-" + fp_operand(token[1:], scalar, vector, pending, sample_pending) + ")"
    if re.fullmatch(r"[vs]\d+", token):
        if token in pending:
            raise ValueError("Constant read before load completion")
        value = (vector if token.startswith("v") else scalar)[token]
        if isinstance(value, int):
            return f"asfloat({value}u)"
        if not isinstance(value, str) or not value.startswith(("sample", "alu", "params[")):
            raise ValueError("Expected a numeric register")
        if sample_pending and value.startswith("sample"):
            raise ValueError("Sample used before vmcnt wait")
        return value
    if re.fullmatch(r"0x[0-9a-f]{1,8}|\d+", token):
        return f"asfloat({int(token, 0)}u)"
    if re.fullmatch(r"\d+\.\d+", token):
        return token + "f"
    raise ValueError(f"Unsupported arithmetic operand {token}")


def lower(instructions, texture_sgpr, interp_sgpr, sampler, constant_descriptor=None):
    """Track descriptor provenance, interpolation and packed exports through register writes."""
    if (texture_sgpr == interp_sgpr or not 0 <= texture_sgpr <= 105 or not 0 <= interp_sgpr <= 105
            or len(sampler) != 4 or any(not 0 <= word <= 0xffffffff for word in sampler)):
        raise ValueError("Invalid scalar inputs or sampler descriptor words")
    scalar = {f"s{texture_sgpr}": "texture_ptr", f"s{interp_sgpr}": "interp_base"}
    vector = {"v0": "bary_i", "v1": "bary_j"}
    lines = [PREAMBLE]
    if constant_descriptor is not None:
        if constant_descriptor < 0 or constant_descriptor % 4:
            raise ValueError("Invalid constant descriptor offset")
        lines.insert(0, "cbuffer Constants : register(b0) { float4 params[16]; };")
    pending = set()
    mask = "entry"
    samples = 0
    sample_pending = False
    exported = False
    ended = False
    for i, line in enumerate(instructions):
        def match(pattern):
            return re.fullmatch(pattern, line)

        if ended:
            raise ValueError("Instruction follows program end")
        if exported and line != "s_endpgm":
            raise ValueError("Unexpected instruction after color export")
        if match(r"(?:s_inst_prefetch|s_nop) (?:0x[0-9a-f]+|\d+)"):
            continue
        if line == "s_waitcnt lgkmcnt(0)":
            pending.clear()
        elif line == "s_waitcnt vmcnt(0)":
            sample_pending = False
        elif m := match(r"s_mov_b64 s\[(\d+):(\d+)\], exec"):
            regs = registers(*m.groups(), 2)
            for n in regs:
                if f"s{n}" in pending:
                    raise ValueError("Write overlaps an outstanding scalar load")
                scalar[f"s{n}"] = (mask, n-regs.start)
        elif line == "s_wqm_b64 exec, exec" and mask == "entry":
            mask = "quad"
        elif m := match(r"s_and_b64 exec, exec, s\[(\d+):(\d+)\]"):
            pair = registers(*m.groups(), 2)
            if mask != "quad" or [scalar.get(f"s{n}") for n in pair] != [("entry",0),("entry",1)]:
                raise ValueError("Unsupported execution mask")
            mask = "entry"
        elif m := match(r"(s_mov_b32|s_movk_i32) (s\d+|m0), (s\d+|0x[0-9a-f]+|\d+)"):
            op, dst, src = m.groups()
            if src in pending or dst in pending:
                raise ValueError("Unresolved scalar memory dependency")
            val = scalar[src] if src.startswith("s") else int(src, 0)
            if op == "s_movk_i32":
                if not isinstance(val, int) or not 0 <= val <= 0xffff:
                    raise ValueError("Invalid movk immediate")
                val = (val if val < 0x8000 else val - 0x10000) & 0xffffffff
            scalar[dst] = val
        elif m := match(r"s_load_dwordx8 s\[(\d+):(\d+)\], s\[(\d+):(\d+)\], null"):
            dst = registers(*m.groups()[:2], 8)
            addr = registers(*m.groups()[2:], 2)
            if any(f"s{n}" in pending for n in (*addr, *dst)) or [scalar.get(f"s{n}") for n in addr] != ["texture_ptr", 0]:
                raise ValueError("Texture address differs from binding contract")
            for j, n in enumerate(dst):
                scalar[f"s{n}"] = ("texture", j)
                pending.add(f"s{n}")
        elif m := match(r"(s_load_dwordx4|s_buffer_load_dwordx4) s\[(\d+):(\d+)\], s\[(\d+):(\d+)\], (0x[0-9a-f]+|null)"):
            op, first, last, low, high, imm = m.groups()
            offset = 0 if imm == "null" else int(imm, 16)
            descriptor = op == "s_load_dwordx4"
            base = [f"s{n}" for n in registers(low, high, 2 if descriptor else 4)]
            expected = ["texture_ptr", 0] if descriptor else [("cbv", n) for n in range(4)]
            if constant_descriptor is None or pending.intersection(base) or [scalar.get(r) for r in base] != expected:
                raise ValueError("Unresolved constant buffer binding")
            if descriptor and offset != constant_descriptor:
                raise ValueError("Constant descriptor differs from binding contract")
            if not descriptor and (offset % 4 or offset + 16 > 256):
                raise ValueError("Constant read exceeds binding or is unaligned")
            for j, n in enumerate(registers(first, last, 4)):
                if f"s{n}" in pending:
                    raise ValueError("Write overlaps an outstanding scalar load")
                word = offset // 4 + j
                scalar[f"s{n}"] = ("cbv", j) if descriptor else f"params[{word // 4}].{'xyzw'[word % 4]}"
                pending.add(f"s{n}")
        elif m := match(r"v_interp_p([12])_f32_e32 (v\d+), (v\d+), attr0\.([xy])"):
            phase, dst, src, component = m.groups()
            if mask != "quad" or scalar.get("m0") != "interp_base":
                raise ValueError("Unknown interpolation context")
            if phase == "1" and vector[src] == "bary_i":
                vector[dst] = ("partial", component)
            elif phase == "2" and vector[src] == "bary_j" and vector[dst] == ("partial", component):
                vector[dst] = f"uv.{component}"
            else:
                raise ValueError("Unpaired interpolation")
        elif m := match(r"image_sample v\[(\d+):(\d+)\], v\[(\d+):(\d+)\], s\[(\d+):(\d+)\], s\[(\d+):(\d+)\] dmask:0xf dim:SQ_RSRC_IMG_2D"):
            operands = m.groups()
            dest, coords, image, state = [registers(*operands[k:k+2], width)
                                          for k, width in zip((0,2,4,6), (4,2,8,4))]
            if (mask != "entry" or any(f"s{n}" in pending for n in (*image, *state))
                    or [scalar.get(f"s{n}") for n in image] != [("texture",j) for j in range(8)]
                    or [scalar.get(f"s{n}") for n in state] != list(sampler)):
                raise ValueError("Unsupported image or sampler binding")
            uv = [vector[f"v{n}"] for n in coords]
            if any(v not in ("uv.x", "uv.y") for v in uv):
                raise ValueError("Unsupported texture coordinate")
            lines.append(f"    float4 sample{i} = tex.Sample(samp, float2({', '.join(uv)}));")
            samples += 1
            sample_pending = True
            for j, n in enumerate(dest):
                vector[f"v{n}"] = f"sample{i}.{'xyzw'[j]}"
        elif m := match(r"v_(log|exp|rcp|max|min|sub|mul_legacy|fma|fmac)_f32(?:_e32|_e64)? (v\d+), (.+?)( clamp)?"):
            op, dst, operands, clamp = m.groups()
            tokens = operands.split(", ")
            if op == "fmac":
                tokens.append(dst)
            arity, template = ALU[op]
            if mask != "entry" or len(tokens) != arity:
                raise ValueError("Invalid arithmetic operation")
            values = [fp_operand(t, scalar, vector, pending, sample_pending) for t in tokens]
            expression = template.format(*values)
            if clamp:
                expression = f"saturate({expression})"
            vector[dst] = f"alu{i}"
            lines.append(f"    precise float alu{i} = {expression};")
        elif m := match(r"v_cvt_pkrtz_f16_f32_e32 (v\d+), (v\d+), (v\d+)"):
            dst, lo, hi = m.groups()
            pair = (vector[lo], vector[hi])
            if mask != "entry" or sample_pending or not all(isinstance(v, str) and v.startswith(("sample", "alu")) for v in pair):
                raise ValueError("Unsupported pack operand")
            vector[dst] = pair
        elif m := match(r"exp mrt0 (v\d+), \1, (v\d+), \2 done compr vm"):
            pairs = [vector[v] for v in m.groups()]
            if mask != "entry" or exported or not all(isinstance(p, tuple) and len(p)==2 and
                    all(v.startswith(("sample", "alu")) for v in p) for p in pairs):
                raise ValueError("Unsupported export")
            args = [f"export16({v})" for pair in pairs for v in pair]
            lines.append(f"    return float4({', '.join(args)});")
            exported = True
        elif line == "s_endpgm" and exported:
            ended = True
        else:
            raise ValueError(f"Unsupported instruction {i}: {line}")
    if not ended or samples != 1:
        raise ValueError("Expected one sample and a terminated color export")
    return "\n".join([*lines, "}",""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("record")
    parser.add_argument("--llvm", required=True, help="LLVM C library with AMDGPU support")
    parser.add_argument("--texture-sgpr", type=int, required=True, help="Entry register for the low texture address")
    parser.add_argument("--interp-sgpr", type=int, required=True, help="Entry register for the interpolation base")
    parser.add_argument("--sampler-words", nargs=4, type=lambda s: int(s, 0), required=True,
                        help="Four descriptor words to bind to the supplied desktop sampler at s0")
    parser.add_argument("--output", type=Path, required=True, help="Destination .hlsl in a local output directory")
    parser.add_argument("--constant-descriptor", type=lambda s: int(s, 0),
                        help="Byte offset of a descriptor mapped to the 256-byte desktop buffer b0")
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.suffix.lower() != ".hlsl" or dest == args.archive.resolve():
        parser.error("Choose an HLSL destination distinct from the input")
    try:
        blob, code = native_program(args.archive, args.record)
        instructions = disassemble(code, args.llvm)
        source = lower(instructions, args.texture_sgpr, args.interp_sgpr, args.sampler_words, args.constant_descriptor)
    except (ValueError, KeyError, struct.error, OSError, AttributeError, zlib.error) as exc:
        parser.exit(1, f"Translation failed: {exc}\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, mode="wb") as f:
        f.write(source.encode("utf-8"))
    Path(f.name).replace(dest)
    print(json.dumps({"record":args.record, "container_sha256":hashlib.sha256(blob).hexdigest(),
                      "native_sha256":hashlib.sha256(code).hexdigest(), "instructions":len(instructions),
                      "hlsl_sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

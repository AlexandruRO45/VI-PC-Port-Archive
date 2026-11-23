"""Pack hand-written RDNA2 assembly into a synthetic FSSM program inside the title's shader container.

    python -I runtime/test/lanterns/fssm_pack.py <out dir> <shader.asm> [...] [--translate] [--mcpu gfx1013]

It is a test helper: the Lanterns guest (runtime/test/lanterns) builds its pipelines from what it writes. It lives here,
not in tools/: make_game.py's translator_digest() hashes every file in tools/, so a new file there would make a resumed
shaders stage throw away the shards it had already translated, for a script the game build never runs.

Each .asm is assembled with llvm-mc (-triple=amdgcn--amdpal -mcpu=gfx1013 -filetype=obj; LLVM_MC names it, see
mc_command.py), and its code is the object's .text section, so branches to labels are resolved by the assembler. The
code goes into an FSSM blob laid out the way docs/design.md and tools/fssm_header.py describe the game's, and the blob
into the container the game ships (DXBC { DXIL { stub, XDAT { 0x0D: XBCH + zlib(FSSM) } } }, tools/fssm_unpack.py), with an
ISG1 input signature part when the pixel shader reads SV_Position.
The tool writes <out>/<name>.dxbc and a carve index <out>/index.json (name, file, stage), the folder shape
make_game.py gets from shader_carve.py. With --translate it then runs the project's own offline chain on that folder,
as make_game.py does for the game: fssm_unpack.py <out>, shader_disasm.py <out>/fssm --out <out>/disasm --mcpu gfx1013,
rdna2spirv.py <out>/disasm --out <out>/spirv. It checks that every shader came back: the unpacked blob is the packed
one, its listing decoded with no invalid word, and spirv/index.json has it with the stage and fssm_sha256 of the
packed blob, which is the key the runtime looks a PSO's shader up by (docs/d3d12_vulkan.md 3.6). Nothing here comes
from the game: the code is ours, the layout is the measured one.

FSSM fields written (offsets in the blob; every other header byte is zero):
    0x00  'FSSM', u16 version 0x0101
    0x08  u32 blob size
    0xC0  u64 code-section offset (HEADER_SIZE); the instructions start CODE_SKIP (0x28) bytes into the section
    0xC8  u64 code-section size: CODE_SKIP + code + padding + data
    0xD4  u32 flags: bit 5 (0x20) set for wave32, clear for wave64 (fssm_header.wave_size)
    0xE8  u64 code end: section offset + size (fssm_header.binding_table)
    0x120 / 0x12C / 0x130  u32 section offset, size, and offset + size (fssm_header.directory_ok)
    0x138 PS: one 8-byte input record per "; input:" line (kind 6, semantic, mask 0xF, index, slot, read method);
          CS: COMPUTE_PGM_RSRC2 and the thread-group size x, y, z (docs/design.md, "Compute shaders")
    code end: 4 x 0xBF9F0000 (s_code_end), u32 0, u32 0, no binding records, 0xFFFFFFFF, u32 0, u16 0, u16 name
          length, the name (fssm_header.binding_table)

Header comments an .asm may carry (before its first instruction):
    ; stage: vs | ps | cs        default: the name's prefix (vs_, ps_, cs_)
    ; wave: 32 | 64              default 32
    ; input: flat | interp [fp16]  PS only, one per table position, in v_interp attr order
    ; data-at: 0xOFFSET          bytes from the first instruction to the data words (what s_getpc_b64 + K reaches)
    ; data: 0xWORD ...           u32 words placed there, after the code's s_endpgm; s_code_end fills the gap
    ; group: X,Y,Z  ; rsrc2: 0x...  CS only (thread-group size and COMPUTE_PGM_RSRC2)
    ; position: xy               PS only: the shader reads SV_Position's x and y (any of xyzw). The container gets an
                                 ISG1 part declaring it with that read mask, which rdna2spirv.py turns into the
                                 system VGPRs v2.. (one per component read, in order), as for the title's shaders
    ; native-variant: NAME       also pack the same code as NAME (its tail name differs, so its FSSM hash does) and
                                 translate NAME without --wave64-emulate even when the others get it: a wave64
                                 compute shader's native translation beside its exact one, to measure the difference

With --wave64-emulate (or LEONIDA_W64_EMULATE=1, as for runtime/test/shaders/build.py) the shaders are translated with
rdna2spirv.py --wave64-emulate, which makes wave64 compute shaders exact on 32-wide subgroups; native variants are
translated in a second rdna2spirv.py run into the same spirv/ folder, whose index.json it merges into.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import zlib

TOOLS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS)
import mc_command  # noqa: E402

VERSION = 0x0101
HEADER_SIZE = 0x200          # where the code section starts; leaves room for 25 PS input records from 0x138
CODE_SKIP = 0x28             # docs/design.md: code starts at u32[0xC0] + 0x28
WAVE32 = 0x20
INPUT_TABLE = 0x138
INPUT_KIND = 6
READ_METHODS = {"flat": 0x01, "interp": 0x02}
FP16 = 0x10
ENDPGM = 0xBF810000
CODE_END_FILLER = 0xBF9F0000
SENTINEL = 0xFFFFFFFF
STAGES = ("vs", "ps", "cs")
MAX_INPUTS = (HEADER_SIZE - INPUT_TABLE) // 8 - 1   # leaves a non-6 byte after the last record


class PackError(ValueError):
    """An .asm header or code that cannot make a well-formed FSSM blob."""


def header_fields(text, name):
    """The packing options from an .asm's leading comment lines (see the module docstring)."""
    fields = {"stage": name[:2].lower() if name[2:3] == "_" else None, "wave": 32, "inputs": [],
              "data_at": None, "data": [], "group": (64, 1, 1), "rsrc2": 0x8088, "native_variant": None,
              "position": 0}
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if not s.startswith(";"):
            break
        m = re.match(r";\s*(stage|wave|input|data-at|data|group|rsrc2|native-variant|position):\s*(.*?)\s*$", s)
        if not m:
            continue
        key, value = m.groups()
        try:
            if key == "stage":
                fields["stage"] = value.lower()
            elif key == "wave":
                fields["wave"] = int(value)
            elif key == "input":
                words = value.lower().split()
                if not words or words[0] not in READ_METHODS or set(words[1:]) - {"fp16"}:
                    raise PackError(f"{name}: input must be 'flat' or 'interp' [fp16], not {value!r}")
                fields["inputs"].append((READ_METHODS[words[0]], "fp16" in words[1:]))
            elif key == "data-at":
                fields["data_at"] = int(value, 0)
            elif key == "data":
                fields["data"] += [int(w, 0) for w in value.split()]
            elif key == "group":
                fields["group"] = tuple(int(v) for v in value.split(","))
            elif key == "rsrc2":
                fields["rsrc2"] = int(value, 0)
            elif key == "position":
                letters = value.lower()
                if not letters or set(letters) - set("xyzw"):
                    raise PackError(f"{name}: position must name components of xyzw, not {value!r}")
                fields["position"] = sum(1 << "xyzw".index(c) for c in set(letters))
            elif key == "native-variant":
                if not re.fullmatch(r"[A-Za-z0-9_]+", value) or value == name:
                    raise PackError(f"{name}: native-variant needs another plain name, not {value!r}")
                fields["native_variant"] = value
        except ValueError as e:
            if isinstance(e, PackError):
                raise
            raise PackError(f"{name}: bad '{key}' value {value!r}") from None
    if fields["stage"] not in STAGES:
        raise PackError(f"{name}: stage must be one of {STAGES} (a '; stage:' line or a vs_/ps_/cs_ name)")
    if fields["wave"] not in (32, 64):
        raise PackError(f"{name}: wave must be 32 or 64")
    if fields["position"] and fields["stage"] != "ps":
        raise PackError(f"{name}: position is for pixel shaders")
    if fields["inputs"] and fields["stage"] != "ps":
        raise PackError(f"{name}: input records are for pixel shaders")
    if len(fields["group"]) != 3:
        raise PackError(f"{name}: group needs X,Y,Z")
    return fields


def pack(code, *, stage, name, wave=32, inputs=(), data_at=None, data=(), group=(64, 1, 1), rsrc2=0x8088):
    """One FSSM blob: the header fields above, the code section and the binding-table tail."""
    if len(code) < 4 or len(code) % 4 or struct.unpack_from("<I", code, len(code) - 4)[0] != ENDPGM:
        raise PackError(f"{name}: the code must be whole dwords ending with s_endpgm")
    if len(inputs) > MAX_INPUTS:
        raise PackError(f"{name}: at most {MAX_INPUTS} input records fit before the code section")
    body = bytearray(code)
    if data:
        if data_at is None or data_at % 4 or data_at < len(code):
            raise PackError(f"{name}: data-at must be a dword offset at or after the code's end (0x{len(code):x})")
        if ENDPGM in data:
            raise PackError(f"{name}: a data word equal to s_endpgm would end the code for shader_disasm.py")
        body += struct.pack("<I", CODE_END_FILLER) * ((data_at - len(code)) // 4)
        body += b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in data)
    section_size = CODE_SKIP + len(body)
    code_end = HEADER_SIZE + section_size
    label = name.encode("ascii")
    tail = (struct.pack("<I", CODE_END_FILLER) * 4 + struct.pack("<II", 0, 0)
            + struct.pack("<IIHH", SENTINEL, 0, 0, len(label)) + label)
    tail += bytes(-len(tail) % 4)
    blob = bytearray(HEADER_SIZE)
    blob[0:4] = b"FSSM"
    struct.pack_into("<H", blob, 4, VERSION)
    struct.pack_into("<QQ", blob, 0xC0, HEADER_SIZE, section_size)
    struct.pack_into("<I", blob, 0xD4, WAVE32 if wave == 32 else 0)
    struct.pack_into("<Q", blob, 0xE8, code_end)
    struct.pack_into("<I", blob, 0x120, HEADER_SIZE)
    struct.pack_into("<I", blob, 0x12C, section_size)
    struct.pack_into("<I", blob, 0x130, code_end)
    if stage == "ps":
        for position, (method, fp16) in enumerate(inputs):
            record = bytes([INPUT_KIND, position, 0x0F, 0, 0, 0, position + 1, method | (FP16 if fp16 else 0)])
            blob[INPUT_TABLE + 8 * position:INPUT_TABLE + 8 * position + 8] = record
    elif stage == "cs":
        struct.pack_into("<4I", blob, INPUT_TABLE, rsrc2, *group)
    blob += bytes(CODE_SKIP) + body + tail
    struct.pack_into("<I", blob, 8, len(blob))
    return bytes(blob)


def isg1(position_mask):
    """An ISG1 part body declaring SV_Position (system value 1, float32, register 0, mask xyzw) with read mask
    position_mask: the element layout rdna2spirv.input_signature_full and shader_meta.read_dxbc_meta parse."""
    name = b"SV_Position\0"
    name += bytes(-len(name) % 4)
    element = struct.pack("<6I", 0, 8 + 32, 0, 1, 3, 0) + bytes((0xF, position_mask, 0, 0)) + struct.pack("<I", 0)
    return struct.pack("<II", 1, 8) + element + name


def container(fssm, position_mask=0):
    """The shader container the game ships: DXBC { DXIL { 16-byte program header stub, XDAT { 0x0D: XBCH +
    zlib(FSSM) } } }, as tools/fssm_unpack.py and the runtime's fssm_program() read it, then an ISG1 part when
    position_mask is not 0."""
    stream = zlib.compress(fssm, 9)
    xbch = b"XBCH" + struct.pack("<5I", 1, len(stream), 0, len(fssm), 0) + stream
    xdat_entry_at = 16 + 12 + 12
    dxil = bytes(16) + b"XDAT" + struct.pack("<II", 1, 12) + struct.pack("<III", 0x0D, xdat_entry_at, len(xbch)) + xbch
    parts = [(b"DXIL", dxil)] + ([(b"ISG1", isg1(position_mask))] if position_mask else [])
    at = 32 + 4 * len(parts)
    offsets, body = [], b""
    for tag, data in parts:
        offsets.append(at + len(body))
        body += tag + struct.pack("<I", len(data)) + data
    total = at + len(body)
    return b"DXBC" + bytes(16) + struct.pack(f"<III{len(parts)}I", 1, total, len(parts), *offsets) + body


ELF_SECTION = struct.Struct("<IIQQQQIIQQ")   # Elf64_Shdr: name, type, flags, addr, offset, size, link, info, ...


def text_section(obj, name):
    """The .text bytes of the ELF64 object llvm-mc wrote. A relocation against .text means a branch or reference the
    assembler could not resolve inside the program (a label that is not defined in it): refused."""
    if obj[:4] != b"\x7fELF" or len(obj) < 0x40 or obj[4] != 2 or obj[5] != 1:
        raise PackError(f"{name}.asm: llvm-mc did not write a little-endian ELF64 object")
    shoff, = struct.unpack_from("<Q", obj, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<3H", obj, 0x3A)
    if shentsize != ELF_SECTION.size or shoff + shnum * shentsize > len(obj) or shstrndx >= shnum:
        raise PackError(f"{name}.asm: llvm-mc's object has a section table this tool cannot read")
    sections = [ELF_SECTION.unpack_from(obj, shoff + i * shentsize) for i in range(shnum)]
    names_at = sections[shstrndx][4]
    text = None
    for section in sections:
        start = names_at + section[0]
        label = obj[start:obj.index(b"\0", start)].decode("ascii", "replace")
        if label in (".rel.text", ".rela.text"):
            raise PackError(f"{name}.asm: llvm-mc left relocations in .text (a branch to a label it does not define?)")
        if label == ".text":
            text = obj[section[4]:section[4] + section[5]]
    if text is None:
        raise PackError(f"{name}.asm: llvm-mc's object has no .text section")
    return text


def assemble(text, name, mcpu="gfx1013", wave=32):
    """The code bytes of the .asm text: llvm-mc assembles it into an ELF object (on stdout) and the code is its .text
    section, so branches to labels come out resolved, as the assembler lays them out. Wave64 code is assembled with
    +wavefrontsize64, so it can name VCC and 64-bit masks as such; the encodings are the same."""
    cmd = mc_command.mc_command() + ["-triple=amdgcn--amdpal", f"-mcpu={mcpu}", "-filetype=obj", "-o", "-"]
    if wave == 64:
        cmd.append("-mattr=+wavefrontsize64,-wavefrontsize32")
    r = mc_command.run(cmd, input=text.encode("utf-8"), capture_output=True)
    errors = r.stderr.decode("utf-8", "replace")
    if r.returncode or "error:" in errors:
        raise PackError(f"{name}.asm did not assemble:\n{errors}")
    return text_section(r.stdout, name)


def pack_file(path, out, mcpu="gfx1013"):
    """Assemble and pack one .asm (and its native variant, if it names one); writes <out>/<name>.dxbc for each.
    Returns [(name, stage, fssm blob, native)]."""
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    fields = header_fields(text, name)
    variant = fields.pop("native_variant")
    position = fields.pop("position")
    code = assemble(text, name, mcpu, fields["wave"])
    packed = []
    for packed_name, native in ((name, False),) + (((variant, True),) if variant else ()):
        blob = pack(code, name=packed_name, **fields)
        with open(os.path.join(out, packed_name + ".dxbc"), "wb") as f:
            f.write(container(blob, position))
        packed.append((packed_name, fields["stage"], blob, native))
    return packed


def run_tool(tool, *args):
    r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, tool), *map(str, args)],
                       capture_output=True, text=True)
    if r.returncode:
        raise PackError(f"{tool} exited with {r.returncode}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r.stdout


def translate(out, packed, mcpu="gfx1013", emulate=False):
    """Run fssm_unpack.py, shader_disasm.py and rdna2spirv.py on <out> and check that every packed shader made it
    through. Returns the rows of spirv/index.json for the packed shaders."""
    run_tool("fssm_unpack.py", out)
    disasm = os.path.join(out, "disasm")
    os.makedirs(disasm, exist_ok=True)
    run_tool("shader_disasm.py", os.path.join(out, "fssm"), "--out", disasm, "--mcpu", mcpu)
    for name, _, blob, _ in packed:
        with open(os.path.join(out, "fssm", name + ".fssm"), "rb") as f:
            if f.read() != blob:
                raise PackError(f"{name}: fssm_unpack.py did not give back the packed blob")
        with open(os.path.join(disasm, name + ".s"), encoding="utf-8") as f:
            first = f.readline()
        if "(header)" not in first or not first.rstrip().endswith(" 0 invalid words"):
            raise PackError(f"{name}: shader_disasm.py did not decode the code from the header: {first.strip()}")
    spirv = os.path.join(out, "spirv")
    natives = [name for name, _, _, native in packed if native] if emulate else []
    if natives:
        # A second listing folder beside disasm/ (rdna2spirv.py reads fssm/, the .dxbc and index.json from the parent).
        native_dir = os.path.join(out, "disasm_native")
        os.makedirs(native_dir, exist_ok=True)
        for name in natives:
            os.replace(os.path.join(disasm, name + ".s"), os.path.join(native_dir, name + ".s"))
    translated = run_tool("rdna2spirv.py", disasm, "--out", spirv, *(["--wave64-emulate"] if emulate else []))
    if natives:
        translated += run_tool("rdna2spirv.py", native_dir, "--out", spirv)
    with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    rows = {}
    for name, stage, blob, _ in packed:
        row = index.get(name)
        want = hashlib.sha256(blob).hexdigest()
        if not row or row.get("stage") != stage or row.get("fssm_sha256") != want:
            raise PackError(f"{name}: spirv/index.json has no {stage} entry with fssm_sha256 {want}; "
                            f"rdna2spirv.py said:\n{translated[-2000:]}")
        rows[name] = row
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pack RDNA2 assembly into synthetic FSSM programs in shader containers.")
    ap.add_argument("out", help="folder for <name>.dxbc and index.json (and fssm/, disasm/, spirv/ with --translate)")
    ap.add_argument("asm", nargs="+", help="hand-written .asm files")
    ap.add_argument("--translate", action="store_true", help="run fssm_unpack, shader_disasm and rdna2spirv after")
    ap.add_argument("--mcpu", default="gfx1013")
    ap.add_argument("--wave64-emulate", action="store_true",
                    help="translate with rdna2spirv.py --wave64-emulate (also LEONIDA_W64_EMULATE=1)")
    a = ap.parse_args(argv)
    emulate = a.wave64_emulate or os.environ.get("LEONIDA_W64_EMULATE") == "1"
    # As build.py: CMake names the spirv-as/spirv-val folder instead of passing a whole PATH.
    if os.environ.get("LEONIDA_SPIRV_TOOLS"):
        os.environ["PATH"] = os.environ["LEONIDA_SPIRV_TOOLS"] + os.pathsep + os.environ.get("PATH", "")
    os.makedirs(a.out, exist_ok=True)
    try:
        packed = [entry for p in a.asm for entry in pack_file(p, a.out, a.mcpu)]
        names = [entry[0] for entry in packed]
        if len(set(names)) != len(names):
            raise PackError("two shaders have the same name: " + ", ".join(sorted({n for n in names if names.count(n) > 1})))
        carve = [{"name": n, "file": n + ".dxbc", "stage": s, "size": os.path.getsize(os.path.join(a.out, n + ".dxbc"))}
                 for n, s, _, _ in packed]
        with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(carve, f, indent=1)
            f.write("\n")
        rows = translate(a.out, packed, a.mcpu, emulate) if a.translate else {}
    except PackError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for name, stage, blob, _ in packed:
        row = rows.get(name, {})
        wave = 32 if struct.unpack_from("<I", blob, 0xD4)[0] & WAVE32 else 64
        code = struct.unpack_from("<Q", blob, 0xC8)[0] - CODE_SKIP
        extra = f", translated, code base {'used' if row.get('uses_code_base') else 'unused'}" if row else ""
        if row and wave == 64:
            extra += f", wave64 exchange {row.get('wave64_exchange') or 'none'}, subgroup size {row.get('subgroup_size')}"
        print(f"{name}: {stage} wave{wave}, {len(blob)} bytes, section 0x{code:x} bytes, "
              f"fssm_sha256 {hashlib.sha256(blob).hexdigest()[:16]}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
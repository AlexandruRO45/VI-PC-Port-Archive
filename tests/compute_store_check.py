"""Check the single-store compute profile with original synthetic kernels."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from pixel_aot_check import envelope, must_reject
import compute_store_aot as aot
from pixel_aot import _read_named_container, disassemble
from resource_inventory import parts

SENTINEL = 0xc5c5c5c5
INDEXED = [
    "s_mov_b32 s12, s7", "s_mov_b32 s13, 0",
    "s_load_dwordx4 s[20:23], s[12:13], 0x30",
    "v_lshl_add_u32 v8, s11, 5, v0",
    "v_mov_b32_e32 v15, 19", "v_mov_b32_e32 v16, -5",
    "s_waitcnt lgkmcnt(0)", "buffer_store_dwordx2 v[15:16], v8, s[20:23], 0 idxen offset:4",
    "s_endpgm",
]
FIXED = [
    "s_mov_b32 s9, 0", "s_load_dwordx4 s[12:15], s[8:9], null",
    "v_mov_b32_e32 v20, 11", "v_mov_b32_e32 v21, 23", "v_mov_b32_e32 v22, -2",
    "s_waitcnt lgkmcnt(0)", "buffer_store_dwordx3 v[20:22], off, s[12:15], 0 offset:20",
    "s_endpgm",
]
MULTI_FIXED = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "s_mov_b32 s10, 0x12345678",
    "s_movk_i32 s11, 0xfffc",
    "v_mov_b32_e32 v4, s10",
    "v_mov_b32_e32 v5, s11",
    "v_mov_b32_e32 v6, v4",
    "v_mov_b32_e32 v7, 7",
    "s_waitcnt lgkmcnt(0)",
    "s_clause 0x1",
    "buffer_store_dwordx2 v[4:5], off, s[12:15], 0 offset:8",
    "buffer_store_dwordx2 v[6:7], off, s[12:15], 0 offset:32",
    "s_endpgm",
]
MULTI_INDEXED = [
    "s_mov_b32 s12, s7",
    "s_mov_b32 s13, 0",
    "s_load_dwordx4 s[20:23], s[12:13], 0x30",
    "s_mov_b32 s14, 0x00abcdef",
    "s_movk_i32 s15, 0xfff0",
    "v_lshl_add_u32 v8, s11, 5, v0",
    "v_mov_b32_e32 v10, s14",
    "v_mov_b32_e32 v11, s15",
    "s_waitcnt lgkmcnt(0) vmcnt(0)",
    "buffer_store_dword v10, v8, s[20:23], 0 idxen",
    "buffer_store_dword v11, v8, s[20:23], 0 idxen offset:8",
    "s_endpgm",
]
INDEX_BINDING = dict(table_reg=7, group_reg=11, descriptor_offset=48, stride=12)
FIXED_BINDING = dict(table_reg=8, descriptor_offset=0)
CASES = [
    ("indexed", INDEXED, 32, INDEX_BINDING),
    ("fixed", FIXED, 1, FIXED_BINDING),
    ("multi_fixed", MULTI_FIXED, 1, FIXED_BINDING),
    ("multi_indexed", MULTI_INDEXED, 32, INDEX_BINDING),
]


def controls():
    for change in (dict(table_reg=6), dict(group_reg=10), dict(descriptor_offset=32), dict(stride=8)):
        must_reject(lambda: aot.lower(INDEXED, 32, **(INDEX_BINDING | change)))
    for old, new in (("lgkmcnt(0)", "lgkmcnt(1)"), ("s11, 5", "s11, 4"),
                     ("s[20:23], 0 idxen", "s[24:27], 0 idxen"), ("s13, 0", "s13, 1")):
        must_reject(lambda: aot.lower([s.replace(old, new) for s in INDEXED], 32, **INDEX_BINDING))
    must_reject(lambda: aot.lower(INDEXED[:6] + INDEXED[7:], 32, **INDEX_BINDING))
    must_reject(lambda: aot.lower(INDEXED[:7] + ["s_mov_b32 s21, 0"] + INDEXED[7:], 32, **INDEX_BINDING))
    must_reject(lambda: aot.lower(FIXED, 32, **FIXED_BINDING))
    must_reject(lambda: aot.lower(["s_endpgm"], 1, **FIXED_BINDING))
    # Overlapping store spans in one invocation are rejected.
    must_reject(lambda: aot.lower(
        MULTI_FIXED[:-1] + ["buffer_store_dword v7, off, s[12:15], 0 offset:12", "s_endpgm"],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        MULTI_INDEXED[:-1] + ["buffer_store_dwordx2 v[10:11], v8, s[20:23], 0 idxen offset:4", "s_endpgm"],
        32, **INDEX_BINDING))
    # Non-constant or pending scalar/vector sources and out-of-range hints/immediates are rejected.
    must_reject(lambda: aot.lower(
        [s.replace("v_mov_b32_e32 v4, s10", "v_mov_b32_e32 v4, s8") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        [s.replace("v_mov_b32_e32 v4, s10", "v_mov_b32_e32 v4, s12") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        [s.replace("v_mov_b32_e32 v6, v4", "v_mov_b32_e32 v6, v0") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        [s.replace("s_clause 0x1", "s_clause 0x40") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        [s.replace("s_movk_i32 s11, 0xfffc", "s_movk_i32 s11, 0x10000") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(aot.group_width, bytes(52))


def binary(directory, assembler, objcopy, library):
    for _, lines, width, binding in CASES:
        src, obj, raw = [directory / x for x in ("test.s", "test.o", "test.bin")]
        src.write_text("\n".join(lines) + "\n", encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", src, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
        code = raw.read_bytes()
        padded = code + bytes.fromhex("00009fbf") * (4 - (len(code) // 4 + 10) % 4)
        payload = envelope(padded, directory_end=(40 + len(padded), 288 + len(padded)), compute_width=width)
        blob = _read_named_container(payload, 1 + len(b"fixture") + 1)
        sections = parts(blob)
        assert aot.group_width(sections[b"PSV0"]) == width
        assert aot.container_code(sections[b"DXIL"], stage=5) == code
        must_reject(lambda: aot.container_code(sections[b"DXIL"], stage=0))
        assert aot.lower(disassemble(code, library), width, **binding) == aot.lower(lines, width, **binding)
        container, output = directory / "input.dxbc", directory / "cli.hlsl"
        container.write_bytes(blob)
        options = [item for key, value in binding.items() for item in ("--" + key.replace("_", "-"), str(value))]
        subprocess.run([sys.executable, "-B", str(Path(aot.__file__)), container, "--output", output,
                        "--llvm", library, *options], check=True)
        assert output.read_text() == aot.lower(lines, width, **binding)


def draw(directory, compiler, renderer, backend):
    for name, lines, width, binding in CASES:
        src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
        src.write_text(aot.lower(lines, width, **binding), encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
        cmd = [renderer, backend, "--compute-store-case", cso, case]
        for groups in ((1, 3) if width == 32 else (1,)):
            words = [SENTINEL] * 320
            if name == "fixed":
                words[5:8] = [11, 23, 0xfffffffe]
                guard_word, stored_word = 0, 5
            elif name == "indexed":
                for i in range(groups * 32):
                    words[3*i+1:3*i+3] = [19, 0xfffffffb]
                guard_word, stored_word = 0, 1
            elif name == "multi_fixed":
                words[2:4] = [0x12345678, 0xfffffffc]
                words[8:10] = [0x12345678, 7]
                guard_word, stored_word = 4, 9
            else:
                for i in range(groups * 32):
                    words[3*i] = 0x00abcdef
                    words[3*i+2] = 0xfffffff0
                guard_word, stored_word = 1, 2
            case.write_bytes(struct.pack("<321I", groups, *words))
            result = subprocess.run(cmd)
            if result.returncode:
                return result.returncode
        words[guard_word] ^= 1
        case.write_bytes(struct.pack("<321I", groups, *words))
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={guard_word} expected=" not in bad.stderr:
            raise AssertionError("A wrong untouched word escaped detection")
        words[guard_word] ^= 1
        words[stored_word] ^= 1
        case.write_bytes(struct.pack("<321I", groups, *words))
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError("A wrong stored word escaped detection")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit({"binary": binary, "draw": draw}[mode](Path(tmp), *args))

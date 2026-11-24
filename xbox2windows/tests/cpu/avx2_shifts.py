"""Compare native and translated AVX2 broadcasts and variable per-lane vector shifts."""
import ctypes
from pathlib import Path
import struct
import sys

import pefile
from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64
BCAST_EXPORTS = (
    "bcast_b256", "bcast_w256", "bcast_d256", "bcast_q256",
    "bcast_b256_reg", "bcast_w256_reg", "bcast_d256_reg", "bcast_q256_reg",
    "bcast_d256_alias",
    "bcast_b128", "bcast_w128", "bcast_d128", "bcast_q128",
    "bcast_b128_reg", "bcast_w128_reg", "bcast_d128_reg", "bcast_q128_reg",
    "bcast_b128_alias",
)
SHIFT_EXPORTS = (
    "sllvd256", "srlvd256", "sravd256", "sllvq256", "srlvq256",
    "sllvd256_mem", "srlvd256_mem", "sravd256_mem", "sllvq256_mem", "srlvq256_mem",
    "sllvd128", "srlvd128", "sravd128", "sllvq128", "srlvq128",
    "sllvd128_mem", "srlvd128_mem", "sravd128_mem", "sllvq128_mem", "srlvq128_mem",
    "sllvd256_alias_src2", "srlvd256_alias_src1", "sravd256_alias_all", "sllvd256_alias_all",
    "sllvd128_alias_src2", "sllvq256_alias_src2",
)


def emit(image, output):
    parts = [
        "#include <intrin.h>\n",
        INTEGER_HEADER,
        'extern "C" __declspec(dllexport) int host_has_avx2() {\n'
        "    int regs[4]{};\n"
        "    __cpuid(regs, 0);\n"
        "    if (regs[0] < 7) return 0;\n"
        "    __cpuid(regs, 1);\n"
        "    if ((regs[2] & (1 << 27)) == 0) return 0;\n"
        "    if ((_xgetbv(0) & 6ull) != 6ull) return 0;\n"
        "    __cpuidex(regs, 7, 0);\n"
        "    return (regs[1] & (1 << 5)) != 0;\n"
        "}\n",
    ]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in BCAST_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = src; rdx = dst;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word src, word dst) {{\n{code}\n}}\n')
        for name in SHIFT_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = vals; rdx = counts; r8 = dst;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word vals, word counts, word dst) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in BCAST_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p], U64
    for name in SHIFT_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], U64
    return lib


def check(native_path, lifted_path):
    lifted = load_lib(lifted_path)
    lifted.host_has_avx2.argtypes, lifted.host_has_avx2.restype = [], ctypes.c_int
    if not lifted.host_has_avx2():
        print("Host CPU/OS lacks AVX2 (CPUID leaf 7 EBX bit 5 / XCR0 YMM); skipping native differential check.")
        sys.exit(77)
    native = load_lib(native_path)
    cases = 0

    src_buf = (ctypes.c_uint8 * 32)(*range(0x10, 0x30))
    dst_buf = (ctypes.c_uint8 * 32)()

    bcast_specs = (
        ("bcast_b256", 1, 32), ("bcast_w256", 2, 32), ("bcast_d256", 4, 32), ("bcast_q256", 8, 32),
        ("bcast_b256_reg", 1, 32), ("bcast_w256_reg", 2, 32), ("bcast_d256_reg", 4, 32), ("bcast_q256_reg", 8, 32),
        ("bcast_d256_alias", 4, 32),
        ("bcast_b128", 1, 16), ("bcast_w128", 2, 16), ("bcast_d128", 4, 16), ("bcast_q128", 8, 16),
        ("bcast_b128_reg", 1, 16), ("bcast_w128_reg", 2, 16), ("bcast_d128_reg", 4, 16), ("bcast_q128_reg", 8, 16),
        ("bcast_b128_alias", 1, 16),
    )
    for name, elem_size, vec_size in bcast_specs:
        want = bytes(src_buf[:elem_size]) * (vec_size // elem_size) + b"\x00" * (32 - vec_size)
        for lib, label in ((native, "native"), (lifted, "lifted")):
            ctypes.memset(dst_buf, 0xff, 32)
            getattr(lib, name)(src_buf, dst_buf)
            got = bytes(dst_buf)
            if got != want:
                raise AssertionError(f"{label} {name} = {got.hex()}, expected {want.hex()}")
        cases += 1

    sll32 = lambda v, c: 0 if c >= 32 else (v << c) & 0xffffffff
    srl32 = lambda v, c: 0 if c >= 32 else v >> c
    sra32 = lambda v, c: ((v - 0x100000000 if v & 0x80000000 else v) >> min(c, 31)) & 0xffffffff
    sll64 = lambda v, c: 0 if c >= 64 else (v << c) & 0xffffffffffffffff
    srl64 = lambda v, c: 0 if c >= 64 else v >> c

    v32_c32_pairs = (
        ((0x00000000, 0x00000001, 0x7fffffff, 0x80000000, 0xffffffff, 0x12345678, 0x87654321, 0x55aa55aa),
         (0, 1, 7, 15, 31, 32, 33, 0xffffffff)),
        ((0x80000000, 0x87654321, 0xffffffff, 0x7fffffff, 0x80000001, 0x12345678, 0xffffffff, 0x55aa55aa),
         (31, 1, 32, 0xffffffff, 0, 31, 33, 16)),
        ((0xffffffff, 0x80000001, 0x12345678, 0x55aa55aa, 0x7fffffff, 0x80000000, 0x00000001, 0x87654321),
         (0, 33, 31, 32, 1, 0, 31, 32)),
    )
    shift32_specs = (
        ("sllvd256", sll32, 8), ("srlvd256", srl32, 8), ("sravd256", sra32, 8),
        ("sllvd256_mem", sll32, 8), ("srlvd256_mem", srl32, 8), ("sravd256_mem", sra32, 8),
        ("sllvd256_alias_src2", sll32, 8), ("srlvd256_alias_src1", srl32, 8),
        ("sllvd128", sll32, 4), ("srlvd128", srl32, 4), ("sravd128", sra32, 4),
        ("sllvd128_mem", sll32, 4), ("srlvd128_mem", srl32, 4), ("sravd128_mem", sra32, 4),
        ("sllvd128_alias_src2", sll32, 4),
    )
    for vals32, cnts32 in v32_c32_pairs:
        c32_raw = struct.pack("<8I", *cnts32)
        assert any(c32_raw[16:]), "upper 128 bits of counts must be non-zero for XMM zeroing check"
        v32_buf = ctypes.create_string_buffer(struct.pack("<8I", *vals32), 32)
        c32_buf = ctypes.create_string_buffer(c32_raw, 32)
        for name, fn, lanes in shift32_specs:
            want = struct.pack(f"<{lanes}I", *(fn(v, c) for v, c in zip(vals32[:lanes], cnts32[:lanes])))
            want += b"\x00" * (32 - 4 * lanes)
            for lib, label in ((native, "native"), (lifted, "lifted")):
                ctypes.memset(dst_buf, 0xff, 32)
                getattr(lib, name)(v32_buf, c32_buf, dst_buf)
                got = bytes(dst_buf)
                if got != want:
                    raise AssertionError(f"{label} {name} = {got.hex()}, expected {want.hex()}")
            cases += 1

    alias_all_vals = (0, 1, 2, 15, 31, 32, 0x80000005, 0xffffffff)
    alias_all_buf = ctypes.create_string_buffer(struct.pack("<8I", *alias_all_vals), 32)
    dummy_buf = ctypes.create_string_buffer(32)
    for name, fn in (("sravd256_alias_all", sra32), ("sllvd256_alias_all", sll32)):
        want = struct.pack("<8I", *(fn(v, v) for v in alias_all_vals))
        for lib, label in ((native, "native"), (lifted, "lifted")):
            ctypes.memset(dst_buf, 0xaa, 32)
            getattr(lib, name)(alias_all_buf, dummy_buf, dst_buf)
            got = bytes(dst_buf)
            if got != want:
                raise AssertionError(f"{label} {name} = {got.hex()}, expected {want.hex()}")
        cases += 1

    v64_c64_pairs = (
        ((0x0123456789abcdef, 0x8000000000000000, 0xffffffffffffffff, 0x0000000000000001), (0, 1, 63, 64)),
        ((0x0123456789abcdef, 0x8000000000000000, 0xffffffffffffffff, 0x0000000000000001), (7, 32, 65, 0xffffffffffffffff)),
        ((0x8000000000000000, 0xffffffffffffffff, 0x0123456789abcdef, 0x0000000000000001), (63, 64, 0, 1)),
        ((0x8000000000000000, 0xffffffffffffffff, 0x0123456789abcdef, 0x0000000000000001), (65, 0xffffffffffffffff, 7, 32)),
    )
    shift64_specs = (
        ("sllvq256", sll64, 4), ("srlvq256", srl64, 4),
        ("sllvq256_mem", sll64, 4), ("srlvq256_mem", srl64, 4),
        ("sllvq256_alias_src2", sll64, 4),
        ("sllvq128", sll64, 2), ("srlvq128", srl64, 2),
        ("sllvq128_mem", sll64, 2), ("srlvq128_mem", srl64, 2),
    )
    for vals64, cnts64 in v64_c64_pairs:
        v64_buf = ctypes.create_string_buffer(struct.pack("<4Q", *vals64), 32)
        c64_buf = ctypes.create_string_buffer(struct.pack("<4Q", *cnts64), 32)
        for name, fn, lanes in shift64_specs:
            want = struct.pack(f"<{lanes}Q", *(fn(v, c) for v, c in zip(vals64[:lanes], cnts64[:lanes])))
            want += b"\x00" * (32 - 8 * lanes)
            for lib, label in ((native, "native"), (lifted, "lifted")):
                ctypes.memset(dst_buf, 0xff, 32)
                getattr(lib, name)(v64_buf, c64_buf, dst_buf)
                got = bytes(dst_buf)
                if got != want:
                    raise AssertionError(f"{label} {name} = {got.hex()}, expected {want.hex()}")
            cases += 1

    print(f"{cases} AVX2 broadcast and variable-shift vector cases matched native and translated DLLs")


def check_rejections():
    # EVEX scalar-broadcast vpsllvd ymm0, ymm1, dword ptr [rdx]{1to8} (62 f2 75 38 47 02) must be rejected.
    evex_bcast = list(Decoder(64, bytes.fromhex("62 f2 75 38 47 02 c3"), ip=0x1000))
    assert evex_bcast[0].is_broadcast
    controls = (
        ("EVEX broadcast vpsllvd", evex_bcast, "unsupported query operation"),
        ("EVEX non-broadcast vpsllvd ymm0, ymm1, ymm2", "62 f2 75 28 47 c2 c3", "unsupported query operation"),
        ("EVEX GPR-source vpbroadcastd ymm0, eax", "62 f2 7d 28 7c c0 c3", "unsupported query operation"),
        ("EVEX 512-bit vpsrlvq zmm0, zmm1, zmm2", "62 f2 f5 48 45 c2 c3", "unsupported query operation"),
        ("untracked op0/op1 vpbroadcastb xmm3, xmm3", "c4 e2 79 78 db c3", "unsupported query operation"),
        ("untracked op0 vpbroadcastb xmm3, xmm0", "c4 e2 79 78 d8 c3", "unsupported query operation"),
        ("untracked op0 mem vpbroadcastq ymm3, qword ptr [rcx]", "c4 e2 7d 59 19 c3", "unsupported query operation"),
        ("untracked op1 vpbroadcastd ymm0, xmm3", "c4 e2 7d 58 c3 c3", "unsupported query operation"),
        ("untracked op0 vpsllvd ymm3, ymm1, ymm2", "c4 e2 75 47 da c3", "unsupported query operation"),
        ("untracked op1 vpsrlvd ymm0, ymm3, ymm2", "c4 e2 65 45 c2 c3", "unsupported query operation"),
        ("untracked op2 vpsravd xmm0, xmm1, xmm3", "c4 e2 71 46 c3 c3", "unsupported query operation"),
        ("untracked op2 vpsllvq ymm0, ymm1, ymm3", "c4 e2 f5 47 c3 c3", "unsupported query operation"),
        ("FS-segment prefix vpbroadcastd ymm0, dword ptr fs:[rcx]", "64 c4 e2 7d 58 01 c3", "unsupported instruction prefix"),
        ("EIP-relative vpsllvd ymm0, ymm1, ymmword ptr [eip]", "67 c4 e2 75 47 05 00 00 00 00 c3", "unsupported EIP-relative operand"),
    )
    for label, candidate, diagnostic in controls:
        insns = list(Decoder(64, bytes.fromhex(candidate), ip=0x1000)) if isinstance(candidate, str) else candidate
        try:
            lower(insns, [], {}, {}, {})
        except ValueError as exc:
            if not str(exc).startswith(f"{diagnostic} at 0x1000"):
                raise AssertionError(f"{label}: unexpected diagnostic {exc}") from exc
        else:
            raise AssertionError(f"accepted unsupported instruction ({label}): {insns[0]}")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check_rejections()
        check(native, other)
    else:
        raise ValueError(mode)

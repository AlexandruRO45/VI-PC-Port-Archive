"""Compare native and translated SSE packed integer instructions."""
import ctypes
from pathlib import Path
import struct
import sys

import pefile
from iced_x86 import Decoder, DecoderOptions

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64

VECTOR2_EXPORTS = (
    "test_pavgb",
    "test_pavgw",
    "test_pmaxsw",
    "test_pmaxub",
    "test_pminsw",
    "test_pminub",
    "test_pmulhuw",
    "test_psadbw",
)
MASK_EXPORTS = ("test_pmovmskb",)
EXTRACT_EXPORTS = tuple(f"test_pextrw_{i}" for i in range(8))
INSERT_EXPORTS = tuple(f"test_pinsrw_{i}" for i in range(8))


def emit(image, output):
    parts = [
        "#include <intrin.h>\n",
        INTEGER_HEADER,
    ]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in VECTOR2_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = b; r8 = out; r9 = flags_out;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word out, word flags_out) {{\n{code}\n}}\n')

        for name in MASK_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = flags_out;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word flags_out) {{\n{code}\n}}\n')

        for name in EXTRACT_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = flags_out;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word flags_out) {{\n{code}\n}}\n')

        for name in INSERT_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = val; r8 = out; r9 = flags_out;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word val, word out, word flags_out) {{\n{code}\n}}\n')

    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in VECTOR2_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes = [U64, U64, U64, ctypes.POINTER(U64)]
        fn.restype = U64
    for name in MASK_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes = [U64, ctypes.POINTER(U64)]
        fn.restype = U64
    for name in EXTRACT_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes = [U64, ctypes.POINTER(U64)]
        fn.restype = U64
    for name in INSERT_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes = [U64, U64, U64, ctypes.POINTER(U64)]
        fn.restype = U64
    return lib


def ref_pavgb(a: bytes, b: bytes) -> bytes:
    return bytes(((a[k] + b[k] + 1) >> 1) for k in range(16))


def ref_pavgw(a: bytes, b: bytes) -> bytes:
    wa = struct.unpack("<8H", a)
    wb = struct.unpack("<8H", b)
    res = [((wa[k] + wb[k] + 1) >> 1) for k in range(8)]
    return struct.pack("<8H", *res)


def ref_pmaxsw(a: bytes, b: bytes) -> bytes:
    wa = struct.unpack("<8h", a)
    wb = struct.unpack("<8h", b)
    res = [max(wa[k], wb[k]) for k in range(8)]
    return struct.pack("<8h", *res)


def ref_pmaxub(a: bytes, b: bytes) -> bytes:
    return bytes(max(a[k], b[k]) for k in range(16))


def ref_pminsw(a: bytes, b: bytes) -> bytes:
    wa = struct.unpack("<8h", a)
    wb = struct.unpack("<8h", b)
    res = [min(wa[k], wb[k]) for k in range(8)]
    return struct.pack("<8h", *res)


def ref_pminub(a: bytes, b: bytes) -> bytes:
    return bytes(min(a[k], b[k]) for k in range(16))


def ref_pmulhuw(a: bytes, b: bytes) -> bytes:
    wa = struct.unpack("<8H", a)
    wb = struct.unpack("<8H", b)
    res = [((wa[k] * wb[k]) >> 16) & 0xffff for k in range(8)]
    return struct.pack("<8H", *res)


def ref_psadbw(a: bytes, b: bytes) -> bytes:
    s0 = sum(abs(a[k] - b[k]) for k in range(8))
    s1 = sum(abs(a[k] - b[k]) for k in range(8, 16))
    return struct.pack("<8H", s0, 0, 0, 0, s1, 0, 0, 0)


def ref_pmovmskb(a: bytes) -> int:
    mask = 0
    for k in range(16):
        if (a[k] & 0x80) != 0:
            mask |= 1 << k
    return mask


def ref_pextrw(a: bytes, lane: int) -> int:
    wa = struct.unpack("<8H", a)
    return wa[lane & 7]


def ref_pinsrw(a: bytes, val: int, lane: int) -> bytes:
    wa = list(struct.unpack("<8H", a))
    wa[lane & 7] = val & 0xffff
    return struct.pack("<8H", *wa)


def check(native_path, lifted_path):
    native = load_lib(native_path)
    lifted = load_lib(lifted_path)

    test_vectors = [
        bytes([0] * 16),
        bytes([0xff] * 16),
        bytes([0x80] * 16),
        bytes([0x7f] * 16),
        bytes([i * 17 % 256 for i in range(16)]),
        bytes([(255 - i * 13) % 256 for i in range(16)]),
        bytes([0x01, 0x80, 0xfe, 0x7f, 0x55, 0xaa, 0x33, 0xcc,
               0x0f, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc]),
        bytes([0x5a] * 16),
        bytes([0xa5] * 16),
    ]

    out_buf = ctypes.create_string_buffer(16)
    flag_slot = ctypes.c_uint64(0)

    vec2_fns = {
        "test_pavgb": ref_pavgb,
        "test_pavgw": ref_pavgw,
        "test_pmaxsw": ref_pmaxsw,
        "test_pmaxub": ref_pmaxub,
        "test_pminsw": ref_pminsw,
        "test_pminub": ref_pminub,
        "test_pmulhuw": ref_pmulhuw,
        "test_psadbw": ref_psadbw,
    }

    vec2_count = 0
    for name, ref_fn in vec2_fns.items():
        for va in test_vectors:
            for vb in test_vectors:
                expected_out = ref_fn(va, vb)
                ptr_a = ctypes.cast(ctypes.c_char_p(va), ctypes.c_void_p).value
                ptr_b = ctypes.cast(ctypes.c_char_p(vb), ctypes.c_void_p).value
                ptr_out = ctypes.cast(out_buf, ctypes.c_void_p).value

                for lib, label in ((native, "native"), (lifted, "lifted")):
                    ctypes.memset(out_buf, 0xcc, 16)
                    flag_slot.value = 0xdeadbeef
                    getattr(lib, name)(ptr_a, ptr_b, ptr_out, ctypes.byref(flag_slot))
                    got = out_buf.raw
                    if got != expected_out:
                        raise AssertionError(f"{label} {name} mismatch: got {got.hex()}, expected {expected_out.hex()}")
                    if flag_slot.value != 2:
                        raise AssertionError(f"{label} {name} flags modified: got {flag_slot.value}, expected 2 (CF=1, ZF=0)")
                vec2_count += 1

    mask_count = 0
    for va in test_vectors:
        expected_mask = ref_pmovmskb(va)
        ptr_a = ctypes.cast(ctypes.c_char_p(va), ctypes.c_void_p).value
        for lib, label in ((native, "native"), (lifted, "lifted")):
            flag_slot.value = 0xdeadbeef
            got_mask = getattr(lib, "test_pmovmskb")(ptr_a, ctypes.byref(flag_slot))
            if got_mask != expected_mask:
                raise AssertionError(f"{label} pmovmskb mismatch: got {got_mask:#x}, expected {expected_mask:#x}")
            if flag_slot.value != 2:
                raise AssertionError(f"{label} pmovmskb flags modified: got {flag_slot.value}, expected 2")
        mask_count += 1

    pextrw_count = 0
    for lane in range(8):
        fn_name = f"test_pextrw_{lane}"
        for va in test_vectors:
            expected_val = ref_pextrw(va, lane)
            ptr_a = ctypes.cast(ctypes.c_char_p(va), ctypes.c_void_p).value
            for lib, label in ((native, "native"), (lifted, "lifted")):
                flag_slot.value = 0xdeadbeef
                got_val = getattr(lib, fn_name)(ptr_a, ctypes.byref(flag_slot))
                if got_val != expected_val:
                    raise AssertionError(f"{label} {fn_name} mismatch: got {got_val:#x}, expected {expected_val:#x}")
                if flag_slot.value != 2:
                    raise AssertionError(f"{label} {fn_name} flags modified: got {flag_slot.value}, expected 2")
            pextrw_count += 1

    pinsrw_count = 0
    insert_values = [0x0000, 0x1234, 0x8000, 0xffff, 0x7fff, 0xbeef]
    for lane in range(8):
        fn_name = f"test_pinsrw_{lane}"
        for va in test_vectors[:4]:
            for val in insert_values:
                expected_out = ref_pinsrw(va, val, lane)
                ptr_a = ctypes.cast(ctypes.c_char_p(va), ctypes.c_void_p).value
                ptr_out = ctypes.cast(out_buf, ctypes.c_void_p).value
                for lib, label in ((native, "native"), (lifted, "lifted")):
                    ctypes.memset(out_buf, 0xcc, 16)
                    flag_slot.value = 0xdeadbeef
                    getattr(lib, fn_name)(ptr_a, val, ptr_out, ctypes.byref(flag_slot))
                    got = out_buf.raw
                    if got != expected_out:
                        raise AssertionError(f"{label} {fn_name} mismatch: got {got.hex()}, expected {expected_out.hex()}")
                    if flag_slot.value != 2:
                        raise AssertionError(f"{label} {fn_name} flags modified: got {flag_slot.value}, expected 2")
                pinsrw_count += 1

    print(f"SSE packed integer test passed: {vec2_count} vec2, {mask_count} mask, {pextrw_count} pextrw, {pinsrw_count} pinsrw cases verified.")


def check_rejections():
    # 1. MMX registers must be rejected
    # paddb mm0, mm1 -> 0f fc c1
    try:
        lower(list(Decoder(64, bytes.fromhex("0f fc c1 c3"), ip=0x1000)), [], {}, {}, {})
    except ValueError as exc:
        if "MMX" not in str(exc):
            raise AssertionError(f"unexpected diagnostic for MMX: {exc}") from exc
    else:
        raise AssertionError("accepted MMX instruction")

    # 2. XMM registers beyond XMM2 must be rejected
    # pavgb xmm3, xmm0 -> 66 0f e0 d8
    try:
        lower(list(Decoder(64, bytes.fromhex("66 0f e0 d8 c3"), ip=0x1000)), [], {}, {}, {})
    except ValueError as exc:
        if "XMM" not in str(exc):
            raise AssertionError(f"unexpected diagnostic for XMM3: {exc}") from exc
    else:
        raise AssertionError("accepted XMM3 instruction")

    # 3. Lock prefix must be rejected
    try:
        lower(list(Decoder(64, bytes.fromhex("f0 01 08 c3"), ip=0x1000)), [], {}, {}, {})
    except ValueError as exc:
        if "prefix" not in str(exc):
            raise AssertionError(f"unexpected diagnostic for lock prefix: {exc}") from exc
    else:
        raise AssertionError("accepted lock prefix on pavgb")

    print("PASS: cpu sse packed rejections")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        check_rejections()
        sys.exit(0)
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(native, other)
        check_rejections()
    else:
        raise ValueError(mode)

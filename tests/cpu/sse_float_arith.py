"""Compare native and translated 128-bit baseline SSE packed single-precision arithmetic instructions."""
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
EXPORTS = (
    "addps_op", "subps_op", "mulps_op", "divps_op",
    "addps_self_op", "subps_self_op", "mulps_self_op", "divps_self_op",
    "addps_xmm1_dst_op", "addps_xmm2_dst_op",
)
BINARY_EXPORTS = ("addps_op", "subps_op", "mulps_op", "divps_op", "addps_xmm1_dst_op", "addps_xmm2_dst_op")
SELF_EXPORTS = ("addps_self_op", "subps_self_op", "mulps_self_op", "divps_self_op")


def emit(image, output):
    parts = [
        INTEGER_HEADER,
        'extern "C" __declspec(dllexport) uint32_t get_mxcsr() { return _mm_getcsr(); }\n',
        'extern "C" __declspec(dllexport) void set_mxcsr(uint32_t v) { _mm_setcsr(v); }\n',
    ]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word dst) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], U64
    return lib


def f2u(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def check(native_path, lifted_path):
    native, lifted = load_lib(native_path), load_lib(lifted_path)
    lifted.get_mxcsr.argtypes, lifted.get_mxcsr.restype = [], ctypes.c_uint32
    lifted.set_mxcsr.argtypes, lifted.set_mxcsr.restype = [ctypes.c_uint32], None

    orig_mxcsr = lifted.get_mxcsr()

    # Normal floats, zeros, infinities, denormals
    POS_ZERO = 0x00000000
    NEG_ZERO = 0x80000000
    POS_ONE = f2u(1.0)
    NEG_ONE = f2u(-1.0)
    POS_TWO = f2u(2.0)
    NEG_TWO = f2u(-2.0)
    POS_HALF = f2u(0.5)
    NEG_HALF = f2u(-0.5)
    POS_PI = f2u(3.1415927)
    POS_E = f2u(2.7182818)
    POS_LARGE = f2u(1e20)
    NEG_LARGE = f2u(-1e20)
    POS_SMALL = f2u(1e-20)
    NEG_SMALL = f2u(-1e-20)
    POS_DENORM_MIN = 0x00000001
    POS_DENORM_MAX = 0x007fffff
    NEG_DENORM_MIN = 0x80000001
    NEG_DENORM_MAX = 0x807fffff
    POS_INF = 0x7f800000
    NEG_INF = 0xff800000

    # Distinct quiet NaNs (qNaN) and signaling NaNs (sNaN) with different signs and payloads.
    # Payload 1 = 0x012345, Payload 2 = 0x054321
    POS_QNAN1 = 0x7fc12345
    NEG_QNAN1 = 0xffc12345
    POS_QNAN2 = 0x7fc54321
    NEG_QNAN2 = 0xffc54321
    POS_SNAN1 = 0x7f812345
    NEG_SNAN1 = 0xff812345
    POS_SNAN2 = 0x7f854321
    NEG_SNAN2 = 0xff854321

    # Edge payloads
    POS_QNAN_MIN = 0x7fc00001
    NEG_QNAN_MAX = 0xffffffff
    POS_SNAN_MIN = 0x7f800001
    NEG_SNAN_MAX = 0xffbfffff

    test_vectors = [
        # 1. Normals and signs
        ((POS_ONE, NEG_TWO, POS_HALF, POS_PI), (POS_TWO, POS_HALF, NEG_ONE, POS_E)),
        ((f2u(10.0), f2u(20.0), f2u(30.0), f2u(40.0)), (f2u(2.0), f2u(4.0), f2u(5.0), f2u(8.0))),
        ((f2u(12345.678), f2u(-9876.543), f2u(0.001), f2u(-0.002)), (f2u(87654.32), f2u(1234.567), f2u(0.999), f2u(1.002))),
        ((f2u(1.0 + 2**-23), f2u(1.0 - 2**-23), f2u(1.5), f2u(2.5)), (f2u(2.0**-23), f2u(-2.0**-23), f2u(0.25), f2u(0.75))),

        # 2. Zeros, infinities, and overflow/underflow
        ((POS_ZERO, NEG_ZERO, POS_INF, NEG_INF), (POS_INF, NEG_INF, POS_ZERO, NEG_ZERO)),
        ((POS_LARGE, NEG_LARGE, POS_SMALL, NEG_SMALL), (POS_SMALL, NEG_SMALL, POS_LARGE, NEG_LARGE)),
        ((POS_DENORM_MIN, POS_DENORM_MAX, NEG_DENORM_MIN, NEG_DENORM_MAX), (POS_DENORM_MAX, POS_DENORM_MIN, NEG_DENORM_MAX, NEG_DENORM_MIN)),

        # 3. Operations generating NaNs or infinities from non-NaNs
        ((POS_ZERO, POS_INF, POS_INF, POS_ONE), (POS_ZERO, POS_INF, NEG_INF, POS_ZERO)),
        ((POS_INF, NEG_INF, POS_ZERO, NEG_ZERO), (POS_ZERO, POS_ZERO, POS_INF, NEG_INF)),

        # 4. Single NaNs against normals, zeros, infinities, and denormals
        ((POS_QNAN1, NEG_QNAN1, POS_SNAN1, NEG_SNAN1), (POS_ONE, NEG_ZERO, POS_INF, POS_DENORM_MIN)),
        ((POS_ONE, NEG_ZERO, POS_INF, POS_DENORM_MIN), (POS_QNAN1, NEG_QNAN1, POS_SNAN1, NEG_SNAN1)),

        # 5. Double NaNs: qNaN vs qNaN, sNaN vs sNaN, qNaN vs sNaN, sNaN vs qNaN with distinct signs and payloads
        ((POS_QNAN1, NEG_QNAN1, POS_SNAN1, NEG_SNAN1), (POS_QNAN2, NEG_QNAN2, POS_SNAN2, NEG_SNAN2)),
        ((POS_QNAN1, NEG_QNAN1, POS_SNAN1, NEG_SNAN1), (POS_SNAN2, NEG_SNAN2, POS_QNAN2, NEG_QNAN2)),
        ((POS_QNAN_MIN, NEG_QNAN_MAX, POS_SNAN_MIN, NEG_SNAN_MAX), (NEG_QNAN_MAX, POS_QNAN_MIN, NEG_SNAN_MAX, POS_SNAN_MIN)),
    ]

    out_n = (ctypes.c_uint8 * 16)()
    out_l = (ctypes.c_uint8 * 16)()

    # Verify that lifted function entry enforces profile MXCSR (0x1f80, control bits 6-15)
    # and preserves caller sticky bits (bits 0-5), exercising the emitted _mm_setcsr line.
    NON_PROFILE_MXCSR = 0x9fa5  # FTZ (bit 15) on, DAZ (bit 6) off, all exceptions masked, sticky bits 0x25
    lifted.set_mxcsr(NON_PROFILE_MXCSR)
    if (lifted.get_mxcsr() & 0xffff) != NON_PROFILE_MXCSR:
        raise AssertionError("failed to pre-set non-profile MXCSR")
    dummy_a = ctypes.create_string_buffer(struct.pack("<4I", POS_ONE, POS_ONE, POS_ONE, POS_ONE), 16)
    dummy_b = ctypes.create_string_buffer(struct.pack("<4I", POS_TWO, POS_TWO, POS_TWO, POS_TWO), 16)
    dummy_out = ctypes.create_string_buffer(16)
    lifted.addps_op(dummy_a, dummy_b, dummy_out)
    post_mxcsr = lifted.get_mxcsr()
    if (post_mxcsr & 0xffc0) != 0x1f80:
        raise AssertionError(
            f"lifted entry did not establish profile MXCSR control bits 0x1f80: got {post_mxcsr:#x}"
        )
    if (post_mxcsr & 0x3f) != (NON_PROFILE_MXCSR & 0x3f):
        raise AssertionError(
            f"lifted entry corrupted caller sticky bits: got {post_mxcsr & 0x3f:#x}, expected {NON_PROFILE_MXCSR & 0x3f:#x}"
        )

    # Profile MXCSR: exceptions masked (bits 7-12 = 0x1f80), nearest-even rounding (bits 13-14 = 0),
    # DAZ and FTZ off (bits 6 and 15 = 0), control mask 0xffc0 matching 0x1f80.
    # Test under both clean sticky bits and existing sticky bits (bits 0-5 = 0x3f) allowed.
    profile_mxcsrs = [
        (orig_mxcsr & 0x3f) | 0x1f80,
        0x1f80 | 0x3f,
    ]

    cases = 0
    try:
        for pmxcsr in profile_mxcsrs:
            lifted.set_mxcsr(pmxcsr)
            current_mxcsr = lifted.get_mxcsr()
            if (current_mxcsr & 0xffc0) != 0x1f80:
                raise AssertionError(f"MXCSR control bits {current_mxcsr:#x} do not match profile 0x1f80")

            for va, vb in test_vectors:
                bytes_a = struct.pack("<4I", *va)
                bytes_b = struct.pack("<4I", *vb)
                buf_a = ctypes.create_string_buffer(bytes_a, 16)
                buf_b = ctypes.create_string_buffer(bytes_b, 16)

                # Binary arithmetic: test BOTH operand orders (va, vb) and (vb, va)
                for name in BINARY_EXPORTS:
                    # Order 1: (va, vb)
                    ctypes.memset(out_n, 0xaa, 16)
                    ctypes.memset(out_l, 0x55, 16)
                    getattr(native, name)(buf_a, buf_b, out_n)
                    getattr(lifted, name)(buf_a, buf_b, out_l)
                    words_n = struct.unpack("<4I", out_n)
                    words_l = struct.unpack("<4I", out_l)
                    if words_n != words_l:
                        raise AssertionError(
                            f"{name}({tuple(hex(x) for x in va)}, {tuple(hex(x) for x in vb)}): "
                            f"lifted {[hex(x) for x in words_l]} != native {[hex(x) for x in words_n]}"
                        )
                    cases += 1

                    # Order 2: (vb, va)
                    ctypes.memset(out_n, 0xaa, 16)
                    ctypes.memset(out_l, 0x55, 16)
                    getattr(native, name)(buf_b, buf_a, out_n)
                    getattr(lifted, name)(buf_b, buf_a, out_l)
                    words_n = struct.unpack("<4I", out_n)
                    words_l = struct.unpack("<4I", out_l)
                    if words_n != words_l:
                        raise AssertionError(
                            f"{name}({tuple(hex(x) for x in vb)}, {tuple(hex(x) for x in va)}): "
                            f"lifted {[hex(x) for x in words_l]} != native {[hex(x) for x in words_n]}"
                        )
                    cases += 1

                # Unary self-arithmetic: test both va and vb
                for name in SELF_EXPORTS:
                    for v, buf in ((va, buf_a), (vb, buf_b)):
                        ctypes.memset(out_n, 0xaa, 16)
                        ctypes.memset(out_l, 0x55, 16)
                        getattr(native, name)(buf, buf, out_n)
                        getattr(lifted, name)(buf, buf, out_l)
                        words_n = struct.unpack("<4I", out_n)
                        words_l = struct.unpack("<4I", out_l)
                        if words_n != words_l:
                            raise AssertionError(
                                f"{name}({tuple(hex(x) for x in v)}): "
                                f"lifted {[hex(x) for x in words_l]} != native {[hex(x) for x in words_n]}"
                            )
                        cases += 1
    finally:
        lifted.set_mxcsr(orig_mxcsr)

    print(f"{cases} baseline 128-bit SSE packed single-precision arithmetic cases matched native and translated DLLs")


def check_rejections():
    # EVEX forms (vaddps, vsubps, vmulps, vdivps) must be rejected.
    evex_opcodes = [
        ("vaddps", "62 f1 74 48 58 c2 c3"),
        ("vsubps", "62 f1 74 48 5c c2 c3"),
        ("vmulps", "62 f1 74 48 59 c2 c3"),
        ("vdivps", "62 f1 74 48 5e c2 c3"),
    ]
    for name, hexcode in evex_opcodes:
        evex = list(Decoder(64, bytes.fromhex(hexcode), ip=0x1000))
        try:
            lower(evex, [], {}, {}, {})
        except ValueError as exc:
            if not str(exc).startswith("unsupported float packed operation at 0x1000") and not str(exc).startswith("unsupported query operation at 0x1000"):
                raise AssertionError(f"unexpected diagnostic {exc}") from exc
        else:
            raise AssertionError(f"accepted EVEX {name}")

    # Legacy XMM memory forms must be rejected without an alignment contract.
    memory_opcodes = [
        ("addps", "0f 58 02 c3"),
        ("subps", "0f 5c 02 c3"),
        ("mulps", "0f 59 02 c3"),
        ("divps", "0f 5e 02 c3"),
    ]
    for name, hexcode in memory_opcodes:
        memory = list(Decoder(64, bytes.fromhex(hexcode), ip=0x1000))
        try:
            lower(memory, [], {}, {}, {})
        except ValueError as exc:
            if not str(exc).startswith("unsupported SSE float arithmetic source at 0x1000"):
                raise AssertionError(f"unexpected diagnostic {exc}") from exc
        else:
            raise AssertionError(f"accepted an {name.upper()} memory operand without an alignment contract")

    # Untracked XMM register sources must be rejected.
    untracked_opcodes = [
        ("addps", "0f 58 c3 c3"),
        ("subps", "0f 5c c3 c3"),
        ("mulps", "0f 59 c3 c3"),
        ("divps", "0f 5e c3 c3"),
    ]
    for name, hexcode in untracked_opcodes:
        untracked = list(Decoder(64, bytes.fromhex(hexcode), ip=0x1000))
        try:
            lower(untracked, [], {}, {}, {})
        except ValueError as exc:
            if not str(exc).startswith("unsupported SSE float arithmetic source at 0x1000"):
                raise AssertionError(f"unexpected diagnostic {exc}") from exc
        else:
            raise AssertionError(f"accepted an {name.upper()} untracked register operand")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check_rejections()
        check(native, other)
    else:
        raise ValueError(mode)

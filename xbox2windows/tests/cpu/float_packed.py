"""Compare native and translated packed and scalar SSE/SSE2/AVX floating point instructions.

MXCSR Configuration Established:
This differential fixture establishes and assumes the default x64 architectural MXCSR state
(0x1F80, standard Windows ABI environment on thread creation):
  - Exception Masks (bits 7-12: PE, UE, OE, ZE, DE, IE) = 111111b (all floating-point
    exceptions masked; operations on NaNs, overflows, underflows, and divisions by zero
    produce IEEE-754 quiet NaNs, infinities, or subnormals without trapping).
  - Rounding Control (bits 13-14: RC) = 00b (Round to nearest / even).
  - Flush-to-Zero (bit 15: FTZ) = 0 (disabled; subnormals preserved in full precision).
  - Denormals-Are-Zeros (bit 6: DAZ) = 0 (disabled; subnormal source operands preserved).
"""
import ctypes
from pathlib import Path
import struct
import sys

from iced_x86 import Decoder
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64

EXPORTS = (
    # Legacy SSE packed
    "addps", "subps", "mulps", "divps", "maxps", "minps", "sqrtps",
    "andps", "andnps", "orps", "xorps",
    # Legacy SSE scalar
    "sqrtss", "maxss", "minss",
    # Legacy SSE2 packed
    "addpd", "subpd", "mulpd", "divpd", "maxpd", "minpd", "sqrtpd",
    "andpd", "andnpd", "orpd", "xorpd",
    # Legacy SSE2 scalar
    "sqrtsd", "maxsd", "minsd",
    # VEX AVX packed single
    "vaddps", "vsubps", "vmulps", "vdivps", "vmaxps", "vminps", "vsqrtps",
    "vandps", "vandnps", "vorps", "vxorps",
    # VEX AVX packed double
    "vaddpd", "vsubpd", "vmulpd", "vdivpd", "vmaxpd", "vminpd", "vsqrtpd",
    "vandpd", "vandnpd", "vorpd", "vxorpd",
    # VEX AVX scalar single
    "vsqrtss", "vmaxss", "vminss",
    # VEX AVX scalar double
    "vsqrtsd", "vmaxsd", "vminsd",
)
EXPORT_NAMES = tuple(f"{op}_op" for op in EXPORTS)


def emit(image, output):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORT_NAMES:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word dst) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORT_NAMES:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], U64
    return lib


def aligned_buffer(data_bytes):
    raw = ctypes.create_string_buffer(len(data_bytes) + 32)
    addr = ctypes.addressof(raw)
    off = (16 - (addr % 16)) % 16
    ptr = addr + off
    ctypes.memmove(ptr, data_bytes, len(data_bytes))
    return raw, ptr


def check(native_path, lifted_path):
    # Rejection controls: reject YMM 256-bit width, memory operands, and untracked registers.
    unsupported_hex = (
        "c5 f4 58 c2",  # vaddps ymm0, ymm1, ymm2 (YMM 256-bit width)
        "0f 58 01",     # addps xmm0, [rcx] (memory operand)
        "c5 f0 58 01",  # vaddps xmm0, xmm1, [rcx] (memory operand)
        "0f 58 c3",     # addps xmm0, xmm3 (untracked XMM register operand 1)
        "0f 58 d8",     # addps xmm3, xmm0 (untracked XMM register destination)
        "c5 e8 58 c3",  # vaddps xmm0, xmm2, xmm3 (untracked XMM register operand 2)
    )
    for raw in unsupported_hex:
        try:
            lower(list(Decoder(64, bytes.fromhex(raw))), [], {}, {}, {})
        except ValueError as error:
            assert "unsupported" in str(error), error
        else:
            raise AssertionError(f"accepted unsupported instruction {raw}")

    native, lifted = load_lib(native_path), load_lib(lifted_path)

    sentinel_a = b"\x5a" * 16
    sentinel_b = b"\xa5" * 16

    # Floating point test components:
    # Quiet NaNs with distinct positive and negative payloads
    qnan_pos = 0x7fc01111
    qnan_neg = 0xffc02222
    # Signaling NaNs with distinct positive and negative payloads
    snan_pos = 0x7f803333
    snan_neg = 0xffa04444
    # Opposite signed zeros
    zero_pos = 0x00000000
    zero_neg = 0x80000000
    # Regular normal values
    norm_pos = struct.unpack("<I", struct.pack("<f", 2.0))[0]
    norm_neg = struct.unpack("<I", struct.pack("<f", -2.0))[0]

    # Test patterns: full 32 bytes (16 bytes low lane + 16 bytes sentinel)
    # 1. Standard arithmetic floats
    p_f1 = struct.pack("<4f", 1.5, -2.5, 3.25, -4.75) + sentinel_a
    p_f2 = struct.pack("<4f", 0.5, 2.0, -1.0, 4.0) + sentinel_b
    p_f3 = struct.pack("<4f", 4.0, 9.0, 16.0, 25.0) + sentinel_a
    p_f4 = struct.pack("<4f", 2.0, 3.0, 4.0, 5.0) + sentinel_b

    # 2. Opposite signed zeros (tested in both operand orders)
    p_zero_a = struct.pack("<4I", zero_pos, zero_neg, zero_pos, zero_neg) + sentinel_a
    p_zero_b = struct.pack("<4I", zero_neg, zero_pos, zero_neg, zero_pos) + sentinel_b

    # 3. Two different signed NaN payloads (quiet and signaling) tested against each other
    p_nan_ab1 = struct.pack("<4I", qnan_pos, snan_pos, qnan_pos, snan_pos) + sentinel_a
    p_nan_ab2 = struct.pack("<4I", qnan_neg, snan_neg, snan_neg, qnan_neg) + sentinel_b

    # 4. NaNs tested against regular numbers (both operand orders)
    p_nan_norm1 = struct.pack("<4I", qnan_pos, snan_pos, qnan_neg, snan_neg) + sentinel_a
    p_nan_norm2 = struct.pack("<4I", norm_pos, norm_neg, norm_pos, norm_neg) + sentinel_b

    # 5. Bitwise patterns
    p_bit1 = struct.pack("<4I", 0xaaaaaaaa, 0x55555555, 0x0f0f0f0f, 0xf0f0f0f0) + sentinel_a
    p_bit2 = struct.pack("<4I", 0xffffffff, 0x00000000, 0x12345678, 0x9abcdef0) + sentinel_b

    # 6. Double precision components
    dqnan_pos = 0x7ff8000012345678
    dqnan_neg = 0xfff8000087654321
    dsnan_pos = 0x7ff0000000000001
    dsnan_neg = 0xfff4000000000002
    dzero_pos = 0x0000000000000000
    dzero_neg = 0x8000000000000000
    dnorm_pos = struct.unpack("<Q", struct.pack("<d", 3.5))[0]
    dnorm_neg = struct.unpack("<Q", struct.pack("<d", -3.5))[0]

    p_d1 = struct.pack("<2d", 2.25, -7.5) + sentinel_a
    p_d2 = struct.pack("<2d", 1.5, 2.5) + sentinel_b
    p_d3 = struct.pack("<2d", 49.0, 81.0) + sentinel_a
    p_d4 = struct.pack("<2d", 7.0, 9.0) + sentinel_b

    # Double precision opposite signed zeros
    p_d_zero_a = struct.pack("<2Q", dzero_pos, dzero_neg) + sentinel_a
    p_d_zero_b = struct.pack("<2Q", dzero_neg, dzero_pos) + sentinel_b

    # Double precision signed NaNs (quiet and signaling)
    p_d_nan_ab1 = struct.pack("<2Q", dqnan_pos, dsnan_pos) + sentinel_a
    p_d_nan_ab2 = struct.pack("<2Q", dqnan_neg, dsnan_neg) + sentinel_b

    # Double precision NaNs vs normals
    p_d_nan_norm1 = struct.pack("<2Q", dqnan_pos, dsnan_neg) + sentinel_a
    p_d_nan_norm2 = struct.pack("<2Q", dnorm_pos, dnorm_neg) + sentinel_b

    patterns = [
        # Regular float arithmetic
        (p_f1, p_f2),
        (p_f3, p_f4),
        # Opposite signed zeros in both operand orders
        (p_zero_a, p_zero_b),
        (p_zero_b, p_zero_a),
        # Signed quiet and signaling NaNs in both operand orders
        (p_nan_ab1, p_nan_ab2),
        (p_nan_ab2, p_nan_ab1),
        # NaNs vs regular numbers in both operand orders
        (p_nan_norm1, p_nan_norm2),
        (p_nan_norm2, p_nan_norm1),
        # Bitwise logic
        (p_bit1, p_bit2),
        # Regular double arithmetic
        (p_d1, p_d2),
        (p_d3, p_d4),
        # Double precision opposite signed zeros in both operand orders
        (p_d_zero_a, p_d_zero_b),
        (p_d_zero_b, p_d_zero_a),
        # Double precision signed NaNs in both operand orders
        (p_d_nan_ab1, p_d_nan_ab2),
        (p_d_nan_ab2, p_d_nan_ab1),
        # Double precision NaNs vs normals in both operand orders
        (p_d_nan_norm1, p_d_nan_norm2),
        (p_d_nan_norm2, p_d_nan_norm1),
    ]

    comparisons = 0
    for a_bytes, b_bytes in patterns:
        raw_a, ptr_a = aligned_buffer(a_bytes)
        raw_b, ptr_b = aligned_buffer(b_bytes)
        raw_out_n, ptr_out_n = aligned_buffer(b"\xaa" * 32)
        raw_out_l, ptr_out_l = aligned_buffer(b"\x55" * 32)

        for name in EXPORT_NAMES:
            ctypes.memset(ptr_out_n, 0xaa, 32)
            ctypes.memset(ptr_out_l, 0x55, 32)
            getattr(native, name)(ptr_a, ptr_b, ptr_out_n)
            getattr(lifted, name)(ptr_a, ptr_b, ptr_out_l)

            res_n = ctypes.string_at(ptr_out_n, 32)
            res_l = ctypes.string_at(ptr_out_l, 32)
            if res_n != res_l:
                raise AssertionError(f"{name}: full 256-bit vector mismatch native {res_n.hex()} != lifted {res_l.hex()}")

            # Verify architectural high-half behavior: legacy XMM preserves, VEX.128 clears
            if name.startswith("v"):
                if res_l[16:32] != b"\x00" * 16:
                    raise AssertionError(f"{name}: VEX.128 write did not clear high vector bits: {res_l[16:32].hex()}")
            else:
                if res_l[16:32] != a_bytes[16:32]:
                    raise AssertionError(f"{name}: legacy XMM write did not preserve high vector bits: {res_l[16:32].hex()}")

            comparisons += 1

    # Aliased operand testing: both source pointers refer to the same buffer
    for p in (p_f1, p_f3, p_zero_a, p_nan_ab1, p_d1, p_d_zero_a, p_d_nan_ab1, p_bit1):
        raw_aliased, ptr_aliased = aligned_buffer(p)
        raw_out_n, ptr_out_n = aligned_buffer(b"\xaa" * 32)
        raw_out_l, ptr_out_l = aligned_buffer(b"\x55" * 32)
        for name in EXPORT_NAMES:
            ctypes.memset(ptr_out_n, 0xaa, 32)
            ctypes.memset(ptr_out_l, 0x55, 32)
            getattr(native, name)(ptr_aliased, ptr_aliased, ptr_out_n)
            getattr(lifted, name)(ptr_aliased, ptr_aliased, ptr_out_l)
            res_n = ctypes.string_at(ptr_out_n, 32)
            res_l = ctypes.string_at(ptr_out_l, 32)
            if res_n != res_l:
                raise AssertionError(f"{name}: aliased 256-bit vector mismatch native {res_n.hex()} != lifted {res_l.hex()}")
            comparisons += 1

    print(f"All {comparisons} float arithmetic operations verified across full 256-bit vector state, signed NaNs/zeros, and aliased operands")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(native, other)
    else:
        raise ValueError(mode)

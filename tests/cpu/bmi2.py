"""Compare native and translated BMI2 bit-manipulation instructions (32-bit and 64-bit)."""
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
MASK32 = 0xffffffff
MASK64 = 0xffffffffffffffff

BZHI_EXPORTS = ("bzhi32", "bzhi64")
MULX_EXPORTS = ("mulx32", "mulx64")
MULX_SAME_EXPORTS = ("mulx32_same", "mulx64_same")
PDEP_PEXT_EXPORTS = ("pdep32", "pdep64", "pext32", "pext64")
SHIFT_EXPORTS = ("sarx32", "sarx64", "shlx32", "shlx64", "shrx32", "shrx64")
RORX_EXPORTS = (
    "rorx32_0", "rorx32_1", "rorx32_7", "rorx32_15", "rorx32_31", "rorx32_32", "rorx32_33", "rorx32_255",
    "rorx64_0", "rorx64_1", "rorx64_7", "rorx64_31", "rorx64_63", "rorx64_64", "rorx64_255",
)


def emit(image, output):
    parts = [
        "#include <intrin.h>\n",
        INTEGER_HEADER,
        'extern "C" __declspec(dllexport) int host_has_bmi2() {\n'
        "    int regs[4]{};\n"
        "    __cpuid(regs, 0);\n"
        "    if (regs[0] < 7) return 0;\n"
        "    __cpuidex(regs, 7, 0);\n"
        "    return (regs[1] & (1 << 8)) != 0;\n"
        "}\n",
    ]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in BZHI_EXPORTS + MULX_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = b; r8 = out_param;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word out_param) {{\n{code}\n}}\n')
        for name in MULX_SAME_EXPORTS + PDEP_PEXT_EXPORTS + SHIFT_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = b;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b) {{\n{code}\n}}\n')
        for name in RORX_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in BZHI_EXPORTS + MULX_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, ctypes.POINTER(U64)], U64
    for name in MULX_SAME_EXPORTS + PDEP_PEXT_EXPORTS + SHIFT_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64], U64
    for name in RORX_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64], U64
    return lib


def ref_bzhi(a, idx, bits):
    mask = (1 << bits) - 1
    a &= mask
    idx &= 0xff
    if idx < bits:
        m = 0 if idx == 0 else ((1 << idx) - 1)
        cf = 0
    else:
        m = mask
        cf = 1
    v = a & m
    zf = 1 if v == 0 else 0
    return v, (cf << 1) | zf


def ref_mulx(a, b, bits):
    mask = (1 << bits) - 1
    a &= mask
    b &= mask
    prod = a * b
    lo = prod & mask
    hi = (prod >> bits) & mask
    return lo, hi


def ref_pdep(a, mask_val, bits):
    mask = (1 << bits) - 1
    a &= mask
    mask_val &= mask
    result = 0
    temp = a
    for k in range(bits):
        if (mask_val >> k) & 1:
            result |= (temp & 1) << k
            temp >>= 1
    return result


def ref_pext(a, mask_val, bits):
    mask = (1 << bits) - 1
    a &= mask
    mask_val &= mask
    result = 0
    k = 0
    for m_idx in range(bits):
        if (mask_val >> m_idx) & 1:
            result |= ((a >> m_idx) & 1) << k
            k += 1
    return result


def ref_rorx(a, count, bits):
    mask = (1 << bits) - 1
    a &= mask
    c = count & (bits - 1)
    if c == 0:
        return a
    return ((a >> c) | (a << (bits - c))) & mask


def ref_sarx(a, count, bits):
    mask = (1 << bits) - 1
    c = count & (bits - 1)
    if bits == 32:
        val = (a & 0xffffffff) if (a & 0x80000000) == 0 else -((~a & 0xffffffff) + 1)
        return (val >> c) & mask
    val = (a & 0xffffffffffffffff) if (a & 0x8000000000000000) == 0 else -((~a & 0xffffffffffffffff) + 1)
    return (val >> c) & mask


def ref_shlx(a, count, bits):
    mask = (1 << bits) - 1
    c = count & (bits - 1)
    return ((a & mask) << c) & mask


def ref_shrx(a, count, bits):
    mask = (1 << bits) - 1
    c = count & (bits - 1)
    return ((a & mask) >> c) & mask


def check(native_path, lifted_path):
    lifted = load_lib(lifted_path)
    lifted.host_has_bmi2.argtypes, lifted.host_has_bmi2.restype = [], ctypes.c_int
    if not lifted.host_has_bmi2():
        print("Host CPU lacks BMI2 (CPUID leaf 7 EBX bit 8); skipping native differential check.")
        sys.exit(77)
    native = load_lib(native_path)
    flag_slot = U64(0)
    values = (
        0, 1, 2, 3, 0x80, 0x100, 0x55aa55aa, 0x7fffffff, 0x80000000, 0xffffffff,
        0x100000000, 0xdeadbeef00000000, 0xdeadbeef00000001, 0x8000000000000000,
        0x7fffffffffffffff, 0xffffffffffffffff, 0x0123456789abcdef,
    )
    counts = (0, 1, 2, 7, 15, 31, 32, 33, 63, 64, 65, 127, 128, 255)
    masks = (
        0, 1, 0x55555555, 0xaaaaaaaa, 0x80000001, 0x0000ffff, 0xffff0000, 0xffffffff,
        0x5555555555555555, 0xaaaaaaaaaaaaaaaa, 0x8000000000000001, 0xffffffffffffffff,
    )

    # 1. BZHI
    bzhi_checks = 0
    for name, bits in (("bzhi32", 32), ("bzhi64", 64)):
        for a in values:
            for idx in counts:
                want_v, want_f = ref_bzhi(a, idx, bits)
                for lib, label in ((native, "native"), (lifted, "lifted")):
                    flag_slot.value = 0xdeadbeef
                    got_v = getattr(lib, name)(a, idx, ctypes.byref(flag_slot))
                    if (got_v, flag_slot.value) != (want_v, want_f):
                        raise AssertionError(
                            f"{label} {name}({a:#x}, {idx}) = ({got_v:#x}, {flag_slot.value:#x}), "
                            f"expected ({want_v:#x}, {want_f:#x})"
                        )
                bzhi_checks += 1

    # 2. MULX
    mulx_checks = 0
    for name, same_name, bits in (("mulx32", "mulx32_same", 32), ("mulx64", "mulx64_same", 64)):
        for a in values:
            for b in values:
                want_lo, want_hi = ref_mulx(a, b, bits)
                for lib, label in ((native, "native"), (lifted, "lifted")):
                    flag_slot.value = 0xdeadbeef
                    got_lo = getattr(lib, name)(a, b, ctypes.byref(flag_slot))
                    got_hi = flag_slot.value
                    if (got_lo, got_hi) != (want_lo, want_hi):
                        raise AssertionError(
                            f"{label} {name}({a:#x}, {b:#x}) = ({got_lo:#x}, {got_hi:#x}), "
                            f"expected ({want_lo:#x}, {want_hi:#x})"
                        )
                    # Check dst_hi == dst_lo (hi wins per Intel SDM)
                    got_same = getattr(lib, same_name)(a, b)
                    if got_same != want_hi:
                        raise AssertionError(
                            f"{label} {same_name}({a:#x}, {b:#x}) = {got_same:#x}, expected {want_hi:#x}"
                        )
                mulx_checks += 1

    # 3. PDEP & PEXT
    bit_checks = 0
    for name, fn, bits in (
        ("pdep32", ref_pdep, 32), ("pdep64", ref_pdep, 64),
        ("pext32", ref_pext, 32), ("pext64", ref_pext, 64),
    ):
        for a in values:
            for m in masks:
                want_v = fn(a, m, bits)
                for lib, label in ((native, "native"), (lifted, "lifted")):
                    got_v = getattr(lib, name)(a, m)
                    if got_v != want_v:
                        raise AssertionError(
                            f"{label} {name}({a:#x}, {m:#x}) = {got_v:#x}, expected {want_v:#x}"
                        )
                bit_checks += 1

    # 4. RORX
    rorx_checks = 0
    for export_name in RORX_EXPORTS:
        bits = 32 if "32" in export_name else 64
        count = int(export_name.split("_")[-1])
        for a in values:
            want_v = ref_rorx(a, count, bits)
            for lib, label in ((native, "native"), (lifted, "lifted")):
                got_v = getattr(lib, export_name)(a)
                if got_v != want_v:
                    raise AssertionError(
                        f"{label} {export_name}({a:#x}) = {got_v:#x}, expected {want_v:#x}"
                    )
            rorx_checks += 1

    # 5. SARX / SHLX / SHRX
    shift_checks = 0
    for name, fn, bits in (
        ("sarx32", ref_sarx, 32), ("sarx64", ref_sarx, 64),
        ("shlx32", ref_shlx, 32), ("shlx64", ref_shlx, 64),
        ("shrx32", ref_shrx, 32), ("shrx64", ref_shrx, 64),
    ):
        for a in values:
            for c in counts:
                want_v = fn(a, c, bits)
                for lib, label in ((native, "native"), (lifted, "lifted")):
                    got_v = getattr(lib, name)(a, c)
                    if got_v != want_v:
                        raise AssertionError(
                            f"{label} {name}({a:#x}, {c}) = {got_v:#x}, expected {want_v:#x}"
                        )
                shift_checks += 1

    print(
        f"{bzhi_checks} BZHI, {mulx_checks} MULX, {bit_checks} PDEP/PEXT, "
        f"{rorx_checks} RORX, and {shift_checks} shift cases matched native and translated DLLs"
    )


def check_rejections():
    # 16-bit forms (e.g. 66h prefix on bzhi, rorx, shlx) are outside the 32/64-bit contract.
    for hex_bytes, desc in (
        ("66 c4 e2 78 f5 c1 c3", "16-bit bzhi"),
        ("66 c4 e2 7b f6 c9 c3", "16-bit mulx"),
        ("66 c4 e2 7b f5 c1 c3", "16-bit pdep"),
        ("66 c4 e2 7a f5 c1 c3", "16-bit pext"),
        ("66 c4 e3 79 f0 c1 07 c3", "16-bit rorx"),
        ("66 c4 e2 79 f7 c1 c3", "16-bit shlx"),
        ("66 c4 e2 7a f7 c1 c3", "16-bit sarx"),
        ("66 c4 e2 7b f7 c1 c3", "16-bit shrx"),
    ):
        try:
            lower(list(Decoder(64, bytes.fromhex(hex_bytes), ip=0x1000)), [], {}, {}, {})
        except ValueError as exc:
            if not str(exc).startswith("unsupported query operation at 0x1000"):
                raise AssertionError(f"unexpected diagnostic for {desc}: {exc}") from exc
        else:
            raise AssertionError(f"accepted {desc}")


if __name__ == "__main__":
    mode, native, *others = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(others[0]))
    elif mode == "check":
        check(native, others[0])
        check_rejections()
    else:
        raise ValueError(mode)

"""Compare native and translated BMI1 bit-manipulation instructions (32-bit and 64-bit)."""
import ctypes
from pathlib import Path
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

BINARY_EXPORTS = ("andn32", "andn64", "bextr32", "bextr64")
UNARY_EXPORTS = ("blsi32", "blsi64", "blsmsk32", "blsmsk64", "blsr32", "blsr64", "tzcnt32", "tzcnt64")


def emit(image, output):
    parts = [
        "#include <intrin.h>\n",
        INTEGER_HEADER,
        'extern "C" __declspec(dllexport) int host_has_bmi1() {\n'
        "    int regs[4]{};\n"
        "    __cpuid(regs, 0);\n"
        "    if (regs[0] < 7) return 0;\n"
        "    __cpuidex(regs, 7, 0);\n"
        "    return (regs[1] & (1 << 3)) != 0;\n"
        "}\n",
    ]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in BINARY_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = b; r8 = flags_out;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word flags_out) {{\n{code}\n}}\n')
        for name in UNARY_EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = flags_out;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word flags_out) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in BINARY_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, ctypes.POINTER(U64)], U64
    for name in UNARY_EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, ctypes.POINTER(U64)], U64
    return lib


def ref_andn(a, b, bits):
    mask = (1 << bits) - 1
    v = (~a & b) & mask
    return v, (1 if v == 0 else 0)


def ref_bextr(a, ctrl, bits):
    mask = (1 << bits) - 1
    a &= mask
    start = ctrl & 0xff
    length = (ctrl >> 8) & 0xff
    if start >= bits or length == 0:
        v = 0
    else:
        shifted = a >> start
        v = shifted if length >= bits else shifted & ((1 << length) - 1)
    return v, (1 if v == 0 else 0)


def ref_blsi(a, bits):
    mask = (1 << bits) - 1
    a &= mask
    v = (a & (-a)) & mask
    return v, ((2 if a != 0 else 0) | (1 if v == 0 else 0))


def ref_blsmsk(a, bits):
    mask = (1 << bits) - 1
    a &= mask
    v = (a ^ (a - 1)) & mask
    return v, (2 if a == 0 else 0)


def ref_blsr(a, bits):
    mask = (1 << bits) - 1
    a &= mask
    v = (a & (a - 1)) & mask
    return v, ((2 if a == 0 else 0) | (1 if v == 0 else 0))


def ref_tzcnt(a, bits):
    mask = (1 << bits) - 1
    a &= mask
    if a == 0:
        return bits, 2
    tz = (a & -a).bit_length() - 1
    return tz, (1 if tz == 0 else 0)


def check(native_path, lifted_path):
    lifted = load_lib(lifted_path)
    lifted.host_has_bmi1.argtypes, lifted.host_has_bmi1.restype = [], ctypes.c_int
    if not lifted.host_has_bmi1():
        print("Host CPU lacks BMI1 (CPUID leaf 7 EBX bit 3); skipping native differential check.")
        sys.exit(77)
    native = load_lib(native_path)
    flag_slot = U64(0)
    values = (
        0, 1, 2, 3, 0x80, 0x100, 0x55aa55aa, 0x7fffffff, 0x80000000, 0xffffffff,
        0x100000000, 0xdeadbeef00000000, 0xdeadbeef00000001, 0x8000000000000000,
        0x7fffffffffffffff, 0xffffffffffffffff, 0x0123456789abcdef,
    )
    controls = (
        0x0000, 0x0100, 0x0400, 0x0804, 0x1010, 0x2000, 0x201f, 0x2020,
        0x4000, 0x403f, 0x4040, 0xff00, 0xffff, 0x0838,
        # Bits above 15 are ignored: the length is bits 15..8 only.
        0x10100, 0x232008, 0xfe0410,
        # Bit 7 of each field matters (length 128, start 128), and lengths 31 and 63 are one
        # short of the operand width.
        0x8000, 0x0180, 0x1f00, 0x3f00,
    )

    binary_checks = 0
    for name, fn, bits, rhs_values in (
        ("andn32", ref_andn, 32, values),
        ("andn64", ref_andn, 64, values),
        ("bextr32", ref_bextr, 32, controls),
        ("bextr64", ref_bextr, 64, controls),
    ):
        for a in values:
            for b in rhs_values:
                want_v, want_f = fn(a, b, bits)
                for lib, label in ((native, "native"), (lifted, "lifted")):
                    flag_slot.value = 0xdeadbeef
                    got_v = getattr(lib, name)(a, b, ctypes.byref(flag_slot))
                    if (got_v, flag_slot.value) != (want_v, want_f):
                        raise AssertionError(
                            f"{label} {name}({a:#x}, {b:#x}) = ({got_v:#x}, {flag_slot.value:#x}), "
                            f"expected ({want_v:#x}, {want_f:#x})"
                        )
                binary_checks += 1

    unary_checks = 0
    for name, fn, bits in (
        ("blsi32", ref_blsi, 32),
        ("blsi64", ref_blsi, 64),
        ("blsmsk32", ref_blsmsk, 32),
        ("blsmsk64", ref_blsmsk, 64),
        ("blsr32", ref_blsr, 32),
        ("blsr64", ref_blsr, 64),
        ("tzcnt32", ref_tzcnt, 32),
        ("tzcnt64", ref_tzcnt, 64),
    ):
        for a in values:
            want_v, want_f = fn(a, bits)
            for lib, label in ((native, "native"), (lifted, "lifted")):
                flag_slot.value = 0xdeadbeef
                got_v = getattr(lib, name)(a, ctypes.byref(flag_slot))
                if (got_v, flag_slot.value) != (want_v, want_f):
                    raise AssertionError(
                        f"{label} {name}({a:#x}) = ({got_v:#x}, {flag_slot.value:#x}), "
                        f"expected ({want_v:#x}, {want_f:#x})"
                    )
            unary_checks += 1

    print(f"{binary_checks} binary and {unary_checks} unary BMI1 cases (values + ZF/CF) matched native and translated DLLs")


def check_rejections():
    # 16-bit tzcnt (66 F3 0F BC C1 = tzcnt ax, cx) is outside the 32/64-bit lowering contract.
    try:
        lower(list(Decoder(64, bytes.fromhex("66 f3 0f bc c1 c3"), ip=0x1000)), [], {}, {}, {})
    except ValueError as exc:
        if not str(exc).startswith("unsupported query operation at 0x1000"):
            raise AssertionError(f"unexpected diagnostic {exc}") from exc
    else:
        raise AssertionError("accepted 16-bit tzcnt")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(native, other)
        check_rejections()
    else:
        raise ValueError(mode)

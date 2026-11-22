"""Compare native and translated 32-bit effective addresses (address-size prefix 67h)."""
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
# Upper halves the hardware must ignore when it forms a 32-bit address.
NOISE = (0, 0xdeadbeef00000000, 0xffffffff00000000)


def emit(image, output):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, params, initial in (
                ("lea_sum32", "word a, word b", "rcx = a; rdx = b;"),
                ("lea_scaled32", "word a, word b", "rcx = a; rdx = b;"),
                ("load32", "word a, word b", "rcx = a; rdx = b;"),
                ("store32", "word a, word b, word v", "rcx = a; rdx = b; r8 = v;")):
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            if pe.get_data(body[0].ip, 1) != b"\x67":
                raise ValueError(f"{name} does not start with a 67h-prefixed memory operand")
            code = lower(body, [initial], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}({params}) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def low_buffer(size):
    """Commit a page below 4 GiB so a zero-extended 32-bit address can reach it."""
    virtual_alloc = ctypes.windll.kernel32.VirtualAlloc
    virtual_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    virtual_alloc.restype = ctypes.c_void_p
    for hint in range(0x10000000, 0xf0000000, 0x01000000):
        address = virtual_alloc(hint, size, 0x3000, 0x04)  # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
        if address and address + size <= 1 << 32:
            return address
    raise RuntimeError("no committable page below 4 GiB")


def functions(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in ("lea_sum32", "lea_scaled32", "load32"):
        getattr(lib, name).argtypes, getattr(lib, name).restype = [U64, U64], U64
    lib.store32.argtypes, lib.store32.restype = [U64, U64, U64], U64
    return lib


def check(native_path, lifted_path):
    native, lifted = functions(native_path), functions(lifted_path)
    lea_cases = [(a | n, b | m) for a, b in ((0xffffffff, 1), (0x10, 0x20), (0x80000000, 0x80000000),
                                             (0, 0), (8, 0), (0x12345678, 0xfffffff0))
                 for n in NOISE for m in NOISE[:2]]
    for name, reference in (("lea_sum32", lambda a, b: (a + b) & MASK32),
                            ("lea_scaled32", lambda a, b: (a + b * 8 - 8) & MASK32)):
        for a, b in lea_cases:
            want = reference(a, b)
            got = getattr(native, name)(a, b)
            if got != want:
                raise AssertionError(f"native {name}({a:#x}, {b:#x}) = {got:#x}, expected {want:#x}")
            got = getattr(lifted, name)(a, b)
            if got != want:
                raise AssertionError(f"lifted {name}({a:#x}, {b:#x}) = {got:#x}, expected {want:#x}")

    base = low_buffer(0x1000)
    words = (ctypes.c_uint32 * 1024).from_address(base)
    # (base register, index register, word slot): direct, upper-half noise and a 32-bit wrap.
    memory_cases = [(base | n, 4 * slot | m, slot) for slot in (0, 3, 1023) for n in NOISE for m in NOISE[:2]]
    memory_cases += [(base + 0x100 | n, (-0x100 + 4 * 5) & MASK32, 5) for n in NOISE]
    for slot in range(1024):
        words[slot] = (0xa5000000 | slot * 0x10001) & MASK32
    for lib, label in ((native, "native"), (lifted, "lifted")):
        for a, b, slot in memory_cases:
            got = lib.load32(a, b)
            if got != words[slot]:
                raise AssertionError(f"{label} load32({a:#x}, {b:#x}) = {got:#x}, expected {words[slot]:#x}")
        for a, b, slot in memory_cases:
            before = list(words)
            value = 0x5a000000 | slot
            if lib.store32(a, b, value) != value or words[slot] != value:
                raise AssertionError(f"{label} store32({a:#x}, {b:#x}) missed slot {slot}")
            before[slot] = value
            if list(words) != before:
                raise AssertionError(f"{label} store32({a:#x}, {b:#x}) wrote outside slot {slot}")
    print(f"{2 * len(lea_cases)} lea and {2 * len(memory_cases)} load/store cases matched the 32-bit "
          "reference on the native and translated images")


def check_rejections():
    # EIP-relative operands would need a 32-bit instruction pointer, which a desktop image does not have.
    try:
        lower(list(Decoder(64, bytes.fromhex("67 8b 05 10 00 00 00 c3"), ip=0x1000)), [], {}, {}, {})
    except ValueError as exc:
        if not str(exc).startswith("unsupported EIP-relative operand at 0x1000"):
            raise AssertionError(f"unexpected diagnostic {exc}") from exc
    else:
        raise AssertionError("accepted an EIP-relative operand")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(native, other)
        check_rejections()
    else:
        raise ValueError(mode)

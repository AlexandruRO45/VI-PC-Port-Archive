"""Compare native and jump-table-translator output for 67h (32-bit) effective addresses."""
import ctypes
from pathlib import Path
import sys

from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import Lifter

U64 = ctypes.c_uint64
MASK32 = 0xffffffff
NOISE = (0, 0xdeadbeef00000000, 0xffffffff00000000)


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


def bind(native_path, lifted_path):
    native, lifted = ctypes.CDLL(str(Path(native_path).resolve())), ctypes.CDLL(str(Path(lifted_path).resolve()))
    pairs = {}
    for name in ("lea_wrap", "lea_back", "load_back", "store_ahead"):
        n, l = getattr(native, name), getattr(lifted, f"lifted_{name}")
        n.argtypes, n.restype = [U64, U64], U64
        l.argtypes, l.restype = [U64] * 4, U64
        pairs[name] = (n, lambda a, b, l=l: l(a, b, 0, 0))
    return pairs


def check(native_path, lifted_path):
    pairs = bind(native_path, lifted_path)
    cases = 0
    for name, reference in (("lea_wrap", lambda x: (x + 1) & MASK32), ("lea_back", lambda x: (x - 8) & MASK32)):
        for low in (0, 4, 8, 0x10, 0x7fffffff, 0xfffffff8, 0xffffffff):
            for noise in NOISE:
                x = low | noise
                for label, function in zip(("native", "lifted"), pairs[name]):
                    got = function(x, 0)
                    if got != reference(x):
                        raise AssertionError(f"{label} {name}({x:#x}) = {got:#x}, expected {reference(x):#x}")
                    cases += 1

    base = low_buffer(0x1000)
    words = (ctypes.c_uint32 * 1024).from_address(base)
    for slot in range(1024):
        words[slot] = 0xa5000000 | slot
    for slot in (2, 100, 1021):
        for noise in NOISE:
            x = (base + 4 * slot) | noise
            for label, (load, store) in zip(("native", "lifted"), zip(pairs["load_back"], pairs["store_ahead"])):
                got = load(x, 0)
                if got != words[slot - 2]:
                    raise AssertionError(f"{label} load_back({x:#x}) = {got:#x}, expected {words[slot - 2]:#x}")
                before = list(words)
                value = 0x5a000000 | slot
                if store(x, value) != value or words[slot + 2] != value:
                    raise AssertionError(f"{label} store_ahead({x:#x}) missed word {slot + 2}")
                before[slot + 2] = value
                if list(words) != before:
                    raise AssertionError(f"{label} store_ahead({x:#x}) wrote outside word {slot + 2}")
                cases += 2
    print(f"{cases} native and translated 32-bit address cases matched the reference")


def check_rejections():
    # EIP- and ESP-based operands have no host meaning in this leaf profile.
    for hexbytes in ("67 8b 05 10 00 00 00", "67 8b 04 24"):
        ins = next(iter(Decoder(64, bytes.fromhex(hexbytes), ip=0x1000)))
        try:
            Lifter().address(ins)
        except ValueError as exc:
            if not str(exc).startswith("unsupported memory operand at RVA 0x1000"):
                raise AssertionError(f"unexpected diagnostic {exc}") from exc
        else:
            raise AssertionError(f"accepted {ins}")


if __name__ == "__main__":
    check(*sys.argv[1:])
    check_rejections()

"""Compare native and translated forward string moves and stores (MOVS/STOS, REP, CLD)."""
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
MASK64 = 0xffffffffffffffff

# name: (element size, is a move, has REP)
EXPORTS = {
    "rep_movsb": (1, True, True), "rep_movsw": (2, True, True),
    "rep_movsd": (4, True, True), "rep_movsq": (8, True, True),
    "rep_stosb": (1, False, True), "rep_stosw": (2, False, True),
    "rep_stosd": (4, False, True), "rep_stosq": (8, False, True),
    "movsq_once": (8, True, False), "stosd_once": (4, False, False),
}
COUNTS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17)
VALUES = (0, 0x1122334455667788, 0xfedcba9876543210, MASK64)
BUFFER = 512
BASE = 128  # every access lies inside [BASE, BASE + 200 + 17 * 8), away from both buffer ends


def emit(image, output):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = dst; rdx = src; r8 = count;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word dst, word src, word count) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, U64], U64
    return lib


def pattern():
    return bytes((index * 37 + 11) & 0xff for index in range(BUFFER))


def reference(memory, name, dst, src, count):
    """Sequential model over a bytearray: one element per step, low addresses first."""
    size, move, rep = EXPORTS[name]
    steps = count if rep else 1
    d, s = dst, src
    for _ in range(steps):
        if move:
            memory[d:d + size] = memory[s:s + size]
            s += size
        else:
            memory[d:d + size] = (src & MASK64).to_bytes(8, "little")[:size]
        d += size
    rcx = 0 if rep else count
    rsi_advance = s - src if move else 0
    return ((((rcx << 20) + rsi_advance) << 20) + (d - dst)) & MASK64


def run(lib, name, dst_offset, src, count):
    """Call an export over a fresh patterned buffer. src is an offset for moves, a value for stores."""
    buffer = ctypes.create_string_buffer(pattern(), BUFFER)
    base = ctypes.addressof(buffer)
    move = EXPORTS[name][1]
    source = base + src if move else src
    # The return encodes RSI and RDI relative to their entry values, so it is independent of base.
    result = getattr(lib, name)(base + dst_offset, source, count)
    return result, bytes(buffer.raw)


def cases():
    for name, (size, move, _) in EXPORTS.items():
        for count in COUNTS:
            if move:
                # Disjoint, destination one element above the source (forward overlap),
                # and destination one element below the source (backward overlap).
                for dst, src in ((BASE + 200, BASE), (BASE + size, BASE), (BASE, BASE + size)):
                    yield name, dst, src, count
            else:
                for value in VALUES:
                    yield name, BASE, value, count


def check(native_path, lifted_path):
    native, lifted = load_lib(native_path), load_lib(lifted_path)
    checked = 0
    smeared = 0
    for name, dst, src, count in cases():
        size, move, rep = EXPORTS[name]
        memory = bytearray(pattern())
        want = reference(memory, name, dst, src, count)
        want_memory = bytes(memory)
        for lib, label in ((native, "native"), (lifted, "lifted")):
            got, got_memory = run(lib, name, dst, src, count)
            if (got, got_memory) != (want, want_memory):
                first = next((n for n, (a, b) in enumerate(zip(got_memory, want_memory)) if a != b), None)
                raise AssertionError(
                    f"{label} {name}(dst={dst}, src={src:#x}, count={count}) = {got:#x}, "
                    f"expected {want:#x}; first differing byte {first}")
        if move and rep and dst == src + size and count >= 2:
            # A memmove of the same range differs from the element-by-element result, so this
            # case would catch a block-copy lowering.
            moved = bytearray(pattern())
            moved[dst:dst + size * count] = bytes(moved[src:src + size * count])
            if bytes(moved) == want_memory:
                raise AssertionError(f"{name} forward-overlap case with count {count} does not discriminate memmove")
            smeared += 1
        checked += 1
    print(f"{checked} string move/store cases ({smeared} forward-overlap) matched native and translated DLLs "
          "in memory, RCX, RSI and RDI")


def expect_rejection(code, prefix, what):
    try:
        lower(list(Decoder(64, bytes.fromhex(code), ip=0x1000)), [], {}, {}, {})
    except ValueError as exc:
        if not str(exc).startswith(prefix):
            raise AssertionError(f"{what}: unexpected diagnostic {exc}") from exc
    else:
        raise AssertionError(f"accepted {what}")


def check_rejections():
    expect_rejection("f2 a4 c3", "unsupported REPNE string move at 0x1000", "repne movsb")
    expect_rejection("67 f3 a4 c3", "unsupported 32-bit string addressing at 0x1000", "rep movsb with 32-bit addresses")
    expect_rejection("67 aa c3", "unsupported 32-bit string addressing at 0x1000", "stosb with 32-bit addresses")
    expect_rejection("64 a4 c3", "unsupported instruction prefix at 0x1000", "movsb with an FS source")
    # Backward strings are not lowered; neither are the compare and load string forms.
    expect_rejection("fd c3", "unsupported query operation at 0x1000", "std")
    expect_rejection("f3 a6 c3", "unsupported query operation at 0x1000", "repe cmpsb")
    expect_rejection("ac c3", "unsupported query operation at 0x1000", "lodsb")
    expect_rejection("f2 ae c3", "unsupported query operation at 0x1000", "repne scasb")
    # The SSE scalar move shares the MOVSD mnemonic but has register operands.
    expect_rejection("f2 0f 10 c1 c3", "unsupported query operation at 0x1000", "movsd xmm0, xmm1")
    print("REPNE, 32-bit addressing, FS, STD, CMPS, LODS, SCAS and SSE MOVSD were rejected")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(native, other)
        check_rejections()
    else:
        raise ValueError(mode)

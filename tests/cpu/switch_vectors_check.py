"""Native/AOT/reference comparisons for vector data moves in the control-flow translator.

mutants IMAGE OUTPUT    write three incorrect translations that the comparison must detect
check GUARD NATIVE LIFTED OPTIMIZED MUTANTS
                        refusal controls, differential comparison, fault comparison and mutation controls
"""
import ctypes as C
import os
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Decoder, Formatter, FormatterSyntax, Mnemonic as M

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva
from cpu_switch_aot import Lifter, PRELUDE, translate

WORD = C.c_uint64
EXPORTS = ("copy_legacy", "copy_aligned", "copy_vex", "copy_vex_aligned", "upper_halves", "zeroing",
           "frame_moves", "save_xmm6", "loop_blocks", "rip_constants", "anchored_vectors")
REFUSALS = {
    "bad_undefined": "undefined register ymm0",
    "bad_after_call": "undefined register ymm0",
    "bad_upper_after_call": "undefined register ymm6",
    "bad_vzeroupper": "undefined register ymm0",
    "bad_legacy_upper": "undefined register ymm0",
    "bad_unrestored": "nonvolatile register xmm6 is not restored",
    "bad_caller_xmm6": "nonvolatile register xmm6 may hold the caller's value",
    "bad_saved_as_data": "hold the saved xmm6, read as data",
    "bad_frame_misaligned": "misaligned 16-byte aligned stack access",
    "bad_frame_aligned32": "32-byte aligned stack access",
    "bad_anchored_misaligned": "misaligned 16-byte aligned stack access at entry RSP-0x20",
    "bad_rip_misaligned": "misaligned 16-byte aligned access to RVA",
    "bad_xor_logic": "unsupported instruction",
    "bad_integer_transfer": "unsupported register",
}
PATTERN16 = struct.pack("<2Q", 0x0123456789abcdef, 0xfedcba9876543210)
PATTERN32 = struct.pack("<4Q", 0x1111111111111111, 0x2222222222222222, 0x3333333333333333, 0x4444444444444444)
SIZE = 192  # destination and source bytes each export may touch
GUARD = 32  # bytes compared on either side of the destination
FILL = 0xa5
ACCESS_VIOLATION = 0xC0000005
UNBOUND = 0xE0494D47
ZERO16 = bytes(16)


def reference(name, src, dst, count):
    """The destination after the export and its return value, from the source bytes alone."""
    d = bytearray(dst)
    if name == "copy_legacy":
        d[16:32], d[32:48], d[0:16] = src[0:16], src[16:32], src[32:48]
        return bytes(d), 48
    if name == "copy_aligned":
        d[0:16], d[16:32], d[32:48] = src[32:48], src[0:16], src[16:32]
        return bytes(d), 1
    if name == "copy_vex":
        d[0:32], d[32:64], d[64:80] = src[48:80], src[0:32], src[32:48]
        return bytes(d), 80
    if name == "copy_vex_aligned":
        d[0:16], d[32:64] = src[0:16], src[32:64]
        return bytes(d), 2
    if name == "upper_halves":
        # Legacy MOVUPS keeps bytes 16-31; VEX VMOVUPS xmm clears them; legacy MOVAPS xmm, xmm keeps them.
        d[0:32] = src[32:48] + src[16:32]
        d[32:64] = src[48:64] + ZERO16
        d[64:96] = src[48:64] + src[16:32]
        return bytes(d), 0
    if name == "zeroing":
        d[0:32] = ZERO16 + src[16:32]
        d[32:64] = ZERO16 + src[16:32]
        d[64:96] = bytes(32)
        d[96:128] = bytes(32)
        d[128:160] = bytes(32)
        d[160:176] = ZERO16
        return bytes(d), 7
    if name == "frame_moves":
        d[0:48] = src[0:48]
        return bytes(d), 3
    if name == "save_xmm6":
        d[0:32] = src[0:32]
        return bytes(d), 4
    if name == "loop_blocks":
        d[0:16 * count] = src[0:16 * count]
        return bytes(d), count
    if name == "rip_constants":
        d[0:16], d[16:48], d[48:64] = PATTERN16, PATTERN32, PATTERN16
        return bytes(d), 5
    if name == "anchored_vectors":
        d[0:16] = src[0:16]
        return bytes(d), 9
    raise KeyError(name)


class Buffer:
    """SIZE bytes at a chosen offset from a 64-byte boundary, with GUARD bytes compared on each side."""

    def __init__(self, offset):
        self.raw = C.create_string_buffer(SIZE + 2 * GUARD + 128)
        base = (C.addressof(self.raw) + GUARD + 63) & ~63
        self.address = base + offset
        self.start = self.address - GUARD

    def fill(self, data):
        C.memset(self.start, FILL, SIZE + 2 * GUARD)
        C.memmove(self.address, data, len(data))

    def read(self):
        return C.string_at(self.start, SIZE + 2 * GUARD)


def bind(path, prefix):
    library = C.CDLL(str(Path(path).resolve()))
    functions = {}
    for name in EXPORTS:
        function = getattr(library, prefix + name)
        function.argtypes, function.restype = [WORD, WORD, WORD], WORD
        functions[name] = function
    return library, functions


def rejection_checks(image):
    with TemporaryDirectory() as tmp:
        for name, expected in REFUSALS.items():
            try:
                translate(image, [name], image_data=True)
            except ValueError as exc:
                assert expected in str(exc), (name, expected, str(exc))
            else:
                raise AssertionError(f"accepted {name}: {expected}")
        # Remove the restore, then the save, of XMM6 in the accepted save_xmm6.
        data = Path(image).read_bytes()
        path = Path(tmp) / "patched.dll"
        with pefile.PE(str(image), fast_load=True) as pe:
            start = export_rva(pe, "save_xmm6")
            moves = [i for i in Decoder(64, bounded_code(pe, start), ip=start) if i.mnemonic == M.VMOVAPS]
            offsets = [(pe.get_offset_from_rva(i.ip), i.len) for i in moves]
        assert len(offsets) == 2, offsets
        for (position, length), expected in zip(reversed(offsets), ("nonvolatile register xmm6 is not restored",
                                                                       "undefined stack byte")):
            altered = bytearray(data)
            altered[position:position + length] = b"\x90" * length
            path.write_bytes(altered)
            try:
                translate(path, ["save_xmm6"])
            except ValueError as exc:
                assert expected in str(exc), (expected, str(exc))
            else:
                raise AssertionError(f"accepted save_xmm6 without its XMM6 move: {expected}")
    # An EVEX form, a segment prefix and a register outside XMM0-XMM15 stay outside this profile.
    for raw, expected in (("62f17c081002", "unsupported vector encoding"), ("640f1002", "unsupported prefix"),
                          ("62e17c081002", "unsupported vector encoding"), ("0f57c1", "unsupported instruction"),
                          ("c5f857c1", "unsupported instruction")):
        instruction = next(iter(Decoder(64, bytes.fromhex(raw), ip=0x1000)))
        try:
            Lifter().statement(instruction)
        except ValueError as exc:
            assert expected in str(exc), (raw, expected, str(exc))
        else:
            raise AssertionError(f"accepted {instruction}")
    print(f"{len(REFUSALS) + 2} whole-PE refusal controls and 5 unsupported forms rejected")


def differential(native_path, *lifted_paths):
    native, native_functions = bind(native_path, "")
    sides = [("native", native_functions)]
    for path in lifted_paths:
        library, functions = bind(path, "lifted_")
        bind_image = library.lifted_bind_image
        bind_image.argtypes, bind_image.restype = [WORD], C.c_int
        code = bind_image(native._handle)
        assert code == 0, f"{path}: lifted_bind_image returned {code}"
        sides.append((Path(path).name, functions))
    generator = random.Random(1)
    sources = [bytes(generator.randrange(256) for _ in range(SIZE)) for _ in range(6)]
    sources += [bytes(range(SIZE)), b"\xff" * SIZE]
    alignments = {"copy_aligned": (0, 16, 48), "copy_vex_aligned": (0, 32)}
    cases = 0
    for name in EXPORTS:
        for offset in alignments.get(name, (0, 1, 8, 15, 16, 31)):
            for source in sources:
                for count in ((0, 1, 3, 11) if name == "loop_blocks" else (0,)):
                    src, dst = Buffer(offset), Buffer((offset * 3) % 32 if name not in alignments else offset)
                    src.fill(source)
                    expected, value = reference(name, source, b"\xa5" * SIZE, count)
                    expected = bytes([FILL]) * GUARD + expected + bytes([FILL]) * GUARD
                    for label, functions in sides:
                        dst.fill(b"\xa5" * SIZE)
                        result = functions[name](dst.address, src.address, count)
                        got = dst.read()
                        assert result == value, (label, name, offset, count, result, value)
                        if got != expected:
                            first = next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b) - GUARD
                            raise AssertionError(f"{label} {name} offset {offset} count {count}: first "
                                                 f"mismatch at destination byte {first}")
                        assert C.string_at(src.address, SIZE) == source, (label, name, "source changed")
                        cases += 1
    print(f"{cases} native, translated and reference comparisons matched over {len(EXPORTS)} exports")


class Pages:
    """Three pages: no access, read/write, no access."""

    def __init__(self):
        kernel = C.WinDLL("kernel32", use_last_error=True)
        kernel.VirtualAlloc.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32, C.c_uint32]
        kernel.VirtualAlloc.restype = C.c_void_p
        kernel.VirtualProtect.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32, C.POINTER(C.c_uint32)]
        kernel.VirtualFree.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32]
        self.kernel = kernel
        self.base = kernel.VirtualAlloc(None, 3 * 4096, 0x3000, 0x04)
        if not self.base:
            raise C.WinError(C.get_last_error())
        old = C.c_uint32()
        for page in (0, 2):
            if not kernel.VirtualProtect(self.base + page * 4096, 4096, 0x01, C.byref(old)):
                raise C.WinError(C.get_last_error())
        self.middle = self.base + 4096

    def close(self):
        if not self.kernel.VirtualFree(self.base, 0, 0x8000):
            raise C.WinError(C.get_last_error())


def faults(guard_path, native_path, lifted_path, optimized_path):
    guard = C.CDLL(str(Path(guard_path).resolve()))
    guarded = guard.guarded_call
    guarded.argtypes = [C.c_void_p, WORD, WORD, C.POINTER(WORD), C.POINTER(WORD)]
    guarded.restype = C.c_uint32
    detailed = guard.guarded_call_detail
    detailed.argtypes = [C.c_void_p, WORD, WORD, C.POINTER(WORD), C.POINTER(WORD), C.POINTER(WORD * 2)]
    detailed.restype = C.c_uint32
    native, _ = bind(native_path, "")
    pages = Pages()
    count = 0
    try:
        libraries = [("native", native, "")]
        for path in (lifted_path, optimized_path):
            library = C.CDLL(str(Path(path).resolve()))
            unbind = library.lifted_unbind_image
            unbind()
            result = WORD()
            code = guarded(getattr(library, "lifted_rip_constants"), Buffer(0).address, Buffer(0).address, None,
                           C.byref(result))
            assert code == UNBOUND, (path, hex(code))
            library.lifted_bind_image.argtypes = [WORD]
            assert library.lifted_bind_image(native._handle) == 0
            libraries.append((Path(path).name, library, "lifted_"))
            count += 1
        page = pages.middle
        page_end = page + 4096
        # (export, destination, source, bytes of the read/write page that may change before the fault)
        scenarios = [
            ("copy_aligned", "aligned", "aligned+8", None),   # misaligned MOVAPS load
            ("copy_aligned", "aligned+8", "aligned", None),   # misaligned MOVAPS store after three loads
            ("copy_vex_aligned", "aligned", "aligned+16", None),  # VMOVDQA ymm load 16 mod 32
            ("copy_vex_aligned", "aligned+16", "aligned", 16),    # the XMM store succeeds, the YMM store faults
            ("copy_legacy", "buffer", page_end - 40, None),       # the third load reaches the last page
            ("copy_legacy", page_end - 24, "buffer", None),       # the first store straddles into the last page
            ("copy_vex", page - 8, "buffer", None),               # a store that starts in the first page
            ("copy_vex", "buffer", page_end - 16, None),          # a 32-byte load that straddles
            ("copy_vex", page_end - 24, "buffer", None),          # a 32-byte store that straddles the end
            ("copy_legacy", page - 4, "buffer", None),            # a 16-byte store that straddles the start
        ]
        outcomes = {}
        for index, (name, destination, source, _) in enumerate(scenarios):
            for label, library, prefix in libraries:
                aligned_dst, aligned_src = Buffer(0), Buffer(0)
                aligned_src.fill(bytes(range(SIZE)))
                C.memset(page, 0x6c, 4096)

                def place(value, buffer):
                    if value == "buffer":
                        return buffer.address
                    if isinstance(value, str):
                        return buffer.address + int(value.partition("+")[2] or 0)
                    return value
                dst, src = place(destination, aligned_dst), place(source, aligned_src)
                aligned_dst.fill(b"\xa5" * SIZE)
                result = WORD(0x5a5a)
                detail = (WORD * 2)()
                code = detailed(getattr(library, prefix + name), dst, src, None, C.byref(result), C.byref(detail))
                # The access type (0 read, 1 write) and inaccessible address must match the native fault too.
                observed = (code, result.value, aligned_dst.read(), C.string_at(page, 4096), tuple(detail))
                assert code == ACCESS_VIOLATION, (label, name, index, hex(code))
                assert outcomes.setdefault(index, observed) == observed, (label, name, index, "differs from native")
                count += 1
        # Every destination keeps its contents except the 16 bytes that VMOVAPD stored at destination + 16 before
        # VMOVDQA faulted, and the page keeps its contents in every case that reaches it.
        untouched = bytes([FILL]) * (SIZE + 2 * GUARD)
        partial = bytes([FILL]) * (GUARD + 16) + bytes(range(16)) + bytes([FILL]) * (SIZE - 32 + GUARD)
        for index, outcome in outcomes.items():
            assert outcome[2] == (partial if index == 3 else untouched), (scenarios[index][0], index)
        writes = {5, 6, 8, 9}  # page faults of stores; a misaligned aligned move (#GP) reports a read of ~0
        for index, outcome in outcomes.items():
            assert outcome[4][0] == (1 if index in writes else 0), (scenarios[index][0], index, outcome[4])
        # copy_legacy at the page start (9) completes two other stores in the page before the faulting one.
        for index in (4, 5, 6, 7, 8):
            assert outcomes[index][3] == b"\x6c" * 4096, (scenarios[index][0], "the page changed before the fault")
    finally:
        pages.close()
    print(f"{count} access violations and unbound-image exceptions matched, including /O2")
    return libraries


MUTANTS = (
    ("upper_halves", "bad_upper", "{ const word a = (rdx + 0x20ull); load_vector<2>(ymm0, a); }",
     "{ const word a = (rdx + 0x20ull); load_vector<2>(ymm0, a); ymm0.q[2] = 0; ymm0.q[3] = 0; }"),
    ("copy_aligned", "bad_alignment", " vector_alignment(a, 15ull);", ""),
    ("copy_legacy", "bad_store", "store_vector<2>(a, ymm3);",
     "for (int k = 0; k < 2; ++k) *reinterpret_cast<volatile std::uint64_t __unaligned*>(a + 8ull * k) = ymm3.q[k];"),
)


def emit_mutants(image, output):
    """Three incorrect translations: a legacy write that clears bytes 16-31, an aligned move without its
    alignment check and a store split into two quadword stores."""
    bodies = []
    for name, export, old, new in MUTANTS:
        source, _ = translate(image, [name])
        assert source.startswith(PRELUDE) and old in source, (name, old)
        bodies.append(source[len(PRELUDE):].replace("lifted_" + name, export).replace(old, new))
    Path(output).write_text(PRELUDE + "\n".join(bodies), encoding="utf-8")


def mutation_controls(guard_path, path):
    guarded = C.CDLL(str(Path(guard_path).resolve())).guarded_call
    guarded.argtypes = [C.c_void_p, WORD, WORD, C.POINTER(WORD), C.POINTER(WORD)]
    guarded.restype = C.c_uint32
    library = C.CDLL(str(Path(path).resolve()))
    source = bytes(range(SIZE))
    src, dst = Buffer(0), Buffer(0)
    src.fill(source)
    dst.fill(b"\xa5" * SIZE)
    library.bad_upper.argtypes, library.bad_upper.restype = [WORD, WORD, WORD], WORD
    library.bad_upper(dst.address, src.address, 0)
    expected, _ = reference("upper_halves", source, b"\xa5" * SIZE, 0)
    assert dst.read()[GUARD:GUARD + SIZE] != expected, "bad_upper matched the reference"
    result = WORD()
    code = guarded(library.bad_alignment, dst.address, src.address + 8, None, C.byref(result))
    assert code != ACCESS_VIOLATION, "bad_alignment faulted on a misaligned operand"
    pages = Pages()
    try:
        C.memset(pages.middle, 0x6c, 4096)
        code = guarded(library.bad_store, pages.middle + 4096 - 24, src.address, None, C.byref(result))
        assert code == ACCESS_VIOLATION and C.string_at(pages.middle, 4096) != b"\x6c" * 4096, \
            "bad_store faulted without a partial store"
    finally:
        pages.close()
    print("3 compiled /O2 semantic mutations detected: cleared upper half, unchecked alignment, partial store")


def record_forms(native, record):
    _, functions = translate(native, EXPORTS, image_data=True)
    formatter = Formatter(FormatterSyntax.NASM)
    forms = {formatter.format(ins).split()[0] for function in functions for ins in function.instructions.values()}
    destination = Path(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(form + "\n" for form in sorted(forms)), encoding="utf-8")


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "mutants":
        emit_mutants(*args)
    elif mode == "check":
        guard, native, lifted, optimized, mutants = args
        record = os.environ.get("FORM_COVERAGE_RECORD")
        if record:
            Path(record).unlink(missing_ok=True)
        rejection_checks(native)
        differential(native, lifted, optimized)
        faults(guard, native, lifted, optimized)
        mutation_controls(guard, mutants)
        if record:
            # Decode the functions actually compared, not the refusal controls in the same image.
            record_forms(native, record)
    else:
        raise ValueError("expected mutants or check")

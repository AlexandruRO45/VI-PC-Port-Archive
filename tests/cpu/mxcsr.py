"""Compare native and translated LDMXCSR, STMXCSR, VLDMXCSR and VSTMXCSR in the opt-in host-MXCSR profile.

The translation runs through mxcsr_entry, which saves the caller's MXCSR and restores it on return, after a
refusal and when a guest memory fault leaves the lowered body. Every rounding mode is round-tripped and changes
an ADDPS halfway case and CVTSS2SI conversions whose operands the test writes at run time. Words with a
reserved bit, an unmasked exception, FTZ or DAZ are refused before the host register changes.

The VEX forms (VEX.128, L = 0, vvvv = 1111) run the same rules. The translation of every export, VEX and mixed
legacy/VEX ones included, needs no AVX: it is checked against fixed expectations and against the native legacy
export. The native VEX exports are compared in the separate check-avx mode, which exits 77 on a host without AVX.
Four exports hold a malformed encoding (VEX.L = 1, vvvv = 1110) that is only decoded and lowered, never run.
"""
import ctypes
from pathlib import Path
import struct
import subprocess
import sys
import threading

import pefile
from iced_x86 import Decoder, FlowControl, Mnemonic as M

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges, image_bytes
from ffx_version_query import INTEGER_HEADER, MXCSR_HEADER, MXCSR_STATUS, Mxcsr, lower, mxcsr_entry, region

U64 = ctypes.c_uint64
LEGACY_EXPORTS = ("mxcsr_round", "mxcsr_leak", "mxcsr_after", "mxcsr_badload", "mxcsr_fault")
VEX_EXPORTS = ("mxcsr_vex_round", "mxcsr_vex_leak", "mxcsr_vex_after", "mxcsr_vex_badload")
MIXED_EXPORTS = ("mxcsr_mix_lv", "mxcsr_mix_vl")
EXPORTS = LEGACY_EXPORTS + VEX_EXPORTS + MIXED_EXPORTS
ROUND_EXPORTS = ("mxcsr_round", "mxcsr_vex_round", *MIXED_EXPORTS)
# Malformed VEX encodings: export -> (mnemonic of the valid twin, the one byte that is wrong). The valid payload
# byte is 0xF8 (R = 1, vvvv = 1111, L = 0, pp = 00).
MALFORMED = {
    "mxcsr_bad_vld_l1": (M.VLDMXCSR, 0xFC),
    "mxcsr_bad_vld_vvvv": (M.VLDMXCSR, 0xF0),
    "mxcsr_bad_vst_l1": (M.VSTMXCSR, 0xFC),
    "mxcsr_bad_vst_vvvv": (M.VSTMXCSR, 0xF0),
}
MXCSR_OPS = (M.LDMXCSR, M.STMXCSR, M.VLDMXCSR, M.VSTMXCSR)
# LDMXCSR, STMXCSR and the VEX forms in each export of mxcsr.asm.
SOURCE_READS = {"mxcsr_round": 4, "mxcsr_leak": 1, "mxcsr_after": 1, "mxcsr_badload": 1, "mxcsr_fault": 1,
                "mxcsr_vex_round": 5, "mxcsr_vex_leak": 1, "mxcsr_vex_after": 1, "mxcsr_vex_badload": 1,
                "mxcsr_mix_lv": 5, "mxcsr_mix_vl": 5}
REJECT = "{{ *mxcsr_status = {status}; return 0; }}"
DEFAULT = 0x1F80  # all six exceptions masked, round to nearest-even, FTZ and DAZ clear
CALLER = 0x7FA0   # a non-default caller word: round toward zero and the sticky precision flag
FILL = 0xA5
STATE = 0x100
GUARDS = [(0x04, 4), (0x0C, 4), (0x64, 4), (0x6C, 4), (0x74, 4), (0x80, 0x80)]


def f32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


ONE_ULP = 2.0 ** -23
# ADDPS lane 0: 1 + 0.75 ulp, lane 1: -(1 + 0.75 ulp). Lanes 2 and 3 are halfway cases: 1 + 0.5 ulp, between
# an even and an odd neighbour, and (1 + 1 ulp) + 0.5 ulp, between an odd and an even one, so to-nearest-even
# rounds them in different directions. CVTSS2SI of 2.75 and -2.75.
ADD_A = (1.0, -1.0, 1.0, 1 + ONE_ULP)
ADD_B = (3 * 2.0 ** -25, -3 * 2.0 ** -25, 2.0 ** -24, 2.0 ** -24)
CVT_C, CVT_D = 2.75, -2.75
# Rounding control -> (four lane bits, CVTSS2SI(2.75), CVTSS2SI(-2.75)).
EXPECT = {
    0: (f32(1 + ONE_ULP), f32(-(1 + ONE_ULP)), f32(1.0), f32(1 + 2 * ONE_ULP), 3, -3),  # to nearest-even
    1: (f32(1.0), f32(-(1 + ONE_ULP)), f32(1.0), f32(1 + ONE_ULP), 2, -3),             # toward negative infinity
    2: (f32(1 + ONE_ULP), f32(-1.0), f32(1 + ONE_ULP), f32(1 + 2 * ONE_ULP), 3, -2),   # toward positive infinity
    3: (f32(1.0), f32(-1.0), f32(1.0), f32(1 + ONE_ULP), 2, -2),                       # toward zero
}
if len(set(EXPECT.values())) != 4:
    raise AssertionError("the probes do not tell the four rounding modes apart")
REFUSED = {0x11F80: "reserved", 0x1D80: "unmasked", 0x9F80: "unsupported", 0x1FC0: "unsupported"}
# A control is (literal, replacement, exports, occurrence): the replacement is made in the named exports only, in
# every occurrence of the literal or, with an index, in that one. Occurrences of the host load and the host read
# follow the instruction order of mxcsr.asm.
CONTROLS = {
    # LDMXCSR checks the word but never loads it.
    "noload": ("_mm_setcsr(want);", "(void)want;", None, None),
    # STMXCSR stores the default word instead of the host's.
    "fixed": (", _mm_getcsr());", ", 0x1f80u);", None, None),
    # The same two breaks in the all-VEX rounding export only.
    "vexnoload": ("_mm_setcsr(want);", "(void)want;", ("mxcsr_vex_round",), None),
    "vexfixed": (", _mm_getcsr());", ", 0x1f80u);", ("mxcsr_vex_round",), None),
    # VLDMXCSR of mxcsr_vex_after accepts every word with an unmasked exception: only the word 0x1D80 loads.
    "vexunmasked": ("(want & 0x1f80u) != 0x1f80u", "(want & 0x1f80u) != 0x1f80u && want == 0u",
                    ("mxcsr_vex_after",), None),
    # One VEX instruction of a mixed export breaks: its load, its read-back, its entry store, its restore.
    "mixvldload": ("_mm_setcsr(want);", "(void)want;", ("mxcsr_mix_lv",), 0),
    "mixvstread": (", _mm_getcsr());", ", 0x1f80u);", ("mxcsr_mix_lv",), 1),
    "mixvstentry": (", _mm_getcsr());", ", 0x1f80u);", ("mxcsr_mix_vl",), 0),
    "mixvldrestore": ("_mm_setcsr(want);", "(void)want;", ("mxcsr_mix_vl",), 1),
}
# The export a control must break, named in the violation; None for the controls that break the legacy exports first.
CONTROL_EXPORT = {"noload": None, "fixed": None, "vexnoload": "mxcsr_vex_round", "vexfixed": "mxcsr_vex_round",
                  "vexunmasked": "mxcsr_vex_after", "mixvldload": "mxcsr_mix_lv", "mixvstread": "mxcsr_mix_lv",
                  "mixvstentry": "mxcsr_mix_vl", "mixvldrestore": "mxcsr_mix_vl"}


def emit(image, output, variant="lifted"):
    parts = {}
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            code = lower(region(pe, *bounds), ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {},
                         host_mxcsr=Mxcsr(REJECT))
            parts[name] = (f"static __declspec(noinline) word {name}_body(word a, word b, word c, word state, "
                           f"int* mxcsr_status) {{\n{code}\n}}\n")
    count = 0
    if variant != "lifted":
        literal, replacement, scope, nth = CONTROLS[variant]
        for name in (scope or EXPORTS):
            text = parts[name]
            found = text.count(literal)
            if nth is not None:
                if found <= nth:
                    raise AssertionError(f"variant {variant}: occurrence {nth} of {literal!r} missing in {name}")
                cut = -1
                for _ in range(nth + 1):
                    cut = text.index(literal, cut + 1)
                parts[name] = text[:cut] + replacement + text[cut + len(literal):]
                count += 1
            else:
                parts[name] = text.replace(literal, replacement)
                count += found
        if not count:
            raise AssertionError(f"variant {variant} changed nothing")
        print(f"variant {variant}: {count} replacements")
    text = MXCSR_HEADER + INTEGER_HEADER + "".join(parts[n] + mxcsr_entry(n, f"{n}_body") for n in EXPORTS)
    output.write_text(text, encoding="ascii")


def rejected_by_default(image):
    """The default profile refuses every LDMXCSR and STMXCSR: each export is lowered with every earlier MXCSR
    instruction replaced by a no-op, so each one is the refused instruction once."""
    refused = []
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            sites = [i for i in body if i.mnemonic in MXCSR_OPS]
            for k, site in enumerate(sites):
                skip = {earlier.ip: ";" for earlier in sites[:k]}
                try:
                    lower(body, ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {}, host_ops=skip)
                except ValueError as exc:
                    if "unsupported query operation" not in str(exc) or f"{site.ip:#x}" not in str(exc):
                        raise AssertionError(f"{name}: expected the refusal of {site} at {site.ip:#x}, got {exc}")
                    refused.append(MXCSR_OPS_NAMES[site.mnemonic])
                else:
                    raise AssertionError(f"{name}: the default profile lowered {site}")
    return refused


MXCSR_OPS_NAMES = {M.LDMXCSR: "LDMXCSR", M.STMXCSR: "STMXCSR", M.VLDMXCSR: "VLDMXCSR", M.VSTMXCSR: "VSTMXCSR"}


def malformed_vex(image):
    """A VEX.128 VLDMXCSR or VSTMXCSR whose L bit or vvvv is wrong is refused, and the refusal is that one byte:
    the same function with the payload byte set to the valid 0xF8 decodes and lowers in the host-MXCSR profile.
    Returns the number of malformed encodings refused."""
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, (mnemonic, payload) in MALFORMED.items():
            (start, end), = function_ranges(pe, export_rva(pe, name))
            raw = bytes(image_bytes(pe, start, end - start, executable=True))
            if raw[:4] not in (bytes([0xC5, payload, 0xAE, 0x11]), bytes([0xC5, payload, 0xAE, 0x19])):
                raise AssertionError(f"{name}: unexpected bytes {raw[:4].hex()}")
            try:
                region(pe, start, end)
            except ValueError as exc:
                if "instruction boundary" not in str(exc):
                    raise AssertionError(f"{name}: unexpected refusal {exc}")
            else:
                raise AssertionError(f"{name}: the malformed encoding was accepted")
            body = list(Decoder(64, raw, ip=start))
            if not body[0].is_invalid:
                raise AssertionError(f"{name}: the first instruction decodes as {body[0]}")
            try:
                lower(body, ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {}, host_mxcsr=Mxcsr(REJECT))
            except ValueError as exc:
                if f"{start:#x}" not in str(exc):
                    raise AssertionError(f"{name}: refusal does not name {start:#x}: {exc}")
            else:
                raise AssertionError(f"{name}: lowered a malformed encoding")
            twin = bytearray(raw)
            twin[1] = 0xF8
            good = list(Decoder(64, bytes(twin), ip=start))
            if good[0].mnemonic != mnemonic or any(i.is_invalid for i in good):
                raise AssertionError(f"{name}: the twin decodes as {good[0]}, expected {mnemonic}")
            code = lower(good, ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {}, host_mxcsr=Mxcsr(REJECT))
            marker = "_mm_setcsr(want)" if mnemonic == M.VLDMXCSR else "_mm_getcsr()"
            if marker not in code:
                raise AssertionError(f"{name}: the valid twin did not lower to {marker}")
    return len(MALFORMED)


def load(path, lifted):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, U64, ctypes.c_void_p], U64
    return lib


def status(lib, name):
    getter = getattr(lib, f"{name}_status")
    getter.argtypes, getter.restype = [], ctypes.c_int
    return getter()


def fresh_state():
    state = (ctypes.c_ubyte * STATE)(*([FILL] * STATE))
    struct.pack_into("<4f", state, 0x10, *ADD_A)
    struct.pack_into("<4f", state, 0x20, *ADD_B)
    struct.pack_into("<4f", state, 0x30, CVT_C, 0, 0, 0)
    struct.pack_into("<4f", state, 0x40, CVT_D, 0, 0, 0)
    return state


def guards_intact(state):
    return all(all(b == FILL for b in bytes(state[o:o + n])) for o, n in GUARDS)


class Host:
    def __init__(self, harness):
        self.lib = ctypes.CDLL(str(Path(harness).resolve()))
        self.lib.get_mxcsr.argtypes, self.lib.get_mxcsr.restype = [], ctypes.c_uint32
        self.lib.set_mxcsr.argtypes, self.lib.set_mxcsr.restype = [ctypes.c_uint32], None

    def get(self):
        return self.lib.get_mxcsr()

    def set(self, word):
        self.lib.set_mxcsr(word)


def round_case(host, lib, rc, sticky, name="mxcsr_round"):
    """(outputs, problem) of a rounding export with rounding control rc and the requested sticky flags. The
    exports other than mxcsr_round also read MXCSR after their final restore, which must be the caller's word."""
    word = DEFAULT | rc << 13 | sticky
    state = fresh_state()
    host.set(CALLER)
    result = getattr(lib, name)(0, word, 0, ctypes.addressof(state))
    after = host.get()
    lanes = struct.unpack_from("<4I", state, 0x50)
    cvt = struct.unpack_from("<i", state, 0x60)[0], struct.unpack_from("<i", state, 0x68)[0]
    readback = struct.unpack_from("<I", state, 0x70)[0]
    outputs = (*lanes, *cvt, readback)
    if result != 7:
        return outputs, f"returned {result:#x}"
    if (*lanes, *cvt) != EXPECT[rc]:
        return outputs, f"rounding control {rc}: lanes/conversions {lanes}, {cvt}, expected {EXPECT[rc]}"
    if readback & ~0x3F != word & ~0x3F or readback & sticky != sticky:
        return outputs, (f"STMXCSR read back {readback:#x} after loading {word:#x}: control bits and the "
                         "requested sticky flags must be kept")
    if struct.unpack_from("<I", state, 0x00)[0] != CALLER:
        return outputs, f"entry STMXCSR stored {struct.unpack_from('<I', state, 0)[0]:#x}, caller {CALLER:#x}"
    if not guards_intact(state):
        return outputs, "a guard byte next to a stored word changed"
    if name != "mxcsr_round" and struct.unpack_from("<I", state, 0x78)[0] != CALLER:
        return outputs, (f"the word read after the final restore is {struct.unpack_from('<I', state, 0x78)[0]:#x}, "
                         f"expected {CALLER:#x}")
    if after != CALLER:
        return outputs, f"caller MXCSR {after:#x} after the call, expected {CALLER:#x}"
    return outputs, None


def rounds_violation(host, native, lib, label, names):
    """The first rounding export of names whose translation differs from fixed expectations or from the native
    legacy mxcsr_round, whose reference is independent of the VEX encodings."""
    for name in names:
        for rc in range(4):
            for sticky in (0, 0x3F):
                got, problem = round_case(host, lib, rc, sticky, name)
                if problem:
                    return f"{label} {name}: {problem}"
                expected, problem = round_case(host, native, rc, sticky)
                if problem:
                    raise AssertionError(f"native mxcsr_round: {problem}")
                if got != expected:
                    return f"{label} {name} rc {rc} sticky {sticky:#x}: {got} != native {expected}"
    return None


def boundary_violation(host, lib, label, leak, after):
    """The first broken boundary rule of the exports leak and after, or None."""
    # The boundary restores a word the guest leaves loaded.
    state = fresh_state()
    host.set(CALLER)
    result = getattr(lib, leak)(0, DEFAULT | 1 << 13, 0, ctypes.addressof(state))
    if result != 7 or status(lib, leak) or host.get() != CALLER:
        return f"{label} {leak}: returned {result:#x}, status {status(lib, leak)}, MXCSR {host.get():#x}"
    # A valid word lets the guest continue; a refused one stops it with its status and loads nothing.
    for word, reason in ((DEFAULT | 2 << 13, None), *REFUSED.items()):
        state = fresh_state()
        host.set(CALLER)
        result = getattr(lib, after)(0, word, 0, ctypes.addressof(state))
        marker = struct.unpack_from("<Q", state, 0x78)[0]
        got = (result, status(lib, after), marker, host.get())
        want = (7, 0, 1, CALLER) if reason is None else (0, MXCSR_STATUS[reason], int.from_bytes([FILL] * 8, "little"), CALLER)
        if got != want:
            return f"{label} {after}({word:#x}): (return, status, marker, MXCSR) {got}, expected {want}"
    return None


def vex_violation(host, native, lib, label):
    """The first broken rule of the VEX and mixed exports of a translated library, or None. Runs without AVX."""
    problem = rounds_violation(host, native, lib, label, ROUND_EXPORTS[1:])
    return problem or boundary_violation(host, lib, label, "mxcsr_vex_leak", "mxcsr_vex_after") or vex_fault(host, lib, label)


def vex_fault(host, lib, label):
    """A guest VLDMXCSR on an inaccessible page faults, leaves no refusal status and restores the caller's word."""
    kernel32, page = alloc_noaccess()
    try:
        state = fresh_state()
        host.set(CALLER)
        try:
            lib.mxcsr_vex_badload(page, 0, 0, ctypes.addressof(state))
        except OSError as exc:
            fault = str(exc)
        else:
            return f"{label} mxcsr_vex_badload returned from an inaccessible load"
        if host.get() != CALLER or status(lib, "mxcsr_vex_badload") != 0:
            return (f"{label} mxcsr_vex_badload: MXCSR {host.get():#x}, status {status(lib, 'mxcsr_vex_badload')} "
                    f"after the fault ({fault})")
    finally:
        free_page(kernel32, page)
    return None


def alloc_noaccess():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    page = kernel32.VirtualAlloc(None, 4096, 0x3000, 0x01)  # MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS
    if not page:
        raise OSError(ctypes.get_last_error(), "VirtualAlloc")
    return kernel32, page


def free_page(kernel32, page):
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    kernel32.VirtualFree(ctypes.c_void_p(page), 0, 0x8000)


def lifted_violation(host, native, lib, label):
    """The first broken rule of a translated library, or None."""
    problem = rounds_violation(host, native, lib, label, ROUND_EXPORTS[:1])
    if problem:
        return problem
    problem = boundary_violation(host, lib, label, "mxcsr_leak", "mxcsr_after")
    if problem:
        return problem
    # The status belongs to the calling thread: a successful call on another thread leaves this thread's
    # refusal status in place.
    state = fresh_state()
    lib.mxcsr_after(0, 0x11F80, 0, ctypes.addressof(state))
    other = []
    worker = threading.Thread(target=lambda: other.append(
        (lib.mxcsr_after(0, DEFAULT, 0, ctypes.addressof(fresh_state())), status(lib, "mxcsr_after"))))
    worker.start()
    worker.join()
    if other != [(7, 0)] or status(lib, "mxcsr_after") != MXCSR_STATUS["reserved"]:
        return f"{label} status per thread: other thread {other}, this thread {status(lib, 'mxcsr_after')}"
    host.set(CALLER)
    # A guest memory fault: once at the LDMXCSR operand itself, once after a valid load. The boundary restores
    # the caller's word both times.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    page = kernel32.VirtualAlloc(None, 4096, 0x3000, 0x01)  # MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS
    if not page:
        raise OSError(ctypes.get_last_error(), "VirtualAlloc")
    try:
        state = fresh_state()
        lib.mxcsr_fault(page, 0x11F80, 0, ctypes.addressof(state))
        if status(lib, "mxcsr_fault") != MXCSR_STATUS["reserved"]:
            return f"{label} mxcsr_fault did not record the reserved-bit refusal before the fault control"
        for name in ("mxcsr_badload", "mxcsr_fault"):
            state = fresh_state()
            host.set(CALLER)
            try:
                getattr(lib, name)(page, DEFAULT | 1 << 13, 0, ctypes.addressof(state))
            except OSError as exc:
                fault = str(exc)
            else:
                return f"{label} {name} returned from an inaccessible load"
            if host.get() != CALLER:
                return f"{label} {name}: MXCSR {host.get():#x} after the fault ({fault}), expected {CALLER:#x}"
            if status(lib, name) != 0:
                return f"{label} {name}: stale refusal status after the fault ({fault})"
    finally:
        free_page(kernel32, page)
    return None


def violation(host, native, lib, label):
    """The first broken rule of a translated library, legacy exports first, then the VEX and mixed ones."""
    return lifted_violation(host, native, lib, label) or vex_violation(host, native, lib, label)


CHILD = """
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
fn = getattr(lib, sys.argv[3])
fn.argtypes, fn.restype = [ctypes.c_uint64] * 3 + [ctypes.c_void_p], ctypes.c_uint64
state = (ctypes.c_ubyte * 256)()
print("calling", flush=True)
try:
    fn(0, int(sys.argv[2]), 0, ctypes.addressof(state))
except OSError as exc:
    print("fault:", exc, flush=True)
else:
    print("returned", flush=True)
"""


def native_reserved(path, export="mxcsr_after"):
    """Reference observation: what a native load of a word with a reserved bit does, in a child process."""
    child = subprocess.run([sys.executable, "-I", "-c", CHILD, str(Path(path).resolve()), str(0x11F80), export],
                           capture_output=True, text=True, timeout=60)
    return child.returncode, child.stdout.strip().replace("\n", " | "), child.stderr.strip()


def reads_in(path):
    """LDMXCSR and STMXCSR reachable from each export of an /O2 DLL, following every branch."""
    with pefile.PE(str(path), fast_load=True) as pe:
        base, image = pe.OPTIONAL_HEADER.ImageBase, pe.get_memory_mapped_image()
        found = {}
        for name in EXPORTS:
            pending, seen, count = [base + export_rva(pe, name)], set(), 0
            while pending:
                ip = pending.pop()
                while ip not in seen:
                    seen.add(ip)
                    if not base <= ip < base + len(image):
                        break
                    instruction = Decoder(64, image[ip - base:ip - base + 15], ip=ip).decode()
                    count += instruction.mnemonic in MXCSR_OPS
                    flow = instruction.flow_control
                    if flow == FlowControl.RETURN:
                        break
                    if flow in (FlowControl.CONDITIONAL_BRANCH, FlowControl.UNCONDITIONAL_BRANCH):
                        pending.append(instruction.near_branch_target)
                        if flow == FlowControl.UNCONDITIONAL_BRANCH:
                            break
                    elif flow == FlowControl.CALL and instruction.near_branch_target:
                        pending.append(instruction.near_branch_target)
                    elif flow not in (FlowControl.NEXT, FlowControl.CALL, FlowControl.INDIRECT_CALL):
                        break
                    ip = instruction.next_ip
            found[name] = count
    return found


def check(harness_path, native_path, lifted_path, *extra_paths):
    host = Host(harness_path)
    original = host.get()
    try:
        refused = rejected_by_default(native_path)
        print(f"the default profile refuses each of the {len(refused)} MXCSR instructions on its own: "
              + ", ".join(f"{refused.count(n)} {n}" for n in MXCSR_OPS_NAMES.values()))
        print(f"{malformed_vex(native_path)} malformed VEX encodings (L = 1, vvvv = 1110) are refused, and each "
              "lowers once its one wrong byte is the valid 0xF8")
        native, lifted = load(native_path, False), load(lifted_path, True)
        problem = violation(host, native, lifted, "lifted")
        if problem:
            raise AssertionError(problem)
        print("lifted: all four rounding modes round-trip and change ADDPS and CVTSS2SI as natively, with and "
              "without sticky flags; adjacent bytes unchanged; the caller's word "
              f"{CALLER:#x} restored after return, refusal and a guest memory fault; refused words "
              f"{', '.join(f'{w:#x}={r}' for w, r in REFUSED.items())} stopped the guest before its next store; "
              "the VEX and mixed legacy/VEX exports match the same fixed expectations and the native legacy export, "
              "with no AVX needed")
        code, out, err = native_reserved(native_path)
        print(f"native reference, LDMXCSR of {0x11F80:#x} in a child process: exit {code:#x} {out} {err}")
        for path in extra_paths:
            stem = Path(path).stem
            if stem.endswith("_o2"):
                reads = reads_in(path)
                short = {n: c for n, c in reads.items() if c < SOURCE_READS[n]}
                if short:
                    raise AssertionError(f"{stem}: fewer LDMXCSR/STMXCSR than the source in {short}")
                problem = violation(host, native, load(path, True), stem)
                if problem:
                    raise AssertionError(problem)
                print(f"{stem}: the /O2 build keeps every rule, and every export still reaches at least its "
                      f"source's LDMXCSR and STMXCSR: {reads}")
                continue
            variant = stem.rsplit("_", 1)[-1]
            problem = violation(host, native, load(path, True), f"control {variant}")
            print(f"control {variant}: {problem}")
            if problem is None:
                raise AssertionError(f"control {variant} kept every rule")
            export = CONTROL_EXPORT.get(variant)
            if export and f" {export}" not in problem:
                raise AssertionError(f"control {variant} broke a rule, but not in {export}: {problem}")
    finally:
        host.set(original)


def host_has_avx():
    """True where the processor and the operating system support AVX (PF_AVX_INSTRUCTIONS_AVAILABLE)."""
    return bool(ctypes.WinDLL("kernel32").IsProcessorFeaturePresent(39))


def check_avx(harness_path, native_path, lifted_path):
    """Native VEX and mixed exports against their translation. Exits 77 on a host without AVX."""
    if not host_has_avx():
        print("host lacks AVX; skipping the native VEX comparison")
        raise SystemExit(77)
    host = Host(harness_path)
    original = host.get()
    try:
        native, lifted = load(native_path, False), load(lifted_path, True)
        for name in ROUND_EXPORTS[1:]:
            for rc in range(4):
                for sticky in (0, 0x3F):
                    want, problem = round_case(host, native, rc, sticky, name)
                    if problem:
                        raise AssertionError(f"native {name}: {problem}")
                    got, problem = round_case(host, lifted, rc, sticky, name)
                    if problem or got != want:
                        raise AssertionError(f"lifted {name} rc {rc} sticky {sticky:#x}: {problem or got} vs native {want}")
        for word in (DEFAULT | 2 << 13, DEFAULT | 3 << 13 | 0x3F):
            outcomes = []
            for lib in (native, lifted):
                state = fresh_state()
                host.set(CALLER)
                result = lib.mxcsr_vex_after(0, word, 0, ctypes.addressof(state))
                outcomes.append((result, struct.unpack_from("<Q", state, 0x78)[0]))
                host.set(CALLER)
            if outcomes[0] != outcomes[1] or outcomes[0] != (7, 1):
                raise AssertionError(f"mxcsr_vex_after({word:#x}): native {outcomes[0]}, lifted {outcomes[1]}")
        print("native VEX and mixed exports match their translation in all four rounding modes, with and without "
              "sticky flags, and for two accepted words of mxcsr_vex_after")
        code, out, err = native_reserved(native_path, "mxcsr_vex_after")
        print(f"native reference, VLDMXCSR of {0x11F80:#x} in a child process: exit {code:#x} {out} {err}")
    finally:
        host.set(original)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "emit":
        image, output, *variant = args
        emit(Path(image), Path(output), *variant)
    elif mode == "check":
        check(*args)
    elif mode == "check-avx":
        check_avx(*args)
    else:
        raise ValueError(mode)

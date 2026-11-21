"""Differential checks and rejection controls for recompiled x64 jump tables."""

import ctypes
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Decoder, FlowControl, Mnemonic, OpKind

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import translate
from cpu_leaf_aot import export_rva

MASK32 = 0xffffffff
GUARDS = (0x5a5a1234, 0xa5a54321)
SENTINEL = 0x0badf00d
SELECTORS = ("select_rva", "select_rel")


def select_reference(selector, operand, swapped=False):
    selector, value = selector & MASK32, operand & MASK32
    case = 7 - selector if swapped and selector in (1, 6) else selector
    if case == 0:
        stored = value + 11
    elif case == 1:
        stored = value ^ 0x5a5a
    elif case in (2, 5):
        stored = value * 3
    elif case in (3, 4):
        stored = value + (7 if case == 3 else 0) - 2
    elif case == 6:
        stored = value << 4
    else:
        stored = ~value
    stored &= MASK32
    return (stored + selector) & MASK32, stored


def ops_reference(program, swapped=False):
    acc = 0
    for position, op in enumerate(program):
        if op > 4:
            return MASK32, len(program) - position
        if swapped and op in (0, 2):
            op = 2 - op
        acc = (acc + 3, acc * 5, acc ^ 0xa5, acc >> 1, -acc)[op] & MASK32
    return acc, acc


def narrow_reference(selector, operand):
    """shift_narrow: byte or word shift of the stored DWORD; x86 masks the count to five bits."""
    selector, value = selector & MASK32, operand & MASK32
    if selector > 5:
        return value, value
    count = (value >> 16) & 31
    bits = 8 if selector < 3 else 16
    mask = (1 << bits) - 1
    part = value & mask
    kind = selector % 3
    if kind == 0:
        part = (part << count) & mask
    elif kind == 1:
        part >>= count
    else:
        signed = part - (1 << bits) if part >> (bits - 1) else part
        part = (signed >> count) & mask
    stored = (value & ~mask & MASK32) | part
    return stored, stored


def narrow_inputs():
    """Each byte and word shift with counts 0..31 (and count bytes above 31) on sign/edge patterns."""
    lows = (0x0001, 0x0080, 0x00ff, 0x8001, 0x7fff, 0xffff, 0x1234, 0xc3a5)
    counts = [*range(32), 0x21, 0x3f, 0x80, 0xff]
    return [(s | high, (0xab << 24) | (c << 16) | low)
            for s in (*range(7), 0x80000000) for c in counts for low in lows for high in (0, 0xdeadbeef << 32)]


def check_narrow(probe, libraries, label):
    invoke = windows_invoker(probe, libraries)
    from direct_check import SENTINELS
    for x, y in narrow_inputs():
        rax, words, registers = invoke("shift_narrow", x, y)
        want, stored = narrow_reference(x, y)
        if (rax, words) != (want, (GUARDS[0], stored, GUARDS[1])) or registers != SENTINELS:
            raise AssertionError(f"{label} shift_narrow({x:#x}, {y:#x}): {rax:#x} {words} {registers}")

def select_inputs():
    selectors = [*range(11), 0x7fffffff, 0x80000000, *range(MASK32 - 6, MASK32 + 1)]
    operands = [0, 1, 2, 0x5a5a, 0x7fffffff, 0x80000000, MASK32]
    rng = random.Random(41)
    operands += [rng.getrandbits(32) for _ in range(8)]
    # Upper halves are not part of the 32-bit contract and must not reach the table index.
    return [(s | high, v | high) for s in selectors for v in operands for high in (0, 0xdeadbeef << 32)]


def programs():
    rng = random.Random(77)
    cases = [bytes([op]) for op in range(7)] + [bytes([0, 1, 2, 3, 4] * 3), bytes([1] * 40), bytes([4, 4, 4])]
    cases += [bytes(rng.randrange(5) for _ in range(rng.randrange(1, 64))) for _ in range(96)]
    cases += [bytes(rng.randrange(5) for _ in range(rng.randrange(0, 12))) + bytes([rng.randrange(5, 256)])
              + bytes(rng.randrange(256) for _ in range(rng.randrange(0, 8))) for _ in range(32)]
    return cases


def outcomes(invoke, select_names, ops_name):
    """Run every input through invoke(name, x, y) -> (rax, words, registers)."""
    results = {}
    for name in select_names:
        results[name] = [invoke(name, x, y) for x, y in select_inputs()]
    buffers = []
    runs = []
    for program in programs():
        buffer = ctypes.create_string_buffer(program, len(program))
        buffers.append(buffer)
        runs.append(invoke(ops_name, ctypes.addressof(buffer), len(program) | 0xfeed << 48))
    results[ops_name] = runs
    return results


def expected(select_names, ops_name, swapped=False):
    results = {}
    for name in select_names:
        results[name] = [(rax, (GUARDS[0], stored, GUARDS[1]))
                         for rax, stored in (select_reference(x, y, swapped) for x, y in select_inputs())]
    results[ops_name] = [(rax, (GUARDS[0], stored, GUARDS[1]))
                         for rax, stored in (ops_reference(p, swapped) for p in programs())]
    return results


def verify(observed, reference, registers=None, label=""):
    for name, rows in reference.items():
        for number, (want, got) in enumerate(zip(rows, observed[name])):
            if got[:2] != want or (registers is not None and got[2] != registers):
                raise AssertionError(f"{label} {name} input {number}: {got} != {want}")


def windows_invoker(probe, libraries):
    from direct_check import Word

    def invoke(name, x, y):
        words = (ctypes.c_uint32 * 3)(GUARDS[0], SENTINEL, GUARDS[1])
        output = ctypes.cast(ctypes.byref(words, 4), ctypes.POINTER(Word))
        registers = (Word * 2)()
        rax = probe(libraries(name), x, y, output, registers)
        return rax, tuple(words), tuple(registers)
    return invoke


def check(probe_path, original_path, lifted_path, swapped_path):
    from direct_check import SENTINELS, register_probe
    probe = register_probe(probe_path)
    native, lifted, swapped = map(ctypes.CDLL, (original_path, lifted_path, swapped_path))
    names = (*SELECTORS, "run_ops")
    original = outcomes(windows_invoker(probe, lambda n: getattr(native, n)), SELECTORS, "run_ops")
    verify(original, expected(SELECTORS, "run_ops"), SENTINELS, "source fixture")
    translated = outcomes(windows_invoker(probe, lambda n: getattr(lifted, "lifted_" + n)), SELECTORS, "run_ops")
    verify(translated, {n: [r[:2] for r in original[n]] for n in names}, SENTINELS, "translation")
    control = outcomes(windows_invoker(probe, lambda n: getattr(swapped, "lifted_" + n)), SELECTORS, "run_ops")
    verify(control, expected(SELECTORS, "run_ops", swapped=True), SENTINELS, "swapped-table control")
    changed = {n: sum(a != b for a, b in zip(original[n], control[n])) for n in names}
    if not all(changed.values()):
        raise AssertionError(f"swapped-table control matched the original: {changed}")
    check_narrow(probe, lambda n: getattr(native, n), "source fixture")
    check_narrow(probe, lambda n: getattr(lifted, "lifted_" + n), "translation")
    check_narrow(probe, lambda n: getattr(swapped, "lifted_" + n), "swapped-table build")
    print(f"{len(select_inputs())} selector pairs per table layout and {len(programs())} programs match on "
          f"return, guarded store, RBX and RSI; swapped tables changed {changed}; {len(narrow_inputs())} byte and "
          f"word memory shifts match x86 count masking")


def instructions(pe, name):
    entry = export_rva(pe, name)
    found = []
    for ins in Decoder(64, pe.get_data(entry, 0x60), ip=entry):
        found.append(ins)
        if ins.flow_control == FlowControl.INDIRECT_BRANCH:
            return entry, found
    raise AssertionError(f"no table dispatch in {name}")


def check_rejections(original_path):
    data = Path(original_path).read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        entry, code = instructions(pe, "select_rva")
        neighbor = export_rva(pe, "select_rel")
        first = lambda mnemonic: next(i for i in code if i.mnemonic == mnemonic)
        offset = lambda rva: pe.get_offset_from_rva(rva)
        bound, sign_extend, lea = first(Mnemonic.JA), first(Mnemonic.MOVSXD), first(Mnemonic.LEA)
        load = next(i for i in code if i.mnemonic == Mnemonic.MOV and i.op1_kind == OpKind.MEMORY)
        copy = code[0]
        table = load.memory_displacement
        dispatch = code[-1]
        after = Decoder(64, pe.get_data(dispatch.next_ip, 0x10), ip=dispatch.next_ip)
        case0, case0_jump = after.decode(), after.decode()
        invert = next(i for i in Decoder(64, pe.get_data(entry, 0x50), ip=entry) if i.mnemonic == Mnemonic.NOT)
        directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(directory.VirtualAddress, directory.Size)
        unwind = next(directory.VirtualAddress + i for i in range(0, len(records), 12)
                      if struct.unpack_from("<I", records, i)[0] == entry)
        record_end = struct.unpack_from("<I", records, unwind - directory.VirtualAddress + 4)[0]

    def branch(ins, target):
        raw = data[offset(ins.ip):offset(ins.ip) + ins.len]
        if ins.len == 2:
            return raw[:1] + struct.pack("b", target - ins.next_ip)
        return raw[:-4] + struct.pack("<i", target - ins.next_ip)

    def signed_bound(ins):
        raw = bytearray(data[offset(ins.ip):offset(ins.ip) + ins.len])
        raw[-5 if ins.len == 6 else 0] += 8  # JA -> JG in either encoding
        return bytes(raw)

    controls = [
        (bound.ip, b"\x90" * bound.len, "unbounded jump-table index"),
        (bound.ip, signed_bound(bound), "not an unsigned upper limit"),
        (sign_extend.ip, bytes.fromhex("4c 8b c9"), "is not the bounded value"),
        (case0.ip, bytes.fromhex("83 c3 0b"), "nonvolatile register rbx"),
        (copy.ip, bytes.fromhex("8b c0"), "undefined register rax"),
        (invert.ip, bytes.fromhex("0f ca"), "unsupported instruction"),
        (table + 12, struct.pack("<I", case0.ip + 1), "overlaps"),
        (table + 12, struct.pack("<I", neighbor), "outside owned ranges"),
        (load.ip + load.len - 4, struct.pack("<I", entry), "jump table at RVA"),
        (case0_jump.ip, branch(case0_jump, lea.ip), "second entry into the dispatch block"),
        (lea.ip + lea.len - 4, struct.pack("<i", struct.unpack_from("<i", data, offset(lea.next_ip) - 4)[0] + 4),
         "image-relative table without an image base"),
        (lea.ip, bytes.fromhex("49 89 ca") + b"\x90" * (lea.len - 3), "is not a RIP-relative address"),
        # Without its own unwind entry the leaf must not run into another function's interval.
        (unwind, struct.pack("<II", entry + copy.len, record_end), "another function's unwind interval"),
    ]
    with TemporaryDirectory() as folder:
        path = Path(folder) / "control.dll"
        for rva, patch, reason in controls:
            damaged = bytearray(data)
            damaged[offset(rva):offset(rva) + len(patch)] = patch
            path.write_bytes(damaged)
            try:
                translate(path, ["select_rva"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"{reason}: {exc}") from exc
            else:
                raise AssertionError(f"accepted control: {reason}")
        # Moving the unwind entry onto the table data leaves a leaf with no .pdata; the
        # section-bounded walk must produce the same translation.
        damaged = bytearray(data)
        damaged[offset(unwind):offset(unwind) + 8] = struct.pack("<II", table, table + 28)
        path.write_bytes(damaged)
        if translate(path, ["select_rva"])[0] != translate(original_path, ["select_rva"])[0]:
            raise AssertionError("leaf without unwind data translated differently")
    print(f"{len(controls)} rejection controls and the leaf-without-unwind translation passed")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "rejections":
        check_rejections(sys.argv[2])
    else:
        check(*sys.argv[1:])
        check_rejections(sys.argv[2])

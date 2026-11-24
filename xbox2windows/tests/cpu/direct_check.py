"""Differential Windows x64 ABI checks for a recompiled direct call."""

import ctypes
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_call_aot import translate
from cpu_leaf_aot import bounded_code, export_rva

Word = ctypes.c_uint64
MASK = (1 << 64) - 1
SENTINELS = (0x0123456789abcdef, 0xfedcba9876543210)
GUARDS = (0x5a5affff12345678, 0xa5a5000087654321)


def observe(probe, fn, x, y):
    trace = (Word * 5)(GUARDS[0], MASK, MASK, MASK, GUARDS[1])
    regs = (Word * 2)()
    output = ctypes.cast(ctypes.byref(trace, ctypes.sizeof(Word)), ctypes.POINTER(Word))
    result = probe(fn, x, y, output, regs)
    return result, tuple(trace), tuple(regs)


def reference(x, y, *, caller, altered=False):
    a, b = ((x + 7) & MASK, (y - 3) & MASK) if caller else (x, y)
    stored = (a - b + 11) & MASK
    returned = (stored + int(altered)) & MASK
    if caller:
        returned = ((returned + x) & MASK) ^ y
    return returned, (GUARDS[0], a, b, stored, GUARDS[1]), SENTINELS


def register_probe(probe_path):
    harness = ctypes.CDLL(probe_path)
    probe = harness.probe
    probe.argtypes = [ctypes.c_void_p, Word, Word, ctypes.POINTER(Word), ctypes.POINTER(Word)]
    probe.restype = Word
    return probe

def input_pairs():
    edges = (0, 1, 2, 3, 7, (1 << 32) - 1, 1 << 32, (1 << 63) - 1, 1 << 63, MASK - 6, MASK)
    inputs = [(x, y) for x in edges for y in edges]
    rng = random.Random(22)
    inputs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(128))
    return inputs


def check(probe_path, original_path, lifted_path, altered_path):
    probe = register_probe(probe_path)
    if observe(probe, ctypes.CDLL(probe_path).clobber_control, 0, 0)[2] != (0, 0):
        raise AssertionError("physical register probe missed the clobber control")
    native, lifted, altered = map(ctypes.CDLL, (original_path, lifted_path, altered_path))
    inputs = input_pairs()
    for caller, source, compiled, control in (
            (True, native.call_mix, lifted.lifted, altered.lifted),
            (False, native.mix_args, lifted.lifted_callee, altered.lifted_callee)):
        for x, y in inputs:
            original = observe(probe, source, x, y)
            actual = observe(probe, compiled, x, y)
            if original != reference(x, y, caller=caller) or actual != original:
                raise AssertionError(f"{source.__name__}({x:#x}, {y:#x}): {original=} {actual=}")
            changed = observe(probe, control, x, y)
            if changed == original or changed != reference(x, y, caller=caller, altered=True):
                raise AssertionError(f"{source.__name__} incorrect-return control: {changed=}")
    print(f"{len(inputs)} input pairs each for caller and callee: arguments, guarded writes, "
          "RBX, RSI and returns match; all altered returns and the register clobber detected")


def call_site(image, caller_name, callee_name):
    # Cross-check the linked rel32 against the two independently exported RVAs.
    with pefile.PE(image, fast_load=True) as pe:
        caller = export_rva(pe, caller_name)
        callee = export_rva(pe, callee_name)
        body = bounded_code(pe, caller)
        sites = [i for i in range(len(body) - 4) if body[i] == 0xe8
                 and caller + i + 5 + struct.unpack_from("<i", body, i + 1)[0] == callee]
        if len(sites) != 1:
            raise AssertionError(f"{caller_name} to {callee_name}: expected one rel32, found {sites}")
        call, = sites
        file_pos = pe.get_offset_from_rva(caller + call + 1)
    return caller, callee, call, file_pos


def check_invalid_targets(image):
    # Modify the linked rel32, keeping both exported names and intervals intact.
    caller, callee, call, file_pos = call_site(image, "call_mix", "mix_args")
    original = Path(image).read_bytes()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "bad-target.dll"
        for target, reason in ((callee + 1, "no unique function interval"),
                               (caller, "overlapping call target")):
            damaged = bytearray(original)
            struct.pack_into("<i", damaged, file_pos, target - (caller + call + 5))
            path.write_bytes(damaged)
            try:
                translate(path, "call_mix")
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(str(exc)) from exc
            else:
                raise AssertionError(f"accepted invalid call target {target:#x}")


if __name__ == "__main__":
    check(*sys.argv[1:])
    check_invalid_targets(sys.argv[2])
    for path in sys.argv[3:]:
        call_site(path, "lifted", "lifted_callee")

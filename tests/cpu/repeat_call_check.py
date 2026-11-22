"""Differential Windows x64 ABI checks for a caller that repeats one direct call."""

import ctypes
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_call_aot import lift_body, translate
from cpu_leaf_aot import bounded_code, export_rva
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe

MOVE_RSI_R8 = bytes.fromhex("49 8b f0")
MOVE_R8_RSI = bytes.fromhex("4c 8b c6")
# The same moves in the opcode 89 form, which other assemblers emit.
ALTERNATE = {MOVE_RSI_R8: bytes.fromhex("4c 89 c6"), MOVE_R8_RSI: bytes.fromhex("49 89 f0")}


def mix(a, b, altered):
    stored = (a - b + 11) & MASK
    return (stored + int(altered)) & MASK, (a, b, stored)


def reference(x, y, *, caller, altered=False):
    if not caller:
        returned, trace = mix(x, y, altered)
        return returned, (GUARDS[0], *trace, GUARDS[1]), SENTINELS
    first, _ = mix((x + 7) & MASK, (y - 3) & MASK, altered)
    # The second call overwrites the trace, so it records the second arguments.
    second, trace = mix((first + x) & MASK, x, altered)
    return (second + x) & MASK, (GUARDS[0], *trace, GUARDS[1]), SENTINELS


def check(probe_path, original_path, lifted_path, altered_path):
    probe = register_probe(probe_path)
    if observe(probe, ctypes.CDLL(probe_path).clobber_control, 0, 0)[2] != (0, 0):
        raise AssertionError("physical register probe missed the clobber control")
    native, lifted, altered = map(ctypes.CDLL, (original_path, lifted_path, altered_path))
    inputs = input_pairs()
    for caller, source, compiled, control in (
            (True, native.call_twice, lifted.lifted, altered.lifted),
            (False, native.mix_args, lifted.lifted_callee, altered.lifted_callee)):
        for x, y in inputs:
            original = observe(probe, source, x, y)
            actual = observe(probe, compiled, x, y)
            if original != reference(x, y, caller=caller) or actual != original:
                raise AssertionError(f"{source.__name__}({x:#x}, {y:#x}): {original=} {actual=}")
            changed = observe(probe, control, x, y)
            # The first altered return feeds the second call, so the caller's trace changes too.
            same_trace = caller and changed[1] == original[1]
            if (changed[0] == original[0] or same_trace
                    or changed != reference(x, y, caller=caller, altered=True)):
                raise AssertionError(f"{source.__name__} incorrect-return control: {changed=}")
    print(f"{len(inputs)} input pairs each for caller and callee: arguments, guarded writes, "
          "RBX, RSI and returns match; every altered callee changes the caller's return and "
          "trace; the register clobber is detected")


def call_sites(image, caller_name, callee_name):
    # Cross-check each linked rel32 against the two independently exported RVAs.
    with pefile.PE(image, fast_load=True) as pe:
        caller = export_rva(pe, caller_name)
        callee = export_rva(pe, callee_name)
        body = bounded_code(pe, caller)
        sites = [i for i in range(len(body) - 4) if body[i] == 0xe8
                 and caller + i + 5 + struct.unpack_from("<i", body, i + 1)[0] == callee]
        if len(sites) != 2:
            raise AssertionError(f"{caller_name} to {callee_name}: expected two rel32, found {sites}")
        offsets = [pe.get_offset_from_rva(caller + site + 1) for site in sites]
        moves = {move: pe.get_offset_from_rva(caller + body.index(move))
                 for move in ALTERNATE if body.count(move) == 1}
    return caller, callee, sites, offsets, moves


def expect_rejected(path, reason):
    try:
        translate(path, "call_twice")
    except ValueError as exc:
        if reason not in str(exc):
            raise AssertionError(str(exc)) from exc
    else:
        raise AssertionError(f"accepted {path.name}, expected {reason!r}")


def check_translation(image, dropped):
    caller, callee, sites, offsets, moves = call_sites(image, "call_twice", "mix_args")
    if len(moves) != 2:
        raise AssertionError(f"expected one of each trace-pointer move, found {sorted(moves)}")
    expected = translate(image, "call_twice")
    original = Path(image).read_bytes()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "patched.dll"
        # The opcode 89 encodings of both moves must translate identically.
        patched = bytearray(original)
        for move, offset in moves.items():
            patched[offset:offset + 3] = ALTERNATE[move]
        path.write_bytes(patched)
        if translate(path, "call_twice") != expected:
            raise AssertionError("opcode 89 trace-pointer moves changed the translation")

        def redirect(targets):
            damaged = bytearray(original)
            for site, offset, target in zip(sites, offsets, targets):
                if target is not None:
                    struct.pack_into("<i", damaged, offset, target - (caller + site + 5))
            path.write_bytes(damaged)
            return path

        # A call into the callee interior names no function interval of its own.
        cases = (((None, callee + 1), "no unique function interval"),
                 ((callee + 1, None), "no unique function interval"),
                 ((callee + 1, callee + 1), "no unique function interval"),
                 ((caller, caller), "overlapping call target"))
        for targets, reason in cases:
            expect_rejected(redirect(targets), reason)

        # Reloading R8 from RBX, which holds an input word, must not pass as the trace.
        foreign = bytearray(original)
        offset = moves[MOVE_R8_RSI]
        foreign[offset:offset + 3] = bytes.fromhex("4c 8b c3")
        path.write_bytes(foreign)
        expect_rejected(path, "R8 reloaded without the output pointer")
    expect_rejected(Path(dropped), "undefined register at RVA")
    print("both call sites and trace-pointer moves found; opcode 89 moves translate identically; "
          f"{len(cases)} redirected calls, the omitted R8 reload and an R8 reload from RBX "
          "are rejected")


def check_mixed_calls():
    # An import call may repeat one IAT slot, but not mix with direct calls or name a second slot.
    prologue = bytes.fromhex("53 56 48 83 ec 28")
    epilogue = bytes.fromhex("48 83 c4 28 5e 5b c3")
    direct = bytes.fromhex("e8 00 01 00 00")
    imported = bytes.fromhex("ff 15 00 02 00 00")
    # Second IAT call targets a different slot from the first.
    other_imported = bytes.fromhex("ff 15 02 02 00 00")
    for calls, reason in (((direct, imported), "mixed import and direct calls"),
                          ((imported, direct), "mixed import and direct calls"),
                          ((imported, other_imported), "call to a second IAT slot")):
        try:
            lift_body(0x1000, prologue + b"".join(calls) + epilogue,
                      caller=True, bind_import=lambda slot: "mix_args")
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(str(exc)) from exc
        else:
            raise AssertionError(f"accepted mixed or distinct calls, expected {reason!r}")
    print("mixed direct/import calls and two distinct IAT slots are rejected")


def check_repeated_import(probe_path, import_path, import_lifted_path):
    probe = register_probe(probe_path)
    native = ctypes.CDLL(import_path)
    lifted = ctypes.CDLL(import_lifted_path)
    inputs = input_pairs()
    for x, y in inputs:
        original = observe(probe, native.call_twice, x, y)
        actual = observe(probe, lifted.lifted, x, y)
        if original != reference(x, y, caller=True) or actual != original:
            raise AssertionError(f"repeat_import({x:#x}, {y:#x}): {original=} {actual=}")
    print(f"{len(inputs)} repeated-import input pairs match native execution, guarded trace, RBX and RSI")


if __name__ == "__main__":
    probe_path, original_path, lifted_path, altered_path, dropped_path = sys.argv[1:6]
    check(probe_path, original_path, lifted_path, altered_path)
    check_translation(original_path, dropped_path)
    check_mixed_calls()
    for path in (lifted_path, altered_path):
        call_sites(path, "lifted", "lifted_callee")
    if len(sys.argv) > 6:
        check_repeated_import(probe_path, sys.argv[6], sys.argv[7])

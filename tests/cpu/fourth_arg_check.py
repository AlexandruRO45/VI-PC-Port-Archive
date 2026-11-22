"""Differential checks and controls for entries translated with three or four integer arguments."""

import ctypes
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Decoder, Mnemonic, Register

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import translate
from cpu_leaf_aot import export_rva
from direct_check import GUARDS, MASK, SENTINELS, Word, input_pairs, observe, register_probe

# The export list of cpu_translation(cpu_fourth_arg ...) in CMakeLists.txt: ":4" adds R9 as a fourth argument.
ENTRIES = ("r9_mix:4", "r9_loop:4", "no_r9:4", "r9_scratch")
# A four-argument entry that never reads R9: the fourth argument must not change any of its results.
CONTROL = "no_r9"
# R9D = 0 with and without upper bits, loop counts 1 to 64 (one under a set sign bit), counts above
# 64, and all ones.
FOURTHS = (0, 1, 2, 64, 65, (1 << 32) - 1, 1 << 32, 1 << 63, (1 << 63) | 3, MASK)


def register_probe4(probe_path):
    probe4 = ctypes.CDLL(probe_path).probe4
    probe4.argtypes = [ctypes.c_void_p, Word, Word, ctypes.POINTER(Word), Word, ctypes.POINTER(Word)]
    probe4.restype = Word
    return probe4


def observe4(probe4, fn, x, y, z):
    """Return value, guarded trace and sampled RBX/RSI after fn(x, y, trace, z)."""
    trace = (Word * 5)(GUARDS[0], MASK, MASK, MASK, GUARDS[1])
    regs = (Word * 2)()
    output = ctypes.cast(ctypes.byref(trace, ctypes.sizeof(Word)), ctypes.POINTER(Word))
    result = probe4(fn, x, y, output, z, regs)
    return result, tuple(trace), tuple(regs)


def reference(name, x, y, z=None):
    """RAX and the guarded trace after name(x, y, trace[, z]); an untouched word keeps the fill."""
    trace = [MASK] * 3
    if name == "r9_mix":
        rax, trace[0] = (x + 2 * z - y) & MASK, z
    elif name == "r9_loop":
        count = z & 0xffffffff
        if 1 <= count <= 64:
            rax, trace[0] = (x + count * y) & MASK, 0
        else:
            rax, trace[0] = x, z
    elif name == "no_r9":
        rax = trace[0] = (x - y) & MASK
    elif name == "r9_scratch":
        low = x & 0xff
        rax, trace[0] = (low + y) & MASK, (y - low) & MASK
    return rax, (GUARDS[0], *trace, GUARDS[1])


def signature(name, count):
    """The generated signature line of an entry translated with count integer arguments."""
    parameters = ", ".join(("word rcx", "word rdx", "word r8", "word r9")[:count])
    return f'extern "C" __declspec(dllexport) word lifted_{name}({parameters}) {{'


def check(probe_path, original_path, lifted_path, lifted_source):
    probe, probe4 = register_probe(probe_path), register_probe4(probe_path)
    # probe4 samples RBX and RSI through its sixth argument; clobber_control zeroes both.
    sampled = observe4(probe4, ctypes.CDLL(probe_path).clobber_control, 1, 2, 3)[2]
    if sampled != (0, 0):
        raise AssertionError(f"probe4 missed the register clobber: {sampled}")
    print("control clobber_control through probe4: RBX and RSI sampled as 0 and 0")
    source, functions = translate(original_path, list(ENTRIES))
    if Path(lifted_source).read_text(encoding="utf-8") != source:
        raise AssertionError(f"{lifted_source} is not the translation of {','.join(ENTRIES)}")
    print(f"the built translation is the translation of {','.join(ENTRIES)}")
    native, lifted = ctypes.CDLL(original_path), ctypes.CDLL(lifted_path)
    inputs = input_pairs()
    total = 0
    for spec, function in zip(ENTRIES, functions):
        name, _, count = spec.partition(":")
        count = int(count or 3)
        if len(function.arguments) != count or signature(name, count) not in source or function.return_width != 8:
            raise AssertionError(f"{spec}: recorded {len(function.arguments)} arguments, "
                                 f"return width {function.return_width}")
        # The host declares as many arguments as the entry was translated with.
        fourths = FOURTHS if count == 4 else (None,)
        varied = 0
        for x, y in inputs:
            seen = set()
            for z in fourths:
                want = (*reference(name, x, y, z), SENTINELS)
                for label, fn in (("native", getattr(native, name)), ("translated", getattr(lifted, "lifted_" + name))):
                    got = observe4(probe4, fn, x, y, z) if count == 4 else observe(probe, fn, x, y)
                    if got != want:
                        raise AssertionError(f"{label} {spec}({x:#x}, {y:#x}, trace, {z}) = {got}, expected {want}")
                    seen.add(got)
                total += 1
            varied += len(seen) > 1
        if count == 3:
            if "    word r9 = 0;" not in source.split(signature(name, 3))[1].split("\n}")[0]:
                raise AssertionError(f"{spec}: R9 is not a local of the three-argument translation")
            print(f"control {name} (three arguments, probe): writes R9 before reading it, r9 is a local, "
                  f"{len(inputs)} pairs match, native and translated")
        elif name == CONTROL:
            if varied:
                raise AssertionError(f"{name}: the fourth argument changed the result for {varied} pairs")
            print(f"control {name} (four arguments, probe4): identical results for all {len(FOURTHS)} fourth "
                  f"arguments on each of {len(inputs)} pairs, native and translated")
        else:
            print(f"{name} (four arguments, probe4): the fourth argument changes the result on {varied} of "
                  f"{len(inputs)} pairs")
    print(f"{total} calls per side match the reference on return, guarded trace, RBX and RSI")


def check_existing_caller(probe_path, narrow_path, narrow_lifted_path):
    """narrow.asm's byte_merge, translated with the default and called through probe as cpu_aot_narrow_returns does."""
    from narrow_check import reference as narrow_reference
    source, (function,) = translate(narrow_path, ["byte_merge"])
    if len(function.arguments) != 3 or signature("byte_merge", 3) not in source or "    word r9 = 0;" not in source:
        raise AssertionError(f"byte_merge default translation:\n{source}")
    probe = register_probe(probe_path)
    native, lifted = ctypes.CDLL(narrow_path).byte_merge, ctypes.CDLL(narrow_lifted_path).lifted_byte_merge
    caller = ctypes.cast(native, ctypes.c_void_p).value
    inputs = input_pairs()
    for x, y in inputs:
        want = (*narrow_reference("byte_merge", x, y, caller), SENTINELS)
        for label, fn in (("native", native), ("translated", lifted)):
            if observe(probe, fn, x, y) != want:
                raise AssertionError(f"{label} byte_merge({x:#x}, {y:#x}) != {want}")
    print(f"control narrow.asm byte_merge (existing entry, three arguments by default, writes R9 before reading it): "
          f"signature (word rcx, word rdx, word r8) with r9 a local, {len(inputs)} pairs through probe match "
          "native and the cpu_aot_narrow_returns reference")


def run_control(spec, change, image, reason):
    try:
        translate(image, [spec])
    except ValueError as exc:
        if reason not in str(exc):
            raise AssertionError(f"{spec} ({change}): expected {reason!r}, got {exc}") from exc
        print(f"rejected {spec} ({change}): {exc}")
    else:
        raise AssertionError(f"accepted control {spec} ({change})")


def check_modes(original_path):
    """r9_mix reads R9 first: three-argument mode must reject it, the explicit fourth argument accepts it."""
    run_control("r9_mix", "three arguments by default", original_path, "undefined register r9")
    source, (function,) = translate(original_path, ["r9_mix:4"])
    if len(function.arguments) != 4 or signature("r9_mix", 4) not in source:
        raise AssertionError(f"r9_mix:4 translation:\n{source}")
    print("accepted r9_mix:4 (explicit fourth argument): signature (word rcx, word rdx, word r8, word r9)")
    if translate(original_path, ["r9_scratch:3"])[0] != translate(original_path, ["r9_scratch"])[0]:
        raise AssertionError("r9_scratch:3 and r9_scratch translate differently")
    print("r9_scratch:3 and r9_scratch give the same translation")
    for spec in ("r9_mix:5", "r9_mix:x"):
        run_control(spec, "invalid argument count", original_path, "unsupported argument count")


def check_rejection(original_path):
    """lea rax, [rcx+r9*2] in r9_mix patched to lea rax, [rcx+r10*2]: R10 is not an argument register."""
    data = Path(original_path).read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        entry = export_rva(pe, "r9_mix")
        lea = Decoder(64, pe.get_data(entry, 0x10), ip=entry).decode()
        offset = pe.get_offset_from_rva(lea.ip)
    patch = bytes.fromhex("4a 8d 04 51")
    replacement = Decoder(64, patch, ip=lea.ip).decode()
    same = lambda i: (i.mnemonic, i.op0_register, i.memory_base, i.memory_index_scale, i.memory_displacement, i.len)
    if (lea.memory_index, replacement.memory_index) != (Register.R9, Register.R10) or lea.mnemonic != Mnemonic.LEA \
            or same(lea) != same(replacement) or len(patch) != lea.len:
        raise AssertionError(f"r9_mix: {lea} and {replacement} differ in more than the index register")
    with TemporaryDirectory() as folder:
        path = Path(folder) / "control.dll"
        damaged = bytearray(data)
        damaged[offset:offset + lea.len] = patch
        path.write_bytes(damaged)
        run_control("r9_mix:4", f"{lea} -> {replacement}", path, "undefined register r10")


if __name__ == "__main__":
    probe_path, original_path, lifted_path, lifted_source, narrow_path, narrow_lifted_path = sys.argv[1:]
    check(probe_path, original_path, lifted_path, lifted_source)
    check_existing_caller(probe_path, narrow_path, narrow_lifted_path)
    check_modes(original_path)
    check_rejection(original_path)

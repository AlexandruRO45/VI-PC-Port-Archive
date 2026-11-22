"""Check the archive reader against independent byte, status and callback observations."""
import ctypes
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

from exe_entry_check import Observation, check
from mapped_exe_chain_check import run_mapped


def wider_exit_control(native):
    # The startup consumes AL. Replace its zero-extension with MOV ECX,EAX
    # and a NOP. The status in AH and above was never established on error.
    from archive.translate_reader import BINDINGS, HOST
    from cpu_stdio_exe_aot import translate

    with pefile.PE(str(native)) as pe, TemporaryDirectory() as tmp:
        entry = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        unwind, = (item.struct for item in pe.DIRECTORY_ENTRY_EXCEPTION if item.struct.BeginAddress == entry)
        code = pe.get_data(entry, unwind.EndAddress - entry)
        original = bytes.fromhex("0f b6 c8")
        check("unique MOVZX ECX,AL in pinned startup", code.count(original), 1)
        offset = pe.get_offset_from_rva(entry + code.index(original))
        image = bytearray(native.read_bytes())
        image[offset:offset + 3] = bytes.fromhex("89 c1 90")
        control = Path(tmp) / native.name
        control.write_bytes(image)
        try:
            translate(control, BINDINGS, HOST, allow_process_exit=True)
        except ValueError as error:
            if "callee argument rcx is undefined" not in str(error):
                raise
        else:
            raise AssertionError("wide exit status accepted from a byte-sized result")


def compare(mapper, native, lifted, codec):
    wider_exit_control(native)
    first = bytes.fromhex("08 00 04 82 24 25 8f 80 7f")
    second = bytes.fromhex("08 00 04 84 24 25 8f 80 7f")
    full = ["alloc 8", "input 8", "output 13", "free"]
    partial = ["alloc 8", "input 4", "input 0", "output 1", "free"]
    cases = [
        (first + second, b"AIAIAIAIAIAIABIBIBIBIBIBIB", 0, full + full + ["completed 2"]),
        (first + second[:5], b"AIAIAIAIAIAIA", 2, full + partial + ["completed 1"]),
    ]
    commands = ([native], [sys.executable, "-B", __file__, "--mapped", mapper, native, codec], [lifted])
    for data, output, code, events in cases:
        trace = ("\n".join(events) + "\n").encode("ascii")
        for command in commands:
            child = subprocess.run(command, input=data, capture_output=True, timeout=15)
            check(str(command), Observation(child.stderr, child.stdout, child.returncode),
                  Observation(trace, output, code))
        print(f"{len(data)} input bytes: {output!r}, exit {code}, {events}")


if __name__ == "__main__":
    if sys.argv[1] == "--mapped":
        mapper, native, codec = map(Path, sys.argv[2:])
        run_mapped(mapper, native, {codec.name.lower(): ctypes.CDLL(str(codec)), "ntdll.dll": ctypes.WinDLL("ntdll")})
        raise AssertionError("the mapped Zig startup returned from process termination")
    compare(*map(Path, sys.argv[1:]))

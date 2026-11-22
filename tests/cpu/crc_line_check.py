"""Run an independently linked console program from its PE entry to process exit."""
import ctypes
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zlib

import pefile

from exe_entry_check import Observation, check, observation
from exe_process import launch
from mapped_exe_chain_check import run_mapped
from cpu_stdio_exe_aot import STDIO, translate


def compare(mapper, image, lifted, cases):
    commands = ([image], [sys.executable, "-B", __file__, "--mapped", mapper, image], [lifted])
    for label, data, output, code in cases:
        expected = Observation(b"", output, code)
        for command in commands:
            check(f"{image.name} {label} {command[0]}", observation(launch(command, data)), expected)
        print(f"{image.name} {label}: {output!r}, exit {code} on all three runners")
    with TemporaryDirectory() as directory:
        sink = Path(directory) / "read-only.txt"
        sink.write_bytes(b"unchanged")
        for command in commands:
            with sink.open("rb") as read_only:
                child = subprocess.run(command, input=b"123456789\n", stdout=read_only,
                                       stderr=subprocess.PIPE, timeout=15)
            check("write failure", (child.returncode, child.stderr, sink.read_bytes()), (2, b"", b"unchanged"))


def test(mapper, native, lifted, control, lifted_control):
    long_line = b"input record " * 1000
    cases = [("check vector", b"123456789\n", b"CBF43926\r\n", 0),
             ("missing line", b"", b"", 1),
             ("empty line", b"\n", b"00000000\r\n", 0),
             ("CRLF and trailing input", b"abc\r\nignored", b"352441C2\r\n", 0),
             ("EOF after data", b"abc", b"352441C2\r\n", 0),
             ("multiple input buffers", long_line, f"{zlib.crc32(long_line):08X}\r\n".encode(), 0)]
    compare(mapper, native, lifted, cases)
    compare(mapper, control, lifted_control, [("changed polynomial", b"123456789\n", b"E3069283\r\n", 0)])

    # A familiar symbol in a different DLL must never bind to the stdio host.
    with TemporaryDirectory() as directory, pefile.PE(str(native)) as pe:
        module, = pe.DIRECTORY_ENTRY_IMPORT
        offset = pe.get_offset_from_rva(module.struct.Name)
        data = bytearray(native.read_bytes())
        data[offset] = ord("x")
        changed = Path(directory) / native.name
        changed.write_bytes(data)
        try:
            translate(changed)
        except ValueError as exc:
            if "unbound console import xpi-ms-win-crt-stdio" not in str(exc):
                raise
        else:
            raise AssertionError("wrong provider accepted")


if __name__ == "__main__":
    if sys.argv[1] == "--mapped":
        import msvcrt
        import os

        # Python sets binary mode on its pipes; the native console EXE starts in UCRT text mode.
        msvcrt.setmode(0, os.O_TEXT)
        msvcrt.setmode(1, os.O_TEXT)
        sys.exit(run_mapped(Path(sys.argv[2]), Path(sys.argv[3]), {STDIO: ctypes.CDLL(STDIO)}))
    test(*map(Path, sys.argv[1:]))

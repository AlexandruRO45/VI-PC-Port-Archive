"""Compare complete Windows processes, including a translated inverted branch."""
import struct
import subprocess
import sys
from typing import NamedTuple


class Observation(NamedTuple):
    arguments: bytes
    output: bytes
    exit_code: int


def run(executable, selector):
    child = subprocess.run([executable], input=selector, capture_output=True, timeout=15)
    return Observation(child.stderr, child.stdout, child.returncode)


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def compare(native, lifted, inverted_native, inverted_lifted):
    references = [
        Observation(struct.pack("<QQ", 0x100000303, 0xfffffffd), bytes.fromhex("1103000000000000"), 0x60d),
        Observation(struct.pack("<QQ", 48, 6), bytes.fromhex("3500000000000000"), 0x57),
    ]
    for selector, expected in enumerate(references):
        data = bytes([selector])
        original = run(native, data)
        check(f"native selector {selector}", original, expected)
        check(f"AOT selector {selector}", run(lifted, data), original)
        inverted = run(inverted_native, data)
        check("inverted native", inverted, references[1 - selector])
        check("inverted AOT", run(inverted_lifted, data), inverted)
        for field in Observation._fields:
            if getattr(original, field) == getattr(inverted, field):
                raise AssertionError(f"selector {selector}: {field} missed the inverted branch")
        first, second = struct.unpack("<QQ", original.arguments)
        print(f"selector={selector} import=({first:#x}, {second:#x}) "
              f"output={original.output.hex()} exit={original.exit_code:#x} (inversion detected)")
    for program in (native, lifted, inverted_native, inverted_lifted):
        check("empty stdin", run(program, b""), Observation(b"", b"", 2))


if __name__ == "__main__":
    compare(*sys.argv[1:])

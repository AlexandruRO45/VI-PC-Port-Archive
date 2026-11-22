"""Check which x64 instruction prefixes the shared integer lowering accepts."""
from pathlib import Path
import sys

from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ffx_version_query import lower

# Each body starts at 0x1000 and ends with ret. The first instruction carries the prefix.
REJECTED = [
    ("65 48 8b 04 25 30 00 00 00 c3", "mov rax, gs:[0x30]"),
    ("65 48 8b 04 25 58 00 00 00 c3", "mov rax, gs:[0x58]"),
    ("65 8b 01 c3", "mov eax, gs:[rcx]"),
    ("64 8b 41 08 c3", "mov eax, fs:[rcx+8]"),
    ("65 89 01 c3", "mov gs:[rcx], eax"),
    ("f0 01 01 c3", "lock add [rcx], eax"),
    ("f0 83 01 01 c3", "lock add dword [rcx], 1"),
    ("f0 ff 01 c3", "lock inc dword [rcx]"),
    ("f0 09 01 c3", "lock or [rcx], eax"),
    ("f0 21 01 c3", "lock and [rcx], eax"),
    ("f0 48 29 11 c3", "lock sub [rcx], rdx"),
]

# Prefixes without an effect on these 64-bit forms must lower exactly as the bare encoding.
NEUTRAL = [
    ("3e 8b 01 c3", "8b 01 c3", "ds: mov eax, [rcx]"),
    ("2e 8b 01 c3", "8b 01 c3", "cs: mov eax, [rcx]"),
    ("26 8b 01 c3", "8b 01 c3", "es: mov eax, [rcx]"),
    ("36 8b 01 c3", "8b 01 c3", "ss: mov eax, [rcx]"),
    ("8b 01 f3 c3", "8b 01 c3", "rep ret"),
    ("01 01 c3", "01 01 c3", "unprefixed add [rcx], eax"),
]


def lowered(hexbytes):
    return lower(list(Decoder(64, bytes.fromhex(hexbytes), ip=0x1000)), [], {}, {}, {})


def statements(hexbytes):
    # A prefix lengthens the instruction and moves later labels; compare the emitted operations.
    return [line.split(": ", 1)[1] for line in lowered(hexbytes).splitlines()
            if line.strip().startswith("block_")]


def check_rejected():
    for hexbytes, text in REJECTED:
        try:
            lowered(hexbytes)
        except ValueError as exc:
            if not str(exc).startswith("unsupported instruction prefix at 0x1000: "):
                raise AssertionError(f"{text}: unexpected diagnostic {exc}") from exc
        else:
            raise AssertionError(f"accepted {text}")


def check_neutral():
    for prefixed, bare, text in NEUTRAL:
        if statements(prefixed) != statements(bare):
            raise AssertionError(f"{text} lowered differently from {bare}")


def check_host_override():
    # A host operation replaces the instruction, so its prefix is not lowered.
    body = list(Decoder(64, bytes.fromhex("65 48 8b 04 25 30 00 00 00 c3"), ip=0x1000))
    code = lower(body, [], {}, {}, {}, host_ops={0x1000: "rax = host_teb();"})
    if "block_1000: rax = host_teb();" not in code:
        raise AssertionError("host operation did not replace the prefixed instruction")


if __name__ == "__main__":
    check_rejected()
    check_neutral()
    check_host_override()
    print(f"{len(REJECTED)} FS/GS or LOCK forms rejected; {len(NEUTRAL)} neutral or unprefixed forms "
          "matched the bare lowering; host override kept")

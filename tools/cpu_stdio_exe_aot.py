"""Translate a return-based PE32+ console entry and its reachable functions with UCRT stdio imports."""
import argparse
from pathlib import Path

import pefile

from cpu_switch_aot import HostImport, PRELUDE, Program


STDIO = "api-ms-win-crt-stdio-l1-1-0.dll"
IMPORTS = {
    (STDIO, b"getchar"): HostImport("host_getchar", (), 4),
    (STDIO, b"putchar"): HostImport("host_putchar", (4,), 4),
    (STDIO, b"fflush"): HostImport("host_fflush", (8,), 4),
}
HOST = """
#include <cstdio>
static word host_getchar() { return static_cast<unsigned>(std::getchar()); }
static word host_putchar(word ch) { return static_cast<unsigned>(std::putchar(static_cast<int>(ch))); }
static word host_fflush(word stream) {
    return static_cast<unsigned>(std::fflush(reinterpret_cast<std::FILE*>(stream)));
}
"""


def translate(image, bindings=IMPORTS, host=HOST, *, allow_process_exit=False):
    with pefile.PE(str(image), fast_load=True) as pe:
        opt = pe.OPTIONAL_HEADER
        if (pe.FILE_HEADER.Machine != 0x8664 or opt.Magic != 0x20b or opt.Subsystem != 3
                or not pe.FILE_HEADER.Characteristics & 2 or pe.FILE_HEADER.Characteristics & 0x2000):
            raise ValueError("expected an AMD64 console EXE")
        for index, name in ((9, "TLS startup"), (13, "delay imports")):
            if len(opt.DATA_DIRECTORY) > index and opt.DATA_DIRECTORY[index].VirtualAddress:
                raise ValueError(f"unsupported {name} in console EXE")
        pe.parse_data_directories(directories=[1])
        imports = {}
        for module in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll = module.dll.decode("ascii").lower()
            for imp in module.imports:
                binding = bindings.get((dll, imp.name))
                if binding is None:
                    symbol = imp.name.decode("ascii") if imp.name else f"#{imp.ordinal}"
                    raise ValueError(f"unbound console import {dll}!{symbol}")
                slot = imp.address - opt.ImageBase
                if slot in imports:
                    raise ValueError(f"duplicate import slot {slot:#x}")
                imports[slot] = binding
        program = Program(pe, imports=imports)
        entry = opt.AddressOfEntryPoint
        if entry not in program.entries:
            raise ValueError(f"unverified process entry at RVA {entry:#x}")
        source, function = program.lift(entry, "process_entry", ())
        reaches_exit = allow_process_exit and any(call.noreturn for call in function.host_calls.values())
        if (function.return_width is None and not reaches_exit
                or function.return_width is not None and function.return_width < 4):
            raise ValueError("process entry must define its 32-bit exit code")
        return "\n".join([PRELUDE, host, *program.sources, source,
                          "int main() { return static_cast<int>(process_entry()); }", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        args.output.write_text(translate(args.image), encoding="ascii")
    except (ValueError, OSError, pefile.PEFormatError) as exc:
        parser.exit(1, f"{exc}\n")

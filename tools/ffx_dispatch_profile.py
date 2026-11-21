"""Identify the first ID 16 writer and reader in the pinned default dispatch graph."""
import argparse
from pathlib import Path
import struct

from ffx_version_query import UPSCALER, checked_image, lower, region
from resource_inventory import parts


def dispatch_slices(pe):
    body = lower(region(pe, 0x13278, 0x132cc),
                 ["uint8_t desc[0x189]{}; desc[0x188] = sharpen;",
                  "r14 = reinterpret_cast<word>(desc); r12 = reinterpret_cast<word>(&provider);"],
                 {}, {}, {0x132cc: "return;"})
    code = "\nvoid dispatch_aliases(RecurrentProvider& provider, bool sharpen) {\n" + body + "\n}\n"
    # Output size lives in the graph builder's stack frame. Its list header at
    # r13 is read during this block, before the new three dimensions are appended.
    writer = lower(region(pe, 0x7e6a, 0x7eaf),
                   ["uint8_t local[512]{}; rbp = reinterpret_cast<word>(local);",
                    "store<uint32_t>(rbp+32,width); store<uint32_t>(rbp+36,height); r13=rsp;"],
                   {}, {}, {0x7eaf: "return load<DispatchGrid>(rbp+0x1f0);"})
    reader = lower(region(pe, 0x13869, 0x13880) + region(pe, 0x1388d, 0x13895),
                   ["uint8_t local[272]{}; rbp = reinterpret_cast<word>(local);",
                    "rsi=width; store<uint32_t>(rbp+0x104,height);"],
                   {}, {}, {0x13880: "goto block_1388d;",
                            0x13895: "return {uint32_t(r8), uint32_t(r9), uint32_t(frame[4])};"})
    return (code + "DispatchGrid producer_grid(uint32_t width, uint32_t height) {\n" + writer +
            "\n}\nDispatchGrid rcas_grid(uint32_t width, uint32_t height) {\n" + reader + "\n}\n")


def inspect(pe):
    base = pe.OPTIONAL_HEADER.ImageBase

    def pointer(rva):
        return pe.get_qword_at_rva(rva) - base

    def bindings(record, offset):
        table = record + offset
        count = pe.get_dword_at_rva(table)
        return {pe.get_string_at_rva(pointer(pointer(table + 8) + 8*i)).decode():
                pe.get_dword_at_rva(pointer(table + 16) + 4*i) for i in range(count)}

    def resource_id(name, start, count):
        for i in range(count):
            row = start + i * 0x84
            if pe.get_data(row + 4, 128).decode("utf-16le").split("\0", 1)[0] == name:
                return pe.get_dword_at_rva(row)
        raise ValueError(f"Unknown binding {name}")

    writers = []
    for index in range(27):
        case = pe.get_dword_at_rva(0x260c0 + 4*index)
        load, = region(pe, case, case + 7)
        record = load.ip_rel_memory_address - 8
        srvs = bindings(record, 0x40)
        uavs = bindings(record, 0x68)
        if any(resource_id(name, 0x6fb60, 12) == 16 for name in srvs):
            raise ValueError(f"Unexpected earlier ID 16 reader in pass {index}")
        for name, slot in uavs.items():
            if resource_id(name, 0x72e00, 12) in (10, 16):
                writers.append((index, name, slot, record))
    if writers != [(26, "rw_mlsr_output_color", 2, 0xbd830)]:
        raise ValueError(f"Unexpected ID 16 writers {writers}")
    if set(struct.unpack("<8I", pe.get_data(0x6fb20, 32))) != {0}:
        raise ValueError("Unexpected RCAS permutation selector")
    reader = 0xbc0d0
    if (bindings(reader, 0x40) != {"r_rcas_input": 18} or
            resource_id("r_rcas_input", 0x6fb60, 12) != 16 or
            bindings(reader, 0x68) != {"rw_rcas_output": 11} or bindings(reader, 0xe0)):
        raise ValueError("Unexpected RCAS shader bindings")

    for name, record, threads in [("producer", writers[0][3], 256), ("rcas", reader, 64)]:
        blob = pointer(record + 8)
        chunks = parts(pe.get_data(blob, pe.get_dword_at_rva(record)))
        if struct.unpack_from("<3I", chunks[b"PSV0"], 40) != (threads, 1, 1):
            raise ValueError(f"Unexpected {name} group size")
        native = chunks[b"ACCC"]
        if native[:4] != b"FSPX":
            raise ValueError("Missing native compute shader")
        count, = struct.unpack_from("<I", native, 8)
        sections = struct.iter_unpack("<4I", native[16:16+count*16])
        (offset, size), = [(off, n) for tag, off, n, _ in sections if tag == 6]
        code = native[offset+0x24:offset+size]
        end, = [i+4 for i in range(0, len(code)-3, 4) if code[i:i+4] == b"\0\0\x81\xbf"]
        print(f"{name} container={blob:#x} threads={threads} code_bytes={end}")
        yield name, code[:end]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    with checked_image(args.image, UPSCALER) as pe:
        programs = list(inspect(pe))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name, code in programs:
        (args.output_directory / (name + ".bin")).write_bytes(code)

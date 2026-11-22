"""Lift an independent MASM query and compare both Windows entry points."""
import ctypes
from pathlib import Path
import sys

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import HEADER, lower, region
from version_query_check import check


def emit(image, output):
    with pefile.PE(str(image), fast_load=True) as pe:
        begin = export_rva(pe, "fixture_query")
        (interval,) = function_ranges(pe, begin)
        label = export_rva(pe, "fixture_name")
        name = pe.get_string_at_rva(label, 64)
        code = lower(region(pe, *interval), ["rcx = reinterpret_cast<word>(query);"],
                     {label: "reinterpret_cast<word>(name)"}, {}, {})
        probe_begin = export_rva(pe, "high_byte_probe")
        (probe_interval,) = function_ranges(pe, probe_begin)
        probe_code = lower(region(pe, *probe_interval), ["rcx = a; rdx = b;"], {}, {}, {})
        addr32_begin = export_rva(pe, "addr32_probe")
        (addr32_interval,) = function_ranges(pe, addr32_begin)
        addr32_code = lower(region(pe, *addr32_interval), ["rcx = a; rdx = b;"], {}, {}, {})
    data = ", ".join(map(str, name + b"\0"))
    output.write_text(HEADER + f"static const char name[] = {{{data}}};\n"
                      + 'extern "C" __declspec(dllexport) uint32_t query_versions(Query* query) {\n'
                      + code + "\n}\n"
                      + 'extern "C" __declspec(dllexport) word high_byte_probe(word a, word b) {\n'
                      + probe_code + "\n}\n"
                      + 'extern "C" __declspec(dllexport) word addr32_probe(word a, word b) {\n'
                      + addr32_code + "\n}\n", encoding="ascii")


def check_pair(native, other, symbol, cases):
    native_fn = getattr(ctypes.CDLL(str(native)), symbol)
    lifted_fn = getattr(ctypes.CDLL(str(other)), symbol)
    for fn in (native_fn, lifted_fn):
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = ctypes.c_uint64
    for a, b in cases:
        expected = native_fn(a, b)
        actual = lifted_fn(a, b)
        if actual != expected:
            raise AssertionError(f"{symbol}({a:#x}, {b:#x}): native={expected:#x} lifted={actual:#x}")
    print(f"{symbol}: {len(cases)} native/lifted cases matched")


def check_high_bytes(native, other):
    check_pair(native, other, "high_byte_probe", (
        (0x123456789abc0100, 0x0000000000000200),
        (0xfffffffffffffe01, 0x0000000000000300),
        (0x000000000000ff55, 0x00000000000001aa),
        (0xdeadbeefcafebabe, 0x1122334455667788),
        (0x00000000000000ff, 0x0000000000000001),
        (0x0000000000000100, 0x00000000000000ff),
    ))
    check_pair(native, other, "addr32_probe", (
        (0x12345678ffffffff, 0xdeadbeef00000002),
        (0xffffffff80000000, 0x0000000140000000),
        (0x0000000100000000, 0x0000000200000000),
        (0x0000000000000005, 0x0000000000000007),
    ))


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        for path, symbol in ((native, "fixture_query"), (other, "query_versions")):
            check(Path(path), symbol, 0xabcd000000001234, "fixture-7")
        check_high_bytes(Path(native), Path(other))
    else:
        raise ValueError(mode)

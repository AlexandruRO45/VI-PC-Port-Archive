"""Compare native and translated arithmetic used to size the descriptor heaps."""
import ctypes
from pathlib import Path
import sys

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region


def emit(image, output):
    with pefile.PE(str(image), fast_load=True) as pe:
        (bounds,) = function_ranges(pe, export_rva(pe, "heap_layout"))
        code = lower(region(pe, *bounds), ["rcx = count; rdx = reinterpret_cast<word>(sizes);"], {}, {}, {})
        (bounds,) = function_ranges(pe, export_rva(pe, "slot_probe"))
        scan = lower(region(pe, *bounds), ["rcx = count; rdx = reinterpret_cast<word>(flags);",
                                         "r8 = reinterpret_cast<word>(sizes); r9 = reinterpret_cast<word>(index);"],
                     {}, {}, {})
        (bounds,) = function_ranges(pe, export_rva(pe, "descriptor_fields"))
        fields = lower(region(pe, *bounds), ["rcx = count; rdx = reinterpret_cast<word>(src);",
                                             "r8 = reinterpret_cast<word>(dst);"], {}, {}, {})
        (bounds,) = function_ranges(pe, export_rva(pe, "view_offset"))
        offset = lower(region(pe, *bounds), ["rcx = slot; rdx = mip; r8 = stride; r9 = kind;"], {}, {}, {})
        (bounds,) = function_ranges(pe, export_rva(pe, "index_guard"))
        guard = lower(region(pe, *bounds), ["rcx = index;"], {}, {}, {})
    output.write_text(INTEGER_HEADER + 'extern "C" __declspec(dllexport) word heap_layout(uint32_t count, '
                      'uint32_t* sizes) {\n' + code + '\n}\n'
                      'extern "C" __declspec(dllexport) uint32_t slot_probe(uint32_t count, const uint8_t* flags, '
                      'const uint32_t* sizes, uint32_t* index) {\n' + scan + '\n}\n'
                      'extern "C" __declspec(dllexport) word descriptor_fields(uint32_t count, const uint8_t* src, '
                      'uint8_t* dst) {\n' + fields + '\n}\n'
                      'extern "C" __declspec(dllexport) uint32_t view_offset(uint32_t slot, uint32_t mip, '
                      'uint32_t stride, uint32_t kind) {\n' + offset + '\n}\n'
                      'extern "C" __declspec(dllexport) uint32_t index_guard(uint32_t index) {\n'
                      + guard + '\n}\n', encoding="ascii")


def check(image, native):
    lib = ctypes.CDLL(str(image.resolve()))
    layout = lib.heap_layout
    uint = ctypes.c_uint32
    layout.argtypes, layout.restype = [uint, ctypes.POINTER(uint)], ctypes.c_uint64
    for count in (0, 1, 2, 17, 0x80000001, 0xffffffff):
        sizes = (uint * 2)()
        result = layout(count, sizes)
        expected = (uint(count << 9).value, uint(count * 0x19000 + 0x10000).value,
                    uint(count * 0xbe000).value | 0x1100)
        if (*sizes, result) != expected:
            raise AssertionError(f"{image.name}: count {count:#x} yielded {(*sizes, result)}, expected {expected}")
    lib.view_offset.argtypes = [uint] * 4
    lib.view_offset.restype = uint
    for slot, mip, stride, kind in ((1, 0, 32, 2), (1, 3, 64, 2), (1, 3, 32, 4),
                                    (511, 0, 32, 1), (511, 1, 64, 2)):
        index = slot + (mip if kind in (1, 2) else 0)
        expected = index * stride if index < 512 else 0xffffffff
        if lib.view_offset(slot, mip, stride, kind) != expected:
            raise AssertionError(f"{image.name}: unexpected view offset for {(slot, mip, stride, kind)}")
    lib.index_guard.argtypes, lib.index_guard.restype = [uint], uint
    for index in (0, 7, 0x7fffffff, 0xfffffffc, 0xfffffffd, 0xfffffffe, 0xffffffff):
        expected = 0 if index == 0xffffffff else 1 if index >= 0xfffffffd else index
        if lib.index_guard(index) != expected:
            raise AssertionError(f"{image.name}: index guard returned {lib.index_guard(index):#x} for {index:#x}")
    if native and not ctypes.windll.kernel32.IsProcessorFeaturePresent(39):  # PF_AVX_INSTRUCTIONS_AVAILABLE
        print("Native vector fixture skipped: AVX unavailable")
        return
    probe = lib.slot_probe
    probe.argtypes = [uint, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(uint), ctypes.POINTER(uint)]
    probe.restype = uint
    for flags in ([], [0xa6], [3, 2], [1, 3]):
        sentinel = 0xfeedface
        index = uint(sentinel)
        values = (0xfffffffe, 1, 3, 5)
        result = probe(len(flags), (ctypes.c_uint8 * len(flags))(*flags), (uint * 4)(*values), ctypes.byref(index))
        selected = next((i for i, flag in enumerate(flags) if not flag & 1), sentinel)
        expected = uint(sum(values)).value if selected != sentinel else 0
        if (index.value, result) != (selected, expected):
            raise AssertionError(f"{image.name}: flags {flags} yielded {(index.value, result)}")
    fields = lib.descriptor_fields
    fields.argtypes = [uint, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
    fields.restype = ctypes.c_uint64
    src = (ctypes.c_uint8 * 32)(*range(32))
    for count in (1, 0x80001234):
        dst = (ctypes.c_uint8 * 96)(*([255] * 96))
        value = fields(count, src, dst)
        if value != (17 if count == 1 else count - 1) or bytes(dst) != bytes(src) + bytes(src[:16]) + bytes(48):
            raise AssertionError(f"{image.name}: descriptor copy or field selection differs for {count:#x}")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), native=True)
        check(Path(other), native=False)
    else:
        raise ValueError(mode)

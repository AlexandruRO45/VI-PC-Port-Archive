"""Compare one translated draw from the same DLL under Windows and pe_map."""

import argparse
import ctypes
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from native_map_check import Mapper, STATUS, kernel
from native_tls_check import ATTACH, DETACH, ENTRY, TLS, calls, load_sink

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel_aot_check import dual_program, translate_dual


Draw = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint32)
GuestDraw = ctypes.CFUNCTYPE(ctypes.c_int, Draw)
SHADER_SELECTOR = 2
FRAME_BYTES = 64 * 64 * 4


def check_draw(base, export, release, sink, draw, label):
    try:
        expected = [[TLS, ATTACH, "module", 0], [ENTRY, ATTACH, "module", 0]]
        assert calls(sink, base) == expected, f"{label}: attach order {calls(sink, base)}"
        address = export(base, b"guest_draw")
        assert address, f"{label}: guest_draw export missing"
        result = GuestDraw(address)(draw)
    finally:
        sink.sink_clear()
        release(base)
    expected = [[TLS, DETACH, "module", 0], [ENTRY, DETACH, "module", 0]]
    assert calls(sink, base) == expected, f"{label}: detach order {calls(sink, base)}"
    return result


def compare(args, directory):
    source = directory / "fixture.hlsl"
    shader = directory / "fixture.cso"
    source.write_text(translate_dual(dual_program("reuse")), encoding="ascii")
    subprocess.run([args.dxc, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source],
                   check=True, timeout=30)
    sink = load_sink(args.sink)
    provider = ctypes.CDLL(str(args.provider))
    mapper = Mapper(args.mapper)
    sink_bindings = {args.sink.name.casefold(): sink}
    bindings = {**sink_bindings, args.provider.name.casefold(): provider}

    sink.sink_clear()
    base, status, detail, _ = mapper(args.guest, sink_bindings)
    if base:
        mapper.release(base)
    assert not base and status == STATUS["unresolved"] and "frame_render" in detail, detail
    assert not calls(sink), f"missing graphics import reached startup: {calls(sink)}"
    print("Withheld graphics import rejected before TLS, DLL entry or draw", flush=True)

    kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
    kernel.LoadLibraryW.restype = ctypes.c_void_p
    kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
    captures = []
    for mode in ("Windows", "pe_map"):
        capture = directory / f"{mode}.rgba"
        selectors = []

        @Draw
        def draw(selector):
            selectors.append(selector)
            if selector != SHADER_SELECTOR:
                return 1
            try:
                return subprocess.run(
                    [args.renderer, args.backend, "--dual-pixel", shader, capture],
                    timeout=30,
                ).returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                print(error, file=sys.stderr)
                return 1

        sink.sink_clear()
        if mode == "Windows":
            dll_directory = args.guest.parent
            assert args.sink.parent == dll_directory and args.provider.parent == dll_directory
            base = kernel.LoadLibraryW(str(args.guest))
            if not base:
                raise ctypes.WinError(ctypes.get_last_error())
            result = check_draw(base, kernel.GetProcAddress, kernel.FreeLibrary, sink, draw, mode)
        else:
            base, status, detail, _ = mapper(args.guest, bindings)
            assert base and status == STATUS["ok"], detail
            result = check_draw(base, mapper.export, mapper.release, sink, draw, mode)
        assert selectors == [SHADER_SELECTOR], f"{mode}: expected exactly one guest draw, got {selectors}"
        if mode == "Windows" and args.backend == "hardware" and result == 77:
            return 77
        assert result == 0, f"{mode}: render returned {result}"
        pixels = capture.read_bytes()
        assert len(pixels) == FRAME_BYTES, f"{mode}: unexpected readback length {len(pixels)}"
        print(f"{mode}: TLS, entry, guest draw and CPU pixel reference verified", flush=True)
        captures.append(pixels)

    assert captures[0] == captures[1], "Windows and pe_map drew different RGBA frames"
    print("Windows and pe_map readbacks are byte-identical")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("guest", "sink", "provider", "mapper", "dxc", "renderer"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    parser.add_argument("backend", choices=("hardware", "warp"))
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        return compare(args, Path(directory))


if __name__ == "__main__":
    sys.exit(main())

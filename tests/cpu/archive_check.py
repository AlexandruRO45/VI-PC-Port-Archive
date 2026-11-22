"""Check the archive reader against independent byte, status and callback observations."""
import ctypes
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

from exe_entry_check import Observation, check
from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS


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


def module_names(image):
    with pefile.PE(str(image)) as pe:
        return {module.dll.decode("ascii").lower() for module in pe.DIRECTORY_ENTRY_IMPORT}


def mapped_exports(mapper, image, providers):
    base, status, detail, _ = mapper(image, providers)
    if not base or status != STATUS["ok"]:
        raise AssertionError(f"mapping {image.name}: {status}, {detail}")
    with pefile.PE(str(image)) as pe:
        return {entry.name: mapper.export(base, entry.name) for entry in pe.DIRECTORY_ENTRY_EXPORT.symbols}


def mapped_archive(mapper_path, native, codec, source=None):
    if source is None:
        provider = ctypes.CDLL(str(codec))
    else:
        mapper = Mapper(mapper_path)
        # Keep all three mappings alive through RtlExitUserProcess in this child.
        blast = mapped_exports(mapper, source, {"vcruntime140.dll": ctypes.CDLL("vcruntime140.dll")})
        hosts = {name: ctypes.CDLL(name) for name in module_names(codec) - {source.name.lower()}}
        provider = mapped_exports(mapper, codec, {**hosts, source.name.lower(): blast})
    run_mapped(mapper_path, native, {codec.name.lower(): provider, "ntdll.dll": ctypes.WinDLL("ntdll")})
    raise AssertionError("the mapped Zig startup returned from process termination")


def rejected_output(decode):
    # blast must pass the caller's actual callback contexts and propagate the refusal.
    byte_ptr = ctypes.POINTER(ctypes.c_ubyte)
    input_type = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(byte_ptr))
    output_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, byte_ptr, ctypes.c_uint)
    decode.argtypes = [input_type, ctypes.c_void_p, output_type, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(byte_ptr)]
    decode.restype = ctypes.c_int
    compressed = bytes.fromhex("00 04 82 24 25 8f 80 7f")
    payload = (ctypes.c_ubyte * len(compressed)).from_buffer_copy(compressed)
    trace = []

    @input_type
    def supply(context, pointer):
        pointer[0] = ctypes.cast(payload, byte_ptr)
        trace.append(("input", context))
        return len(payload) if len(trace) == 1 else 0

    @output_type
    def refuse(context, pointer, size):
        trace.append(("output", context, ctypes.string_at(pointer, size)))
        return 1

    result = decode(supply, 101, refuse, 202, None, None)
    check("output refusal", (result, trace), (1, [("input", 101), ("output", 202, b"AIAIAIAIAIAIA")]))


def compare(mapper, native, lifted, codec, *blast_images):
    wider_exit_control(native)
    first = bytes.fromhex("08 00 04 82 24 25 8f 80 7f")
    second = bytes.fromhex("08 00 04 84 24 25 8f 80 7f")
    full = ["alloc 8", "input 8", "output 13", "free"]
    partial = ["alloc 8", "input 4", "input 0", "output 1", "free"]
    cases = [
        (first + second, b"AIAIAIAIAIAIABIBIBIBIBIBIB", 0, full + full + ["completed 2"]),
        (first + second[:5], b"AIAIAIAIAIAIA", 2, full + partial + ["completed 1"]),
    ]
    mapped = [sys.executable, "-B", __file__, "--mapped", mapper, native, codec]
    if blast_images:
        source, translated, adapter = blast_images
        for dll in (source, translated):
            rejected_output(ctypes.CDLL(str(dll)).blast)
        mapped.append(source)
        check("native codec binding", source.name.lower() in module_names(codec), True)
        check("AOT reader binding", adapter.name.lower() in module_names(lifted), True)
        check("AOT decoder binding", translated.name.lower() in module_names(adapter), True)
        for image in (lifted, adapter, translated):
            check("no native decoder fallback", module_names(image) & {source.name.lower(), codec.name.lower()}, set())
    commands = ([native], mapped, [lifted])
    for data, output, code, events in cases:
        trace = ("\n".join(events) + "\n").encode("ascii")
        for command in commands:
            child = subprocess.run(command, input=data, capture_output=True, timeout=15)
            check(str(command), Observation(child.stderr, child.stdout, child.returncode),
                  Observation(trace, output, code))
        print(f"{len(data)} input bytes: {output!r}, exit {code}, {events}")


if __name__ == "__main__":
    if sys.argv[1] == "--mapped":
        mapped_archive(*map(Path, sys.argv[2:]))
    compare(*map(Path, sys.argv[1:]))

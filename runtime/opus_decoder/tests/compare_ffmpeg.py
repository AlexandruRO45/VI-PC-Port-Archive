"""Exercise the built native C ABI with generated packets and FFmpeg's native Opus decoder."""
import argparse
import ctypes as ct
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from awc_opus import write_ogg_opus

FRAMES = 960
GUARD = 0xA5
TOLERANCE = 1 / 32768


def invoke(args, data=None):
    return subprocess.run(args, input=data, stdout=subprocess.PIPE, check=True, timeout=15).stdout


def fixture(directory, channels):
    # Distinct left and right signals and two consecutive packets exercise decoder history.
    samples = [round(9000 * math.sin(i * (0.039 + 0.023 * ch + 0.011 * channels)))
               for i in range(2 * FRAMES) for ch in range(channels)]
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    path = directory / f"{channels}.ogg"
    ffmpeg = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1"]
    invoke(ffmpeg + ["-f", "s16le", "-ar", "48000", "-ac", str(channels), "-i", "pipe:0",
                     "-c:a", "libopus", "-application", "lowdelay", "-b:a", "96k", "-vbr", "off",
                     "-frame_duration", "20", "-frames:a", "2", str(path)], pcm)
    raw = invoke(ffmpeg + ["-i", str(path), "-map", "0:a:0", "-c:a", "copy",
                           "-frames:a", "2", "-f", "data", "pipe:1"])
    # 96 kbit/s CBR, 20 ms = 240 bytes per packet.
    assert len(raw) == 480, (channels, len(raw))
    packets = [raw[:240], raw[240:]]
    # Compare every frame including priming samples. The runtime API takes raw packets.
    oracle = directory / f"oracle-{channels}.ogg"
    write_ogg_opus(str(oracle), packets, channels, 17 + channels, 2 * FRAMES, pre_skip=0)
    expected = invoke(ffmpeg + ["-c:a", "opus", "-i", str(oracle), "-f", "f32le", "pipe:1"])
    assert len(expected) == 2 * FRAMES * channels * 4
    return packets, expected


def compare(library):
    api = ct.CDLL(str(library.resolve()))
    handle = ct.c_void_p
    size = ct.c_size_t
    api.leonida_opus_create.argtypes = [ct.c_uint32, ct.POINTER(handle)]
    api.leonida_opus_decode.argtypes = [handle, ct.c_void_p, size, ct.c_void_p, size, ct.POINTER(size)]
    api.leonida_opus_destroy.argtypes = [handle]
    api.leonida_opus_destroy.restype = None
    handles = []
    outputs = [(ct.c_uint8 * (FRAMES * 8 + 64))() for _ in range(2)]
    max_error = 0.0
    try:
        with tempfile.TemporaryDirectory() as temporary:
            fixtures = [fixture(Path(temporary), ch) for ch in (1, 2)]
            for channels in (1, 2):
                decoder = handle()
                assert api.leonida_opus_create(channels, ct.byref(decoder)) == 0 and decoder.value
                handles.append(decoder)
            for sequence in (0, 1):
                for stream, (packets, expected) in enumerate(fixtures):
                    index = stream ^ sequence
                    output = outputs[index]
                    other = bytes(outputs[1-index])
                    ct.memset(output, GUARD, len(output))
                    packet = ct.create_string_buffer(packets[sequence])
                    byte_count = FRAMES * (stream + 1) * 4
                    written = size(123)
                    decode = api.leonida_opus_decode
                    if sequence == 1:
                        assert decode(handles[stream], packet, 240, output, byte_count-1, ct.byref(written)) == 2
                        assert written.value == 0 and bytes(output) == bytes([GUARD]) * len(output)
                    assert decode(handles[stream], packet, 240, output, len(output), ct.byref(written)) == 0
                    assert written.value == byte_count
                    assert bytes(output)[byte_count:] == bytes([GUARD]) * (len(output)-byte_count)
                    assert bytes(outputs[1-index]) == other
                    fmt = f"<{byte_count // 4}f"
                    actual = struct.unpack_from(fmt, output)
                    wanted = struct.unpack_from(fmt, expected, sequence * byte_count)
                    assert max(map(abs, wanted)) > 0.1
                    assert all(math.isfinite(x) for x in actual)
                    error = max(abs(a-b) for a, b in zip(actual, wanted))
                    assert error <= TOLERANCE, (stream, sequence, error)
                    max_error = max(max_error, error)
    finally:
        for decoder in handles:
            api.leonida_opus_destroy(decoder)
    print(f"4 packet decodes, 5760 float samples, maximum absolute error {max_error:.8f}")


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run this check with Python assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    compare(parser.parse_args().library)

"""Check aggregate scene metadata using compiled project shaders and synthetic envelopes."""
from pathlib import Path
import struct
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from resource_inventory import scene_records


def record(blob, *, repeat_delta=0, decoded_delta=0, tail=b""):
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(blob) + encoder.flush() + tail
    size = len(payload)
    return b"owned\0\0" + struct.pack("<4I", size, 0, size + repeat_delta, len(blob) + decoded_delta) + payload


def main():
    compute_path, pixel_path, validator = map(Path, sys.argv[1:])
    compute, pixel = compute_path.read_bytes(), pixel_path.read_bytes()
    # Give the stub a DXIL program header and four bytes without LLVM framing.
    stub = struct.pack("<II4sIII", (5 << 16) | 0x60, 7, b"DXIL", 0x100, 16, 4) + bytes(4)
    container = (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 44 + len(stub), 1, 36)
                 + b"DXIL" + struct.pack("<I", len(stub)) + stub)
    bad_magic = container[:-len(stub)] + stub[:8] + b"BAD!" + stub[12:]
    invalid = scene_records(record(bad_magic))
    assert invalid["records"] == 0, invalid
    assert invalid["candidate_rejections"] == {"no_program_header": 1}, invalid

    data = (record(bad_magic) + record(compute, tail=b"pad") + record(compute)
            + record(pixel) + record(container))
    result = scene_records(data, validator)
    assert result["records"] == 4 and result["stages"] == {"compute": 3, "pixel": 1}
    assert result["framed_bitcode"] == 3 and result["section_counts"]["DXIL"] == 4
    assert result["trailing_byte_counts"] == {3: 1, 0: 3}
    assert result["validator_samples"] == 2
    assert all(row["status"] == "0x00000000" for row in result["first_per_stage"].values())
    for data, reason in [(record(compute, repeat_delta=1), "unequal_stored_lengths"),
                         (record(pixel)[:-3], "truncated_span"),
                         (record(compute, decoded_delta=-1), "decoded_length_or_eof"),
                         (record(b"wrong tag" + compute[9:]), "container_framing")]:
        result = scene_records(data)
        assert result["records"] == 0 and result["candidate_rejections"][reason] >= 1
    assert scene_records(b"")["records"] == 0


if __name__ == "__main__":
    main()

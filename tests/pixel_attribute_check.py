"""Verify entry-mask interpolation of all four attr0 components with no sampler."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from pixel_aot_check import attribute_color, lower, must_reject, translate_attribute


def run(directory, compiler, renderer, backend):
    for rg in (False, True):
        code = attribute_color(rg)
        must_reject(lambda: lower(code, 14, 8, None))
        must_reject(translate_attribute, [s.replace("v8, v1, attr0.w", "v8, v1, attr0.z") for s in code])
        must_reject(translate_attribute, [s.replace("v9, v0, attr0.z", "v9, v1, attr0.z") for s in code])
        must_reject(translate_attribute, [s.replace("attr0.z", "attr1.z") for s in code])
        src, cso, case = [directory / name for name in ("color.hlsl", "color.cso", "color.case")]
        src.write_text(translate_attribute(code), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", cso, src], check=True)
        cmd = [renderer, backend, "--check-pixel-attribute-rg" if rg else "--check-pixel-attribute", cso, case]
        data = bytearray(256)
        for y in range(32):
            for x in range(32):
                u, v = (x + 0.5) / 32, (y + 0.5) / 32
                data += struct.pack("<4f", 1-v, 0.25 + 0.5*u, -1 if rg else u, -1 if rg else v)
        case.write_bytes(data)
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
        # A wrong channel at an interior pixel must be detected.
        struct.pack_into("<f", data, 256 + (9 * 32 + 17) * 16, 0)
        case.write_bytes(data)
        wrong = subprocess.run(cmd, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch x=17 y=9" not in wrong.stderr:
            raise AssertionError("Wrong interpolated channel escaped detection")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(run(Path(tmp), *sys.argv[1:]))

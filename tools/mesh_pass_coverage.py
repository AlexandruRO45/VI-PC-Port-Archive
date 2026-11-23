"""Compare a Vulkan pass coverage mask with CPU-projected triangle coverage.

Supply a JSON array of triangles in framebuffer coordinates (x right, y down)
and a coverage mask with one byte per pixel (nonzero = a fragment covered the
pixel). The mask comes from vk_harness's VKH_COVERAGE_MASK stencil readback, so
the fragment colour cannot hide a covered pixel.

Triangles must already be clipped and culled as appropriate for the pass.
Coverage is sampled at pixel centres using the top-left edge rule after the
projected vertices are snapped to the device's subpixel grid (default 8 bits,
the value reported by VkPhysicalDeviceLimits::subPixelPrecisionBits).

The optional --raw RGBA32F readback is streamed only to validate its size; it
is never used to decide coverage.
"""
import argparse
import json
import math
from pathlib import Path
import struct

CHUNK = 64 * 1024


def snap(value, bits):
    """Round a projected coordinate to the device's subpixel grid.

    `bits` is VkPhysicalDeviceLimits::subPixelPrecisionBits (0 disables the
    snap). Python's round() is ties-to-even; the measured NVIDIA behaviour for
    an exact half-step is the even grid coordinate, which this matches.
    """
    if bits <= 0:
        return value
    scale = 1 << bits
    return round(value * scale) / scale


def edge(a, b, x, y):
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def rasterize(triangles, width, height, subpixel_bits=8):
    """A bytearray of width*height; 1 means the CPU projection covers the pixel."""
    mask = bytearray(width * height)
    for tri in triangles:
        a, b, c = [(snap(p[0], subpixel_bits), snap(p[1], subpixel_bits)) for p in tri]
        area = edge(a, b, *c)
        if not area:
            continue
        if area < 0:
            b, c = c, b
        edges = ((a, b), (b, c), (c, a))
        left, top = (max(0, math.ceil(min(a[i], b[i], c[i]) - 0.5)) for i in range(2))
        right, bottom = (min(size - 1, math.floor(max(a[i], b[i], c[i]) - 0.5))
                         for i, size in enumerate((width, height)))
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                for start, end in edges:
                    distance = edge(start, end, x + 0.5, y + 0.5)
                    inclusive = end[1] < start[1] or (end[1] == start[1] and end[0] > start[0])
                    if distance < 0 or (distance == 0 and not inclusive):
                        break
                else:
                    mask[y * width + x] = 1
    return mask


def _chunks(path):
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                return
            yield chunk


def _check_size(path, expected, kind):
    seen = 0
    for chunk in _chunks(path):
        seen += len(chunk)
        if seen > expected:
            raise ValueError(f"{kind} is larger than {expected} bytes")
    if seen != expected:
        raise ValueError(f"{kind} is {seen} bytes, expected {expected}")


def compare(triangles, coverage, width, height, raw=None, subpixel_bits=8):
    """Return the IoU and the missing/extra counts for one coverage mask."""
    if not 0 <= subpixel_bits <= 16:
        raise ValueError("subpixel bits must be in [0, 16]")
    expected = rasterize(triangles, width, height, subpixel_bits)
    expected_count = sum(expected)
    if expected_count == 0:
        raise ValueError("CPU projection covers no pixel centres")
    actual = intersection = seen = 0
    for chunk in _chunks(coverage):
        if seen + len(chunk) > width * height:
            raise ValueError("coverage mask is larger than the requested dimensions")
        for i, value in enumerate(chunk):
            if value:
                actual += 1
                if expected[seen + i]:
                    intersection += 1
        seen += len(chunk)
    if seen != width * height:
        raise ValueError(f"coverage mask is {seen} bytes, expected {width * height}")
    if raw is not None:
        _check_size(raw, width * height * 16, "RGBA32F readback")
    union = expected_count + actual - intersection
    return {
        "iou": intersection / union if union else 1.0,
        "missing": expected_count - intersection,
        "extra": actual - intersection,
        "expected": expected_count,
        "actual": actual,
        "mask": expected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("triangles", type=Path)
    parser.add_argument("coverage", type=Path,
                        help="one byte per pixel; nonzero means the GPU covered the pixel")
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("--raw", type=Path, default=None,
                        help="optional RGBA32F readback, streamed only to validate its size")
    parser.add_argument("--subpixel-bits", type=int, default=8,
                        help="VkPhysicalDeviceLimits::subPixelPrecisionBits (default 8; 0 disables)")
    parser.add_argument("--min-iou", type=float, default=1.0)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or not 0 <= args.min_iou <= 1:
        parser.error("dimensions must be positive and --min-iou must be in [0, 1]")
    try:
        triangles = json.loads(args.triangles.read_text(encoding="utf-8"))
        if not isinstance(triangles, list) or any(
            not isinstance(tri, list) or len(tri) != 3 or any(
                not isinstance(point, list) or len(point) != 2 or any(
                    type(coord) not in (int, float) or not math.isfinite(coord) for coord in point
                ) for point in tri
            ) for tri in triangles
        ):
            parser.error("expected triangles of three finite [x, y] points")
        result = compare(triangles, args.coverage, args.width, args.height,
                         raw=args.raw, subpixel_bits=args.subpixel_bits)
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))
    print(f"IoU {result['iou']:.6f}, missing {result['missing']}, extra {result['extra']}")
    return int(result["iou"] < args.min_iou)


if __name__ == "__main__":
    raise SystemExit(main())

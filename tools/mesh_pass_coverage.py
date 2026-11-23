"""Compare an RGBA32F Vulkan readback with CPU-projected triangle coverage.

Supply a JSON array of triangles in framebuffer coordinates (x right, y down).
Triangles must already be clipped and culled as appropriate for the pass.
Coverage is sampled at pixel centers using the top-left edge rule.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def edge(a, b, x, y):
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def rasterize(triangles, width, height):
    covered = set()
    for a, b, c in triangles:
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
                    covered.add(y * width + x)
    return covered


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("triangles", type=Path)
    parser.add_argument("readback", type=Path)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
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
        raw = args.readback.read_bytes()
        if len(raw) != args.width * args.height * 16:
            parser.error("readback size does not match the RGBA32F dimensions")
        expected = rasterize(triangles, args.width, args.height)
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))
    if not expected:
        parser.error("CPU projection covers no pixel centers")
    actual = {i for i, rgba in enumerate(struct.iter_unpack("<4f", raw))
              if rgba != (-1.0, -1.0, -1.0, -1.0)}
    iou = len(expected & actual) / len(expected | actual)
    print(f"IoU {iou:.6f}, missing {len(expected - actual)}, extra {len(actual - expected)}")
    return int(iou < args.min_iou)


if __name__ == "__main__":
    raise SystemExit(main())

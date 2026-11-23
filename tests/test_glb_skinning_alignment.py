"""Synthetic alignment and zero-count bounds tests for tools/glb_skinning.py. No game data."""
from __future__ import annotations

import json
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import glb_skinning  # noqa: E402

_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _build_glb(gltf: dict, bin_data: bytes) -> bytes:
    json_bytes = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    bin_bytes = bytes(bin_data) + b"\x00" * ((-len(bin_data)) % 4)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    return b"".join((
        struct.pack("<4sII", b"glTF", 2, total_length),
        struct.pack("<II", len(json_bytes), glb_skinning.CHUNK_TYPE_JSON),
        json_bytes,
        struct.pack("<II", len(bin_bytes), glb_skinning.CHUNK_TYPE_BIN),
        bin_bytes,
    ))


def _skinned_doc(
    *,
    pos_view_off: int = 0,
    pos_acc_off: int = 0,
    joints_view_off: int = 0,
    joints_acc_off: int = 0,
    joints_component: int = 5121,
) -> tuple[dict, bytes]:
    # Reserve generous zero-padded space for arbitrary offset testing.
    buf = bytearray(512)
    pos_base = pos_view_off + pos_acc_off
    for i, pt in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
        struct.pack_into("<3f", buf, pos_base + i * 12, *pt)
    pos_view_len = pos_acc_off + 36

    weights_view_off = 128
    for i, w in enumerate(((1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))):
        struct.pack_into("<4f", buf, weights_view_off + i * 16, *w)

    joints_base = 256 + joints_view_off + joints_acc_off
    jfmt = "<4H" if joints_component == 5123 else "<4B"
    jsize = 8 if joints_component == 5123 else 4
    for i, j in enumerate(((0, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0))):
        struct.pack_into(jfmt, buf, joints_base + i * jsize, *j)
    joints_view_len = joints_acc_off + 3 * jsize

    ibm_view_off = 384
    for i in range(2):
        struct.pack_into("<16f", buf, ibm_view_off + i * 64, *_IDENTITY)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Joint0"}, {"name": "Joint1"}, {"mesh": 0, "skin": 0}],
        "skins": [{"joints": [0, 1], "inverseBindMatrices": 3}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0, "WEIGHTS_0": 1, "JOINTS_0": 2}
            }]
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_view_off, "byteLength": pos_view_len},
            {"buffer": 0, "byteOffset": weights_view_off, "byteLength": 48},
            {"buffer": 0, "byteOffset": 256 + joints_view_off, "byteLength": joints_view_len},
            {"buffer": 0, "byteOffset": ibm_view_off, "byteLength": 128},
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": pos_acc_off, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "byteOffset": joints_acc_off, "componentType": joints_component, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "byteOffset": 0, "componentType": 5126, "count": 2, "type": "MAT4"},
        ],
        "buffers": [{"byteLength": len(buf)}],
    }
    return gltf, bytes(buf)


class TestGlbSkinningAlignment(unittest.TestCase):
    def test_aligned_nonzero_offsets_accepted(self):
        for kw in (
            {"pos_view_off": 4, "pos_acc_off": 8, "joints_component": 5123, "joints_view_off": 2, "joints_acc_off": 4},
            {"pos_view_off": 8, "pos_acc_off": 4, "joints_component": 5121, "joints_view_off": 1, "joints_acc_off": 3},
        ):
            with self.subTest(kw=kw):
                gltf, buf = _skinned_doc(**kw)
                report = glb_skinning.analyze_glb(_build_glb(gltf, buf))
                self.assertTrue(report["ok"], report["errors"])

    def test_misaligned_accessor_byte_offset_rejected(self):
        for kw in (
            {"pos_view_off": 0, "pos_acc_off": 2},
            {"joints_component": 5123, "joints_view_off": 0, "joints_acc_off": 1},
        ):
            with self.subTest(kw=kw):
                gltf, buf = _skinned_doc(**kw)
                with self.assertRaisesRegex(
                    glb_skinning.GlbSkinError, r"byteOffset \d+ is not a multiple of the component size"
                ):
                    glb_skinning.analyze_glb(_build_glb(gltf, buf))

    def test_misaligned_effective_buffer_offset_rejected(self):
        for kw in (
            {"pos_view_off": 2, "pos_acc_off": 0},
            {"joints_component": 5123, "joints_view_off": 1, "joints_acc_off": 0},
        ):
            with self.subTest(kw=kw):
                gltf, buf = _skinned_doc(**kw)
                with self.assertRaisesRegex(
                    glb_skinning.GlbSkinError, r"buffer offset \d+ is not a multiple of the component size"
                ):
                    glb_skinning.analyze_glb(_build_glb(gltf, buf))

    def test_zero_count_accessor_bounds(self):
        buf = b"\x00" * 16
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 16}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 8}],
            "accessors": [
                {"bufferView": 0, "byteOffset": 8, "componentType": 5126, "count": 0, "type": "VEC3"},
                {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 0, "type": "VEC3"},
            ],
        }
        reader = glb_skinning._Reader(gltf, buf)
        values, meta = reader.read(0, "test.zero_ok")
        self.assertEqual(values, [])
        self.assertEqual(meta["count"], 0)
        with self.assertRaisesRegex(glb_skinning.GlbSkinError, r"exceeds bufferViews\[0\] length 8"):
            reader.read(1, "test.zero_oob")


if __name__ == "__main__":
    unittest.main()

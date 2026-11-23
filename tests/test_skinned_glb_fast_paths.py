"""Parity tests for the column-wise fast paths in tools/skinned_glb.py.

    python -I tests/test_skinned_glb_fast_paths.py

Every stream is generated here. Each fast path is compared byte for byte with the
per-vertex / per-row reference it replaced, with numpy hidden (standard-library path)
and, when numpy is installed, with numpy on.
"""
import math
import os
import random
import struct
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import skinned_glb  # noqa: E402

try:
    import numpy
except ImportError:  # pragma: no cover - the stdlib-only run is the point of half the tests
    numpy = None


def reference_unpack(data, slot):
    out = bytearray()
    for at in range(0, len(data), 16):
        for word in struct.unpack_from("<4I", data, at):
            index = word & 0xFFF
            if index > 0xFF:
                raise skinned_glb.SkinError(
                    f"skin stream slot {slot} joint index {index} does not fit u8")
            out.append(index)
    return bytes(out)


def reference_normalise(j0, j1, w0, w1, count, local_bones, local_to_joint):
    """The per-vertex loop the column pass replaced (valid streams only)."""
    j0_out, j1_out, w0_out, w1_out = bytearray(), bytearray(), bytearray(), bytearray()
    sum_min, sum_max = math.inf, -math.inf
    for vertex in range(count):
        local = list(j0[4 * vertex:4 * vertex + 4]) + list(j1[4 * vertex:4 * vertex + 4])
        raw = list(w0[4 * vertex:4 * vertex + 4]) + list(w1[4 * vertex:4 * vertex + 4])
        total = sum(raw)
        if total <= 0:
            raise skinned_glb.SkinError("a vertex has no skin weight")
        sum_min, sum_max = min(sum_min, total), max(sum_max, total)
        for slot, weight in enumerate(raw):
            if weight and local[slot] >= local_bones:
                raise skinned_glb.SkinError("a weighted joint index is outside the local palette")
        remapped = [local_to_joint[index] if raw[slot] else 0 for slot, index in enumerate(local)]
        floats = [weight / total for weight in raw]
        j0_out.extend(struct.pack("<4H", *remapped[:4]))
        j1_out.extend(struct.pack("<4H", *remapped[4:]))
        w0_out.extend(struct.pack("<4f", *floats[:4]))
        w1_out.extend(struct.pack("<4f", *floats[4:]))
    return {
        "j0": bytes(j0_out), "j1": bytes(j1_out), "w0": bytes(w0_out), "w1": bytes(w1_out),
        "sum_min": sum_min / 255.0, "sum_max": sum_max / 255.0,
    }


def random_streams(rng, count, local_bones, *, stray_zero_weight_indices=True):
    """Eight influences per vertex; unweighted slots may carry any index byte."""
    rows = {"j0": bytearray(), "j1": bytearray(), "w0": bytearray(), "w1": bytearray()}
    for _ in range(count):
        used = rng.randint(1, 8)
        weights = [rng.randint(1, 255) if slot < used else 0 for slot in range(8)]
        rng.shuffle(weights)
        indices = []
        for weight in weights:
            if weight or not stray_zero_weight_indices:
                indices.append(rng.randrange(min(local_bones, 256)))
            else:
                indices.append(rng.randrange(256))
        rows["j0"].extend(indices[:4])
        rows["j1"].extend(indices[4:])
        rows["w0"].extend(weights[:4])
        rows["w1"].extend(weights[4:])
    return {key: bytes(value) for key, value in rows.items()}


class NormaliseModes(unittest.TestCase):
    """Run one assertion with numpy hidden and, when installed, with numpy on."""

    def modes(self):
        yield "stdlib", mock.patch.object(skinned_glb, "_np", None)
        if numpy is not None:
            yield "numpy", mock.patch.object(skinned_glb, "_np", numpy)

    def call(self, streams, count, local_bones, local_to_joint, joint_count=None):
        return skinned_glb.normalise_weights(
            streams["j0"], streams["j1"], streams["w0"], streams["w1"],
            count, local_bones, local_to_joint, joint_count)


class NormaliseWeightsParityTest(NormaliseModes):
    def check(self, seed, count, local_bones, joint_count=None, mapping=None):
        rng = random.Random(seed)
        streams = random_streams(rng, count, local_bones)
        if mapping is None:
            mapping = [rng.randrange(joint_count or 0x10000) for _ in range(local_bones)]
        expected = reference_normalise(
            streams["j0"], streams["j1"], streams["w0"], streams["w1"],
            count, local_bones, mapping)
        for name, patch in self.modes():
            with patch:
                got = self.call(streams, count, local_bones, mapping, joint_count)
            for key in ("j0", "j1", "w0", "w1"):
                self.assertEqual(got[key], expected[key], f"{name} {key} seed {seed}")
            self.assertEqual(got["sum_min"], expected["sum_min"], name)
            self.assertEqual(got["sum_max"], expected["sum_max"], name)

    def test_random_streams_match_the_per_vertex_reference(self):
        for seed, count, bones in ((1, 1, 1), (2, 7, 3), (3, 500, 40), (4, 257, 255),
                                   (5, 300, 256), (6, 300, 400)):
            with self.subTest(seed=seed):
                self.check(seed, count, bones)

    def test_joint_count_bound_is_respected(self):
        self.check(11, 200, 30, joint_count=1000)

    def test_largest_u16_joint_survives(self):
        self.check(12, 64, 8, mapping=[0xFFFF, 0xFF00, 0x00FF, 0x0100, 1, 2, 3, 4])

    def test_index_zero_mapped_to_a_nonzero_joint_stays_zero_when_unweighted(self):
        # One weight on slot 0; the other seven slots carry index 0, which maps to joint 9.
        j = bytes(4)
        streams = {"j0": j, "j1": j, "w0": bytes([255, 0, 0, 0]), "w1": bytes(4)}
        for name, patch in self.modes():
            with patch:
                got = self.call(streams, 1, 2, [9, 5])
            self.assertEqual(struct.unpack("<4H", got["j0"]), (9, 0, 0, 0), name)
            self.assertEqual(struct.unpack("<4H", got["j1"]), (0, 0, 0, 0), name)
            self.assertEqual(struct.unpack("<4f", got["w0"]), (1.0, 0.0, 0.0, 0.0), name)

    def test_weights_round_once_to_float32_like_struct(self):
        # 1/3, 1/7 and friends differ between float32 rounding and a double -> float32 cast
        # only if the division is not done in double first.
        rng = random.Random(21)
        streams = random_streams(rng, 2000, 5)
        expected = reference_normalise(
            streams["j0"], streams["j1"], streams["w0"], streams["w1"], 2000, 5, [0, 1, 2, 3, 4])
        with mock.patch.object(skinned_glb, "_np", None):
            got = self.call(streams, 2000, 5, [0, 1, 2, 3, 4])
        self.assertEqual(got["w0"], expected["w0"])
        self.assertEqual(got["w1"], expected["w1"])


class NormaliseWeightsErrorTest(NormaliseModes):
    def assert_error(self, streams, count, local_bones, mapping, message):
        # The stdlib path must name the same first error as the per-vertex reference.
        with self.assertRaises(skinned_glb.SkinError) as expected:
            reference_normalise(streams["j0"], streams["j1"], streams["w0"], streams["w1"],
                                count, local_bones, mapping)
        with mock.patch.object(skinned_glb, "_np", None):
            with self.assertRaises(skinned_glb.SkinError) as got:
                self.call(streams, count, local_bones, mapping)
        self.assertEqual(str(got.exception), str(expected.exception))
        self.assertIn(message, str(got.exception))
        if numpy is not None:
            with mock.patch.object(skinned_glb, "_np", numpy):
                with self.assertRaisesRegex(skinned_glb.SkinError, message):
                    self.call(streams, count, local_bones, mapping)

    def test_vertex_without_weight(self):
        streams = {"j0": bytes(8), "j1": bytes(8),
                   "w0": bytes([1, 0, 0, 0, 0, 0, 0, 0]), "w1": bytes(8)}
        self.assert_error(streams, 2, 1, [0], "no skin weight")

    def test_weighted_index_outside_the_palette(self):
        streams = {"j0": bytes([0, 0, 0, 3]), "j1": bytes(4),
                   "w0": bytes([10, 0, 0, 5]), "w1": bytes(4)}
        self.assert_error(streams, 1, 3, [0, 1, 2], "outside the local palette")

    def test_stray_index_on_a_zero_weight_is_not_an_error(self):
        streams = {"j0": bytes([0, 200, 0, 0]), "j1": bytes([255, 0, 0, 0]),
                   "w0": bytes([10, 0, 0, 0]), "w1": bytes(4)}
        for name, patch in self.modes():
            with patch:
                got = self.call(streams, 1, 3, [4, 5, 6])
            self.assertEqual(struct.unpack("<4H", got["j0"]), (4, 0, 0, 0), name)
            self.assertEqual(struct.unpack("<4H", got["j1"]), (0, 0, 0, 0), name)

    def test_stdlib_reports_the_first_failing_vertex(self):
        # Vertex 0 has an outside index, vertex 1 has no weight; the per-vertex order wins.
        streams = {"j0": bytes([9, 0, 0, 0, 0, 0, 0, 0]), "j1": bytes(8),
                   "w0": bytes([4, 0, 0, 0, 0, 0, 0, 0]), "w1": bytes(8)}
        with mock.patch.object(skinned_glb, "_np", None):
            with self.assertRaisesRegex(skinned_glb.SkinError, "outside the local palette"):
                self.call(streams, 2, 2, [0, 1])

    def test_oversized_joint_with_weight_is_rejected_and_unweighted_is_not(self):
        mapping = [0x10000, 1]
        weighted = {"j0": bytes(4), "j1": bytes(4), "w0": bytes([5, 0, 0, 0]), "w1": bytes(4)}
        unweighted = {"j0": bytes(4), "j1": bytes(4), "w0": bytes([0, 5, 0, 0]), "w1": bytes(4)}
        for name, patch in self.modes():
            with patch:
                with self.assertRaisesRegex(skinned_glb.SkinError, "65535"):
                    self.call(weighted, 1, 2, mapping)
                unweighted["j0"] = bytes([0, 1, 0, 0])
                got = self.call(unweighted, 1, 2, mapping)
            self.assertEqual(struct.unpack("<4H", got["j0"]), (0, 1, 0, 0), name)


class UnpackPackedJointsTest(unittest.TestCase):
    def words(self, rng, count, *, high=0xA5CFF000):
        return b"".join(struct.pack("<4I", *(high | rng.randrange(256) for _ in range(4)))
                        for _ in range(count))

    def test_valid_rows_match_the_reference(self):
        rng = random.Random(31)
        for count in (0, 1, 2, 1000):
            data = self.words(rng, count)
            self.assertEqual(skinned_glb._unpack_packed_joints(data, 8),
                             reference_unpack(data, 8))

    def test_bits_12_to_31_are_ignored(self):
        data = struct.pack("<4I", 0xFFFFF0FF, 0x00000000, 0x1234F001, 0x0000F002)
        self.assertEqual(skinned_glb._unpack_packed_joints(data, 8), bytes([0xFF, 0, 1, 2]))

    def test_twelve_bit_index_above_u8_names_the_first_offender(self):
        rng = random.Random(32)
        data = bytearray(self.words(rng, 50))
        struct.pack_into("<I", data, 16 * 7 + 8, 0xABC000 | 0x100)
        struct.pack_into("<I", data, 16 * 9, 0xABC000 | 0xFFF)
        with self.assertRaises(skinned_glb.SkinError) as expected:
            reference_unpack(bytes(data), 9)
        with self.assertRaises(skinned_glb.SkinError) as got:
            skinned_glb._unpack_packed_joints(bytes(data), 9)
        self.assertEqual(str(got.exception), str(expected.exception))
        self.assertIn("joint index 256 does not fit u8", str(got.exception))

    def test_partial_row_still_fails_like_the_reference(self):
        data = self.words(random.Random(33), 3)[:-4]
        with self.assertRaises(struct.error):
            reference_unpack(data, 8)
        with self.assertRaises(struct.error):
            skinned_glb._unpack_packed_joints(data, 8)


class StridedAccessorTest(unittest.TestCase):
    @staticmethod
    def reference(bin_data, start, element, stride, count):
        return b"".join(bin_data[start + i * stride:start + i * stride + element]
                        for i in range(count))

    def test_interleaved_rows_match_per_row_slicing(self):
        rng = random.Random(41)
        blob = bytes(rng.randrange(256) for _ in range(4096))
        for kind, component, element in (("VEC3", 5126, 12), ("VEC2", 5126, 8),
                                         ("VEC4", 5121, 4), ("SCALAR", 5123, 2)):
            for count in (0, 1, 31, 32, 33, 100):
                for stride in (element, element + 4, element + 12):
                    for offset in (0, 3):
                        if offset + (max(count - 1, 0)) * stride + element > len(blob):
                            continue
                        gltf = {
                            "bufferViews": [{"buffer": 0, "byteOffset": offset,
                                             "byteStride": stride, "byteLength": 4000}],
                            "accessors": [{"bufferView": 0, "componentType": component,
                                           "count": count, "type": kind}],
                        }
                        got, _ = skinned_glb.accessor_bytes(gltf, blob, 0)
                        self.assertEqual(
                            got, self.reference(blob, offset, element, stride, count),
                            (kind, count, stride, offset))

    def test_last_row_ending_on_the_last_byte_is_gathered(self):
        count, stride, element = 40, 24, 12
        blob = bytes(range(256)) * 4
        blob = blob[:(count - 1) * stride + element]
        gltf = {
            "bufferViews": [{"buffer": 0, "byteStride": stride, "byteLength": len(blob)}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": count,
                           "type": "VEC3"}],
        }
        got, _ = skinned_glb.accessor_bytes(gltf, blob, 0)
        self.assertEqual(got, self.reference(blob, 0, element, stride, count))
        with self.assertRaisesRegex(skinned_glb.SkinError, "outside the BIN chunk"):
            skinned_glb.accessor_bytes(gltf, blob[:-1], 0)


class PaddedVertexRowsTest(unittest.TestCase):
    @staticmethod
    def reference(payload, element_size):
        padding = bytes((-element_size) % 4)
        return b"".join(payload[i:i + element_size] + padding
                        for i in range(0, len(payload), element_size))

    def test_unaligned_rows_are_padded_like_the_per_row_join(self):
        rng = random.Random(51)
        for kind, component, element in (("VEC2", 5121, 2), ("VEC3", 5121, 3),
                                         ("VEC3", 5123, 6), ("SCALAR", 5121, 1)):
            for count in (1, 31, 32, 33, 200):
                payload = bytes(rng.randrange(256) for _ in range(count * element))
                builder = skinned_glb._GlbBuilder()
                index = builder.accessor(payload, component, kind, count,
                                         skinned_glb.ARRAY_BUFFER)
                view = builder.views[builder.accessors[index]["bufferView"]]
                expected = self.reference(payload, element)
                stride = element + (-element) % 4
                self.assertEqual(view["byteStride"], stride)
                self.assertEqual(bytes(builder.bin[view["byteOffset"]:
                                                   view["byteOffset"] + view["byteLength"]]),
                                 expected, (kind, component, count))

    def test_payload_that_is_not_a_whole_number_of_rows_keeps_the_old_behaviour(self):
        payload = bytes(range(32 * 3 + 1))
        builder = skinned_glb._GlbBuilder()
        index = builder.accessor(payload, 5121, "VEC3", 32, skinned_glb.ARRAY_BUFFER)
        view = builder.views[builder.accessors[index]["bufferView"]]
        self.assertEqual(bytes(builder.bin[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]),
                         self.reference(payload, 3))

    def test_aligned_rows_are_stored_unchanged(self):
        payload = bytes(range(200))
        builder = skinned_glb._GlbBuilder()
        index = builder.accessor(payload, 5126, "VEC2", 25, skinned_glb.ARRAY_BUFFER)
        view = builder.views[builder.accessors[index]["bufferView"]]
        self.assertNotIn("byteStride", view)
        self.assertEqual(bytes(builder.bin[view["byteOffset"]:view["byteOffset"] + 200]), payload)


if __name__ == "__main__":
    unittest.main()

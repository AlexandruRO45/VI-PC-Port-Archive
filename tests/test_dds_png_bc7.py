"""BC7 decoder of tools/dds_png.py against a plain per-pixel reference, from synthetic blocks only
(no game data).

The decoder precomputes every mode's bit layout, interpolates palettes with big-integer lanes and
copies pixels with strided assignments; the reference below reads the fields one by one and
interpolates every pixel, so the two share nothing but the public BPTC tables.

    python -I tests/test_dds_png_bc7.py
"""
import importlib.util
import os
import random
import struct
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "dds_png.py")
    spec = importlib.util.spec_from_file_location("dds_png_bc7_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dds = load_module()

# (subsets, partition bits, rotation bits, selector bits, colour bits, alpha bits,
#  endpoint p-bits, shared p-bits, index bits, second index bits) per mode, from the BC7 spec.
SPEC_MODES = (
    (3, 4, 0, 0, 4, 0, 1, 0, 3, 0),
    (2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
    (3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
    (2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
    (1, 0, 2, 1, 5, 6, 0, 0, 2, 3),
    (1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
    (1, 0, 0, 0, 7, 7, 1, 0, 4, 0),
    (2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
)
SPEC_WEIGHTS = {
    2: (0, 21, 43, 64),
    3: (0, 9, 18, 27, 37, 46, 55, 64),
    4: (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64),
}


def reference_block(block: int) -> bytes:
    """Row-major RGBA8 of one block, decoded pixel by pixel."""
    if block & 0xFF == 0:
        return bytes(64)
    mode = next(m for m in range(8) if block >> m & 1)
    (subsets, partition_bits, rotation_bits, selector_bits, color_bits, alpha_bits,
     endpoint_pbits, shared_pbits, index_bits, index2_bits) = SPEC_MODES[mode]
    cursor = mode + 1

    def read(count):
        nonlocal cursor
        value = (block >> cursor) & ((1 << count) - 1)
        cursor += count
        return value

    partition = read(partition_bits)
    rotation = read(rotation_bits)
    selector = read(selector_bits)
    endpoints = [[read(color_bits) for _ in range(2 * subsets)] for _ in range(3)]
    if alpha_bits:
        endpoints.append([read(alpha_bits) for _ in range(2 * subsets)])
    if endpoint_pbits:
        pbits = [read(1) for _ in range(2 * subsets)]
    elif shared_pbits:
        shared = [read(1) for _ in range(subsets)]
        pbits = [shared[e // 2] for e in range(2 * subsets)]
    else:
        pbits = None

    expanded = []
    for channel, values in enumerate(endpoints):
        precision = alpha_bits if channel == 3 else color_bits
        row = []
        for e, v in enumerate(values):
            if pbits is not None:
                v = (v << 1) | pbits[e]
                precision_e = precision + 1
            else:
                precision_e = precision
            row.append((v << (8 - precision_e)) | (v >> (2 * precision_e - 8)))
        expanded.append(row)

    # Partition tables straight from the public masks.
    if subsets == 1:
        subset_of = [0] * 16
        anchors = {0: [0]}
    elif subsets == 2:
        mask = dds._bptc.BC7_PART2_PACKED[partition]
        subset_of = [(mask >> p) & 1 for p in range(16)]
        anchors = {0: [0], 1: [dds._bptc.BC7_ANCHOR2[partition]]}
    else:
        mask = dds._bptc.BC7_PART3_PACKED[partition]
        subset_of = [(mask >> (2 * p)) & 3 for p in range(16)]
        anchors = {0: [0], 1: [dds._bptc.BC7_ANCHOR3A[partition]],
                   2: [dds._bptc.BC7_ANCHOR3B[partition]]}
    anchor_pixels = {p for plist in anchors.values() for p in plist}

    def read_indices(bits):
        return [read(bits - 1 if p in anchor_pixels else bits) for p in range(16)]

    first = read_indices(index_bits)
    second = read_indices(index2_bits) if index2_bits else None
    if second is None:
        color_idx, color_bits_n, alpha_idx, alpha_bits_n = first, index_bits, first, index_bits
    elif selector:
        color_idx, color_bits_n, alpha_idx, alpha_bits_n = second, index2_bits, first, index_bits
    else:
        color_idx, color_bits_n, alpha_idx, alpha_bits_n = first, index_bits, second, index2_bits

    out = bytearray()
    for p in range(16):
        s = subset_of[p]
        pixel = []
        for channel in range(len(expanded)):
            idx, nbits = (alpha_idx, alpha_bits_n) if channel == 3 else (color_idx, color_bits_n)
            w = SPEC_WEIGHTS[nbits][idx[p]]
            e0, e1 = expanded[channel][2 * s], expanded[channel][2 * s + 1]
            pixel.append((e0 * (64 - w) + e1 * w + 32) >> 6)
        if len(pixel) == 3:
            pixel.append(255)
        if rotation:  # 1, 2, 3 swap alpha with red, green, blue after interpolation
            pixel[rotation - 1], pixel[3] = pixel[3], pixel[rotation - 1]
        out += bytes(pixel)
    return bytes(out)


def reference_image(payload: bytes, width: int, height: int) -> bytes:
    blocks_x, blocks_y = (width + 3) // 4, (height + 3) // 4
    out = bytearray(width * height * 4)
    for by in range(blocks_y):
        for bx in range(blocks_x):
            at = 16 * (by * blocks_x + bx)
            rgba = reference_block(int.from_bytes(payload[at:at + 16], "little"))
            for y in range(4):
                for x in range(4):
                    px, py = bx * 4 + x, by * 4 + y
                    if px < width and py < height:
                        dst = 4 * (py * width + px)
                        out[dst:dst + 4] = rgba[4 * (4 * y + x):4 * (4 * y + x) + 4]
    return bytes(out)


def forced_block(rng, mode, **fields):
    """A random block with the given mode and (optionally) partition/rotation/selector set."""
    value = rng.getrandbits(128)
    spec = SPEC_MODES[mode]
    value &= ~((1 << (mode + 1)) - 1)
    value |= 1 << mode
    cursor = mode + 1
    for name, bits in (("partition", spec[1]), ("rotation", spec[2]), ("selector", spec[3])):
        if name in fields and bits:
            value &= ~(((1 << bits) - 1) << cursor)
            value |= (fields[name] & ((1 << bits) - 1)) << cursor
        cursor += bits
    return value


def random_payload(rng, blocks, duplicate=0.0):
    out = bytearray()
    for _ in range(blocks):
        if out and rng.random() < duplicate:
            out += out[-16:]
        else:
            out += forced_block(rng, rng.randrange(8)).to_bytes(16, "little")
    return bytes(out)


class Bc7BlockTests(unittest.TestCase):
    def test_layout_constants_match_the_spec_table(self):
        self.assertEqual(dds._BC7_MODES, SPEC_MODES)
        self.assertEqual(dds._BC7_WEIGHTS, SPEC_WEIGHTS)

    def test_every_mode_and_partition(self):
        rng = random.Random(0xB7)
        for mode, spec in enumerate(SPEC_MODES):
            for partition in range(1 << spec[1]):
                for _ in range(12):
                    block = forced_block(rng, mode, partition=partition)
                    self.assertEqual(dds._decode_bc7_block(block), reference_block(block),
                                     f"mode {mode} partition {partition} block {block:032x}")

    def test_every_rotation_and_selector(self):
        rng = random.Random(0xB8)
        for mode in (4, 5):
            for rotation in range(4):
                for selector in range(2):
                    for _ in range(40):
                        block = forced_block(rng, mode, rotation=rotation, selector=selector)
                        self.assertEqual(dds._decode_bc7_block(block), reference_block(block),
                                         f"mode {mode} rotation {rotation} selector {selector}")

    def test_random_blocks(self):
        rng = random.Random(0xB9)
        for _ in range(4000):
            block = rng.getrandbits(128)
            self.assertEqual(dds._decode_bc7_block(block), reference_block(block), f"{block:032x}")

    def test_extreme_field_values(self):
        full = (1 << 128) - 1
        for mode in range(8):
            for filler in (0, full):
                block = (filler & ~((1 << (mode + 1)) - 1)) | (1 << mode)
                self.assertEqual(dds._decode_bc7_block(block), reference_block(block),
                                 f"mode {mode} filler {filler:x}")

    def test_reserved_modes_decode_to_transparent_black(self):
        for block in (0, 1 << 8, 1 << 127, 0xFFFF << 8):
            self.assertEqual(dds._decode_bc7_block(block), bytes(64))
            self.assertEqual(reference_block(block), bytes(64))

    def test_anchor_indices_never_use_their_omitted_bit(self):
        # With every stored index bit set, anchor pixels read one bit fewer than the others.
        for mode in range(8):
            spec = SPEC_MODES[mode]
            for partition in range(1 << spec[1]):
                block = ((1 << 128) - 1) & ~((1 << (mode + 1)) - 1) | (1 << mode)
                if spec[1]:
                    block &= ~(((1 << spec[1]) - 1) << (mode + 1))
                    block |= partition << (mode + 1)
                self.assertEqual(dds._decode_bc7_block(block), reference_block(block),
                                 f"mode {mode} partition {partition}")


class Bc7ImageTests(unittest.TestCase):
    def check(self, width, height, *, seed, duplicate=0.0):
        rng = random.Random(seed)
        blocks = ((width + 3) // 4) * ((height + 3) // 4)
        payload = random_payload(rng, blocks, duplicate)
        got = dds._decode_bc7(payload, 0, width, height)
        self.assertEqual(len(got), width * height * 4)
        self.assertEqual(got, reference_image(payload, width, height), f"{width}x{height}")

    def test_aligned_and_unaligned_sizes(self):
        sizes = [(1, 1), (2, 2), (3, 5), (4, 4), (5, 3), (7, 9), (13, 2), (16, 16), (17, 1),
                 (1, 17), (33, 17), (130, 6), (4, 33), (31, 31), (64, 8)]
        for index, (width, height) in enumerate(sizes):
            self.check(width, height, seed=100 + index)

    def test_repeated_blocks_use_the_block_cache_correctly(self):
        self.check(64, 64, seed=1, duplicate=0.8)
        self.check(37, 21, seed=2, duplicate=0.5)

    def test_more_unique_blocks_than_the_cache_holds(self):
        side = 4 * 66  # 66 * 66 = 4356 blocks, above the 4096-entry cache
        self.assertGreater((side // 4) ** 2, dds._BC7_BLOCK_CACHE_LIMIT)
        rng = random.Random(3)
        blocks = bytearray(random_payload(rng, (side // 4) ** 2))
        # Repeat an early (cached) block at the end, and a late (uncached) block just before it.
        blocks[-16:] = blocks[:16]
        blocks[-32:-16] = blocks[16 * 4200:16 * 4201]
        payload = bytes(blocks)
        got = dds._decode_bc7(payload, 0, side, side)
        self.assertEqual(got, reference_image(payload, side, side))

    def test_offset_and_buffer_types(self):
        rng = random.Random(4)
        payload = random_payload(rng, 6)
        want = reference_image(payload, 11, 6)
        for wrap in (bytes, bytearray, memoryview):
            data = wrap(b"\x55" * 9 + payload + b"\xaa" * 5)
            self.assertEqual(dds._decode_bc7(data, 9, 11, 6), want, wrap.__name__)

    def test_truncated_payload(self):
        payload = random_payload(random.Random(5), 3)
        with self.assertRaises(dds.DdsError):
            dds._decode_bc7(payload, 0, 12, 4 + 1)
        with self.assertRaises(dds.DdsError):
            dds._decode_bc7(payload[:-1], 0, 12, 4)

    def test_dds_level_decode(self):
        rng = random.Random(6)
        width, height = 10, 6
        payload = random_payload(rng, 3 * 2)
        header = bytearray(128)
        header[0:4] = b"DDS "
        struct.pack_into("<I", header, 4, 124)
        struct.pack_into("<I", header, 12, height)
        struct.pack_into("<I", header, 16, width)
        struct.pack_into("<I", header, 28, 1)
        struct.pack_into("<I", header, 76, 32)
        struct.pack_into("<I", header, 80, 0x4)
        header[84:88] = b"DX10"
        data = bytes(header) + struct.pack("<IIIII", 98, 3, 0, 1, 0) + payload
        w, h, rgba = dds.decode_dds(data)
        self.assertEqual((w, h), (width, height))
        self.assertEqual(rgba, reference_image(payload, width, height))


if __name__ == "__main__":
    unittest.main()

"""image_bvh64_intersect_ray on box nodes: a child whose box is empty (min > max) is a miss.

A builder marks an inactive child (NaN triangle, mask-0 instance) with an empty box. The slab test alone treats
min > max on every axis as a box spanning everything, so the child behind it was still returned. The CPU reference
tools/xdxr_bvh.py `_box_test` returns a miss first (TestXboxBvh4Node). One pixel shader per case runs the instruction
on a hand-built box32 or box16 node and exports the four child ids (0xFFFFFFFF = miss). The ray starts at (0.5, 0.5, -1)
and runs along +z. Child 0 is the box [0,1]^3, which the ray enters; the others are as each test says.

    python -m unittest tests/semantic/test_image_bvh_empty_box.py -v
"""
import struct
import unittest

from support import mrt0_words, translate_lines

MISS = 0xFFFFFFFF
NODE = 0x1040
CHILD = (0x11, 0x12, 0x13, 0x14)
HIT = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
EMPTY = ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))         # min > max on every axis
EMPTY_X = ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))       # min > max on x only
EMPTY_Z = ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0))       # min > max on z only, the ray's axis
OFF = ((5.0, 5.0, 5.0), (6.0, 6.0, 6.0))           # a real box the ray misses


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def h(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def run(boxes, b16):
    lines = []
    mov = lambda r, v: lines.append(f"v_mov_b32_e32 v{r}, 0x{v:x}")   # noqa: E731
    mov(1, 0x08 | (4 if b16 else 5))
    mov(2, 0)
    mov(3, f2u(100.0))
    for i, v in enumerate([f2u(0.5), f2u(0.5), f2u(-1.0), 0, 0, f2u(1.0), 0x7F800000, 0x7F800000, f2u(1.0)]):
        mov(4 + i, v)
    lines += ["s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0", "s_mov_b32 s6, -1", "s_mov_b32 s7, 0",
              "image_bvh64_intersect_ray v[20:23], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]",
              "exp mrt0 v20, v21, v22, v23 done vm", "s_endpgm"]
    spv = translate_lines("ps_bvh_empty_box", lines)
    mem = {NODE + 4 * i: CHILD[i] for i in range(4)}
    for i, (lo, hi) in enumerate(boxes):
        if b16:
            hv = [h(x) for x in lo + hi]                  # min x,y,z then max x,y,z
            for w in range(3):
                mem[NODE + 16 + 12 * i + 4 * w] = hv[2 * w] | (hv[2 * w + 1] << 16)
        else:
            for c, x in enumerate(lo + hi):
                mem[NODE + 16 + 24 * i + 4 * c] = f2u(x)
    return mrt0_words(spv, memory=mem)


class EmptyBox(unittest.TestCase):
    def check(self, b16):
        with self.subTest(b16=b16):
            self.assertEqual(run([HIT, OFF, OFF, OFF], b16), [CHILD[0], MISS, MISS, MISS])   # control
            self.assertEqual(run([HIT, EMPTY, OFF, OFF], b16), [CHILD[0], MISS, MISS, MISS])
            self.assertEqual(run([EMPTY, HIT, EMPTY_X, EMPTY_Z], b16), [MISS, CHILD[1], MISS, MISS])

    def test_box32(self):
        self.check(False)

    def test_box16(self):
        self.check(True)


if __name__ == "__main__":
    unittest.main()

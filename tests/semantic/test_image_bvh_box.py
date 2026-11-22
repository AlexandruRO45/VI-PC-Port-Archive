"""image_bvh64_intersect_ray on box nodes: the children a box16 and a box32 node return, and that each reads only its
own bytes (a box16 node is 64 bytes, a box32 node 128).

One pixel shader per case runs the instruction on a hand-built node and exports the four result words. The node's
children are four ids; their boxes put child 1 at z 2..3 and child 0 at z 5..6 on the ray from (0.5, 0.5, 0) along +z,
child 2 beside the ray and child 3 behind its origin. With the descriptor's sort bit set the result is child 1,
child 0 (by entry distance), then two misses (0xFFFFFFFF). Memory reads outside the node fail the test: the box16
path used to load the box32 words too, up to byte 112, past the end of a 64-byte node (off the end of the buffer for
a BVH's last node).

    python -m unittest tests/semantic/test_image_bvh_box.py -v
"""
import struct
import unittest

from support import mrt0_words, translate_lines

NODE = 0x1000                     # descriptor base 0x10 (256-byte units) = 0x1000; node id 0x00 | type
CHILDREN = [0x200, 0x208, 0x210, 0x218]
BOXES = [((0, 0, 5), (1, 1, 6)), ((0, 0, 2), (1, 1, 3)), ((4, 0, 0), (5, 1, 1)), ((0, 0, -3), (1, 1, -2))]
MISS = 0xFFFFFFFF


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def h2u(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


class NodeMemory(dict):
    """The node's words; a read of any other address fails (the interpreter reads with .get)."""

    def __init__(self, words, size):
        super().__init__(words)
        self.size = size

    def get(self, addr, default=0):
        if not NODE <= addr < NODE + self.size:
            raise AssertionError(f"read at {addr:#x}, outside the {self.size}-byte node at {NODE:#x}")
        return super().get(addr, default)


def node(box16):
    words = {NODE + 4 * i: c for i, c in enumerate(CHILDREN)}
    for i, (lo, hi) in enumerate(BOXES):
        if box16:
            h = [h2u(v) for v in (*lo, *hi)]
            for k in range(3):
                words[NODE + 16 + 12 * i + 4 * k] = h[2 * k] | h[2 * k + 1] << 16
        else:
            for k, v in enumerate((*lo, *hi)):
                words[NODE + 16 + 24 * i + 4 * k] = f2u(v)
    return NodeMemory(words, 64 if box16 else 128)


def run(box16):
    lines = []
    mov = lambda r, v: lines.append(f"v_mov_b32_e32 v{r}, 0x{v:x}")   # noqa: E731
    mov(1, 4 if box16 else 5)                                          # node id: offset 0, type box16 / box32
    mov(2, 0)
    mov(3, f2u(100.0))                                                 # ray extent
    for i, v in enumerate([f2u(0.5), f2u(0.5), 0, 0, 0, f2u(1.0), 0x7F7FFFFF, 0x7F7FFFFF, f2u(1.0)]):
        mov(4 + i, v)                                                  # origin, direction, inverse direction
    lines += ["s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0x80000000", "s_mov_b32 s6, -1", "s_mov_b32 s7, 0",
              "image_bvh64_intersect_ray v[20:23], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]",
              "exp mrt0 v20, v21, v22, v23 done vm", "s_endpgm"]
    spv = translate_lines(f"ps_bvh_box{16 if box16 else 32}", lines)
    return mrt0_words(spv, memory=node(box16))


class BoxNodes(unittest.TestCase):
    def test_box16_reads_only_its_64_bytes(self):
        self.assertEqual(run(True), [CHILDREN[1], CHILDREN[0], MISS, MISS])

    def test_box32_reads_only_its_128_bytes(self):
        self.assertEqual(run(False), [CHILDREN[1], CHILDREN[0], MISS, MISS])


if __name__ == "__main__":
    unittest.main()

"""image_store with a texel coordinate outside the image: RDNA2 drops the write, so the translation must too.

Measured natively on gfx1036 (RDNA2) with tools/hw_oracle on a linear 4x4 R32_UINT T#: image_store at x = width, x = -1,
y = height, y = -1, x = the row pitch and x = 1000 left every seeded word of the image, its row padding and the bytes
after it unchanged. An OpImageWrite outside the image is undefined in Vulkan (the runtime does not enable
robustImageAccess), so every write must sit behind a branch on "coordinate < OpImageQuerySize" per axis, unsigned so
that -1 is out too, together with the EXEC test.

Standard library only; no GPU, no LLVM, no game files.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_image_atomics_32 import check_cfg, translate_cs  # noqa: E402

LINE = re.compile(r"^\s*(?:(%\w+) = )?(Op\w+)(.*)$")
DIMS = {"SQ_RSRC_IMG_1D": ("v4", 1), "SQ_RSRC_IMG_2D": ("v[4:5]", 2),
        "SQ_RSRC_IMG_2D_ARRAY": ("v[4:6]", 3), "SQ_RSRC_IMG_3D": ("v[4:6]", 3)}


def parse(spv):
    out, body = [], False
    for raw in spv.splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        res, op, rest = m.groups()
        body = body or op == "OpFunction"
        if body:
            out.append((res, op, rest.split()))
    return out


class ImageStoreBoundsTests(unittest.TestCase):
    def test_every_write_is_behind_a_size_test(self):
        for dim, (addr, n) in DIMS.items():
            with self.subTest(dim=dim):
                spv = translate_cs(["v_mov_b32_e32 v4, 1", "v_mov_b32_e32 v5, 2", "v_mov_b32_e32 v6, 3",
                                    "v_mov_b32_e32 v8, 7",
                                    f"image_store v8, {addr}, s[4:11] dmask:0x1 dim:{dim} unorm", "s_waitcnt vmcnt(0)"])
                check_cfg(spv)
                lines = parse(spv)
                defs = {r: (o, a) for r, o, a in lines if r}
                writes = [i for i, (_r, o, _a) in enumerate(lines) if o == "OpImageWrite"]
                self.assertEqual(len(writes), 3)       # the sint, uint and float views
                sizes = [r for r, (o, _a) in defs.items() if o == "OpImageQuerySize"]
                self.assertEqual(len(sizes), 1, "no OpImageQuerySize before the writes")
                # the outermost guard: the first conditional branch after the size query
                q = next(i for i, (r, _o, _a) in enumerate(lines) if r == sizes[0])
                first = next(i for i in range(q, len(lines)) if lines[i][1] == "OpBranchConditional")
                seen, raw, todo = set(), set(), [lines[first][2][0]]
                while todo:
                    v = todo.pop()
                    raw.add(v)
                    if v in seen or v not in defs:
                        continue
                    seen.add(v)
                    todo += defs[v][1]
                # the EXEC test (a kernel that never changes EXEC folds it to the constant %true)
                self.assertTrue(raw & {"%active", "%true"} or any(defs[v][0] == "OpLoad" and defs[v][1][1] == "%active"
                                                                  for v in seen))
                if n == 1:
                    tests = [v for v in seen if defs[v][0] == "OpULessThan" and defs[v][1][2] == sizes[0]]
                    self.assertTrue(tests, "1D: coordinate < size missing")
                else:
                    for ax in range(n):
                        tests = [v for v in seen if defs[v][0] == "OpULessThan"
                                 and defs[defs[v][1][2]] == ("OpCompositeExtract", ["%int", sizes[0], str(ax)])]
                        self.assertTrue(tests, f"axis {ax}: coordinate < size (unsigned) missing")


if __name__ == "__main__":
    unittest.main()

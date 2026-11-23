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
    def test_every_write_is_guarded_by_its_typed_view_size(self):
        for dim, (addr, n) in DIMS.items():
            with self.subTest(dim=dim):
                spv = translate_cs(["v_mov_b32_e32 v4, 1", "v_mov_b32_e32 v5, 2", "v_mov_b32_e32 v6, 3",
                                    "v_mov_b32_e32 v8, 7",
                                    f"image_store v8, {addr}, s[4:11] dmask:0x1 dim:{dim} unorm", "s_waitcnt vmcnt(0)"])
                check_cfg(spv)
                lines = parse(spv)
                defs = {r: (o, a) for r, o, a in lines if r}
                writes = [a[0] for _r, o, a in lines if o == "OpImageWrite"]
                self.assertEqual(len(writes), 3)       # one for each typed storage view
                sizes = [r for r, (o, _a) in defs.items() if o == "OpImageQuerySize"]
                self.assertEqual(len(sizes), 3)        # integer views may not share the float view's extent

                blocks, current = {}, None
                for r, op, args in lines:
                    if op == "OpLabel":
                        current = r
                        blocks[current] = []
                    elif current is not None:
                        blocks[current].append((r, op, args))

                def dependencies(root):
                    seen, todo = set(), [root]
                    while todo:
                        value = todo.pop()
                        if value in seen:
                            continue
                        seen.add(value)
                        if value in defs:
                            todo.extend(arg for arg in defs[value][1] if arg.startswith("%"))
                    return seen

                matched_queries = set()
                for image in writes:
                    self.assertEqual(defs[image][0], "OpLoad")
                    write_type, write_pointer = defs[image][1][:2]
                    self.assertEqual(defs[write_pointer][0], "OpAccessChain")
                    write_index = defs[write_pointer][1][-1]
                    matching = []
                    for query in sizes:
                        query_image = defs[query][1][1]
                        self.assertEqual(defs[query_image][0], "OpLoad")
                        query_type, query_pointer = defs[query_image][1][:2]
                        if query_type == write_type:
                            self.assertEqual(defs[query_pointer][0], "OpAccessChain")
                            self.assertEqual(defs[query_pointer][1][-1], write_index)
                            matching.append(query)
                    self.assertEqual(len(matching), 1, "each write must query its own typed descriptor")
                    query = matching[0]
                    matched_queries.add(query)

                    guarded = False
                    for i, (_r, op, args) in enumerate(lines):
                        if op != "OpBranchConditional" or query not in dependencies(args[0]):
                            continue
                        true_block, false_block = blocks.get(args[1], []), blocks.get(args[2], [])
                        if not any(o == "OpImageWrite" and a[0] == image for _d, o, a in true_block):
                            continue
                        self.assertFalse(any(o == "OpImageWrite" and a[0] == image for _d, o, a in false_block))
                        self.assertIn("%active", dependencies(args[0]))
                        tests = [v for v in dependencies(args[0]) if v in defs and defs[v][0] == "OpULessThan"]
                        self.assertEqual(len(tests), n)
                        for ax in range(n):
                            if n == 1:
                                size_component = query
                            else:
                                extracts = [r for r, (o, a) in defs.items()
                                            if o == "OpCompositeExtract" and a[1:] == [query, str(ax)]]
                                self.assertEqual(len(extracts), 1)
                                size_component = extracts[0]
                            self.assertTrue(any(defs[v][1][2] == size_component for v in tests),
                                            f"axis {ax}: coordinate < typed-view size missing")
                        self.assertLess(next(j for j, (r, _o, _a) in enumerate(lines) if r == query), i)
                        guarded = True
                        break
                    self.assertTrue(guarded, "OpImageWrite is not guarded by its typed-view size")
                self.assertEqual(matched_queries, set(sizes))


if __name__ == "__main__":
    unittest.main()

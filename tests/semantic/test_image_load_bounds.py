"""image_load with a texel coordinate outside the image returns 0 on RDNA2, and so must the translation.

Measured natively on gfx1036 (RDNA2) with tools/hw_oracle on a linear 4x4 R32_UINT T#: image_load at x = width, x = -1,
y = height and x = the row pitch returned 0 (the in-range control returned the texel). OpImageFetch and OpImageRead
outside the image are undefined in Vulkan (the runtime does not enable robustImageAccess), so every read must sit behind
a branch on "coordinate < OpImageQuerySize(Lod)" per axis, unsigned so that -1 is out too, and a lane outside gets a
zero vector through an OpPhi. A descriptor lookup miss (index 0) keeps reading the fallback image unguarded.

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


class ImageLoadBoundsTests(unittest.TestCase):
    def check(self, extra, reads_op, size_op, dim, n):
        addr = DIMS[dim][0]
        spv = translate_cs(["v_mov_b32_e32 v4, 1", "v_mov_b32_e32 v5, 2", "v_mov_b32_e32 v6, 3",
                            f"image_load v[8:11], {addr}, s[4:11] dmask:0xf dim:{dim} unorm{extra}",
                            "s_waitcnt vmcnt(0)"])
        check_cfg(spv)
        lines = parse(spv)
        defs = {r: (o, a) for r, o, a in lines if r}
        reads = [i for i, (_r, o, _a) in enumerate(lines) if o == reads_op]
        self.assertEqual(len(reads), 3)                    # the sint, uint and float views
        for i in reads:
            # the branch that opens the block holding the read
            b = max(j for j in range(i) if lines[j][1] == "OpBranchConditional")
            self.assertEqual(lines[b + 1][0], lines[b][2][1])
            cond, merge = lines[b][2][0], lines[b][2][2]
            seen, todo = set(), [cond]
            while todo:
                v = todo.pop()
                if v in seen or v not in defs:
                    continue
                seen.add(v)
                todo += defs[v][1]
            sizes = [v for v in seen if defs[v][0] == size_op]
            self.assertEqual(len(sizes), 1, f"{size_op} missing from the guard")
            for ax in range(n):
                if n == 1:
                    tests = [v for v in seen if defs[v][0] == "OpULessThan" and defs[v][1][2] == sizes[0]]
                else:
                    tests = [v for v in seen if defs[v][0] == "OpULessThan"
                             and defs[defs[v][1][2]] == ("OpCompositeExtract", ["%int", sizes[0], str(ax)])]
                self.assertTrue(tests, f"axis {ax}: coordinate < size (unsigned) missing")
            # the merge block starts with the OpPhi that gives a lane outside the image its zeros
            at = next(j for j, (r, o, _a) in enumerate(lines) if r == merge and o == "OpLabel")
            phi_res, phi_op, phi_args = lines[at + 1]
            self.assertEqual(phi_op, "OpPhi")
            zero = defs[phi_args[3]]
            self.assertEqual(zero[0], "OpCompositeConstruct")
            self.assertEqual(len(set(zero[1][1:])), 1, "all four components are the same zero")

    def test_sampled_fetch_is_behind_a_size_test(self):
        for dim, (_addr, n) in DIMS.items():
            for extra in ("", " d16"):
                with self.subTest(dim=dim, extra=extra):
                    self.check(extra, "OpImageFetch", "OpImageQuerySizeLod", dim, n)

    def test_coherent_storage_read_is_behind_a_size_test(self):
        for dim, (_addr, n) in DIMS.items():
            with self.subTest(dim=dim):
                self.check(" glc", "OpImageRead", "OpImageQuerySize", dim, n)


if __name__ == "__main__":
    unittest.main()

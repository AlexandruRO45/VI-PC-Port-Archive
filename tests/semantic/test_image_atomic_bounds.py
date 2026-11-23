"""Image atomics with a texel coordinate outside the image: RDNA2 skips them, and so must the translation.

Measured natively on gfx1036 (RDNA2) with tools/hw_oracle on a linear 4x4 R32_UINT T# (docs/design.md, "Image
atomics, size and LOD queries"): image_atomic_{umax,umin,or,add,swap,cmpswap} at x = width, x = -1, y = height, y = -1,
x = the row pitch and x = 1000 leave every word of the image, its row padding and the bytes after it untouched, and
return 0 with glc. In Vulkan an OpImageTexelPointer outside the image is undefined, so the translation has to test the
coordinate itself: every texel pointer must sit behind a branch on "x < width" and "y < height" (and "z < depth"),
unsigned so that -1 is out too, against OpImageQuerySize of the same image, and with glc a lane that skipped gets 0.

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

# op -> (vdata, dmask)
OPS = {op: ("v2", "0x1") for op in (
    "image_atomic_add", "image_atomic_sub", "image_atomic_and", "image_atomic_or", "image_atomic_xor",
    "image_atomic_smin", "image_atomic_smax", "image_atomic_umin", "image_atomic_umax", "image_atomic_swap",
    "image_atomic_inc", "image_atomic_dec", "image_atomic_fmin", "image_atomic_fmax")}
OPS.update({"image_atomic_cmpswap": ("v[2:3]", "0x3"), "image_atomic_fcmpswap": ("v[2:3]", "0x3")})
DIMS = {"SQ_RSRC_IMG_2D": ("v[4:5]", 2), "SQ_RSRC_IMG_3D": ("v[4:6]", 3)}


def parse(spv):
    """[(result id or None, opcode, operand list)] for the function body."""
    out = []
    body = False
    for raw in spv.splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        res, op, rest = m.groups()
        if op == "OpFunction":
            body = True
        if body:
            out.append((res, op, rest.split()))
    return out


def translate(op, dim, glc):
    vdata, dmask = OPS[op]
    addr, _ = DIMS[dim]
    return translate_cs([
        "v_mov_b32_e32 v2, 7",
        "v_mov_b32_e32 v3, 9",
        "v_mov_b32_e32 v4, 1",
        "v_mov_b32_e32 v5, 2",
        "v_mov_b32_e32 v6, 3",
        f"{op} {vdata}, {addr}, s[4:11] dmask:{dmask} dim:{dim} unorm" + (" glc" if glc else ""),
        "s_waitcnt vmcnt(0)",
    ])


class ImageAtomicBoundsTests(unittest.TestCase):
    def guard_of(self, lines, n):
        """Check that the one texel pointer is guarded by the coordinate test; return (pointer index, merge label)."""
        defs = {res: (op, args) for res, op, args in lines if res}
        tps = [i for i, (_r, op, _a) in enumerate(lines) if op == "OpImageTexelPointer"]
        self.assertEqual(len(tps), 1)
        tp = tps[0]
        _ptr_ty, image_ptr, coord, _sample = lines[tp][2]
        # the image's size, queried through the same descriptor the texel pointer uses
        sizes = [res for res, (op, args) in defs.items() if op == "OpImageQuerySize"
                 and defs.get(args[1], ("", []))[0] == "OpLoad" and defs[args[1]][1][1] == image_ptr]
        self.assertEqual(len(sizes), 1, "no OpImageQuerySize of the atomic's image")
        size = sizes[0]
        # the branch that leads to the pointer, and every value its condition is built from
        branch = next(i for i in range(tp, -1, -1) if lines[i][1] == "OpBranchConditional")
        cond, then_lbl, merge_lbl = lines[branch][2]
        self.assertEqual(lines[branch + 1][0], then_lbl)
        seen, todo = set(), [cond]
        while todo:
            v = todo.pop()
            if v in seen or v not in defs:
                continue
            seen.add(v)
            todo += defs[v][1]
        for axis in range(n):
            tests = [v for v in seen if defs[v][0] == "OpULessThan" and
                     defs[defs[v][1][1]] == ("OpCompositeExtract", ["%int", coord, str(axis)]) and
                     defs[defs[v][1][2]] == ("OpCompositeExtract", ["%int", size, str(axis)])]
            self.assertTrue(tests, f"axis {axis}: the branch does not test coordinate < size (unsigned)")
        # the EXEC test is the branch around this one: an inactive lane never reaches the descriptor lookup
        outer = next(i for i in range(branch - 1, -1, -1) if lines[i][1] == "OpBranchConditional"
                     and lines[i][2][2] not in (merge_lbl,))
        outer_seen, todo = set(), [lines[outer][2][0]]
        while todo:
            v = todo.pop()
            if v in outer_seen or v not in defs:
                continue
            outer_seen.add(v)
            todo += defs[v][1]
        loads = [res for res in outer_seen if defs[res][0] == "OpLoad" and defs[res][1][1] == "%active"]
        # (a kernel that never changes EXEC folds the test to the constant %true)
        self.assertTrue(loads or lines[outer][2][0] == "%true", "the branch around the bounds test must be the EXEC test")
        return tp, branch, outer

    def test_every_image_atomic_is_skipped_outside_the_image(self):
        for op in OPS:
            for dim, (_addr, n) in DIMS.items():
                for glc in (False, True):
                    with self.subTest(op=op, dim=dim, glc=glc):
                        spv = translate(op, dim, glc)
                        check_cfg(spv)
                        lines = parse(spv)
                        tp, branch, outer = self.guard_of(lines, n)
                        self.assertLess(outer, branch)
                        # with glc a lane that is active but outside the image gets 0 in vdata: written before
                        # the bounds branch, overwritten by the old value inside it
                        zero_stores = [i for i in range(outer, branch)
                                       if lines[i][1] == "OpStore" and lines[i][2][1] == "%c0"]
                        if glc:
                            self.assertTrue(zero_stores, "glc must write 0 to vdata before the bounds branch")
                        else:
                            self.assertFalse(zero_stores)


if __name__ == "__main__":
    unittest.main()

"""wave64_halves: the per-opcode caches behind operands() / base_op() change speed, not results.

`operands` used to run a chain of str.startswith tests on every call (it was a third of an analysis); it now reads
what the opcode name alone decides from a small cache (`_op_class`). The tests pin that to a literal copy of the
previous implementation over every opcode the module names plus hand-built operand shapes, and check that the
caches are bounded. Hand-written instructions only; no game data, llvm-mc or GPU.
"""
import itertools
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
import wave64_halves as wh  # noqa: E402


def old_base_op(op):
    for suffix in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(suffix):
            return op[:-len(suffix)]
    return op


def old_operands(ins):
    """operands() as it was before the opcode classification was cached (verbatim)."""
    op, a = old_base_op(ins["op"]), ins["args"]
    mods = ins.get("mods") or []
    defs, uses = [], []
    if op.startswith(wh.NO_DEST) and not op.startswith(("s_cmp", "s_bitcmp", "v_cmpx")) or op in ("s_endpgm", "s_barrier"):
        dests, srcs = [], list(a)
    elif op.startswith(("s_cmp", "s_bitcmp")):
        dests, srcs = [], list(a)
        defs.append("scc")
    elif op.startswith("v_cmpx"):
        dests, srcs = [], list(a)
        defs.append("exec")
    elif op.startswith("v_cmp"):
        if len(a) == 2:
            dests, srcs = [], list(a)
            defs.append("vcc")
        else:
            dests, srcs = [a[0]], list(a[1:])
    elif op.startswith("buffer_atomic") or op.startswith(("global_atomic", "flat_atomic")):
        dests = [a[0]] if "glc" in mods and a else []
        srcs = list(a)
    elif op.startswith("ds_"):
        returns = op.startswith(("ds_read", "ds_bpermute", "ds_permute", "ds_swizzle", "ds_consume", "ds_append")) \
            or "_rtn" in op
        dests, srcs = ([a[0]], list(a[1:])) if returns and a else ([], list(a))
    elif op == "v_swap_b32" and len(a) >= 2:
        dests, srcs = [a[0], a[1]], [a[0], a[1]]
    elif a:
        if op in wh.RMW_DEST:
            dests, srcs = [a[0]], [a[0], *a[1:]]
        else:
            dests, srcs = [a[0]], list(a[1:])
        if op.startswith("v_") and len(a) > 2 and (
                "_co_" in op or op in ("v_mad_u64_u32", "v_mad_i64_i32", "v_div_scale_f32", "v_div_scale_f64")) \
                and a[1].get("k") in ("sgpr", "special", "slist"):
            dests, srcs = [a[0], a[1]], list(a[2:])
    else:
        dests, srcs = [], []
    if any(op.startswith(p) for p in wh.SAVEEXEC):
        defs += ["exec", "scc"]
    if op.startswith("s_") and not op.startswith(wh.SCC_LESS) and "scc" not in defs and not op.startswith("s_cbranch"):
        defs.append("scc")
    if op in ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32",
              "v_addc_co_u32", "v_subb_co_u32", "v_subbrev_co_u32",
              "v_cndmask_b32", "v_addc_u32", "v_subb_u32") and len(srcs) == 2:
        uses.append("vcc")
    if op in ("v_div_fmas_f32", "v_div_fmas_f64") and "vcc" not in [r for x in srcs for r in wh._regs(x)]:
        uses.append("vcc")
    if op in ("s_cselect_b32", "s_cselect_b64", "s_addc_u32", "s_subb_u32", "s_cmov_b32", "s_cmov_b64",
              "s_cmovk_i32", "s_cbranch_scc0", "s_cbranch_scc1"):
        uses.append("scc")
    if op in ("s_cbranch_vccz", "s_cbranch_vccnz"):
        uses.append("vcc")
    if op in wh.MOVREL_OPS:
        uses.append("m0")
    return dests, srcs, defs, uses


def outcome(fn, ins):
    """The result, or the exception type (a v_cmp without operands raised IndexError before and still does)."""
    try:
        return fn(ins)
    except Exception as exc:   # noqa: BLE001
        return type(exc)


def opcodes():
    names = set(wh.NO_DEST) | set(wh.LOADS) | set(wh.SINKS) | set(wh.SCC_LESS) | set(wh.SAVEEXEC) | set(wh.RMW_DEST) \
        | set(wh.MOVREL_OPS) | set(wh.BRANCHES) | {
            "s_cmp_eq_u32", "s_bitcmp1_b32", "v_cmpx_lt_f32", "v_cmp_lt_f32", "v_cmp_class_f32", "v_cndmask_b32",
            "v_add_co_ci_u32", "v_add_co_u32", "v_sub_co_ci_u32", "v_mad_u64_u32", "v_div_scale_f32",
            "v_div_fmas_f32", "v_div_fmas_f64", "v_swap_b32", "v_mov_b32", "v_add_f32", "v_fmac_f32",
            "s_and_b64", "s_or_b32", "s_cselect_b64", "s_addc_u32", "s_cmov_b64", "s_endpgm", "s_barrier",
            "buffer_atomic_add", "global_atomic_add", "flat_atomic_swap", "ds_read_b32", "ds_write_b32",
            "ds_add_rtn_u32", "ds_bpermute_b32", "ds_swizzle_b32", "s_cbranch_vccz", "s_cbranch_scc1",
            "s_mov_b64", "s_bcnt1_i32_b64", "v_readlane_b32", "v_writelane_b32", "exp", "s_getpc_b64"}
    out = set()
    for n in names:
        for suffix in ("", "_e32", "_e64", "_sdwa", "_dpp"):
            out.add(n + suffix)
    return sorted(out)


def sg(r, n=1):
    return {"k": "sgpr", "r": r, "n": n}


def vg(r, n=1):
    return {"k": "vgpr", "r": r, "n": n}


SHAPES = [
    [],
    [vg(1)],
    [sg(2), vg(3)],
    [vg(1), vg(2), vg(3)],
    [vg(1), sg(4, 2), vg(5), vg(6)],
    [vg(1), {"k": "special", "t": "vcc"}, vg(5), vg(6)],
    [vg(1), {"k": "slist", "r": [4, "vcc_lo"]}, vg(5), vg(6)],
    [vg(1), {"k": "special", "t": "vcc"}, vg(5)],
    [vg(1), {"k": "imm", "v": 3}, vg(5), {"k": "special", "t": "vcc"}],
    [sg(2, 2), {"k": "special", "t": "exec"}, sg(6, 2)],
    [{"k": "special", "t": "vcc"}, vg(5)],
    [vg(1), vg(2), vg(3), vg(4), vg(5)],
]


class OperandsEquivalence(unittest.TestCase):
    def test_operands_matches_the_previous_implementation(self):
        n = 0
        for op, args, mods in itertools.product(opcodes(), SHAPES, ([], ["glc"], ["clamp"])):
            ins = {"op": op, "args": args, "mods": mods}
            self.assertEqual(outcome(wh.operands, ins), outcome(old_operands, ins), (op, args, mods))
            n += 1
        self.assertGreater(n, 5000)

    def test_operands_returns_fresh_lists(self):
        ins = {"op": "s_and_saveexec_b64", "args": [sg(2, 2), sg(4, 2)]}
        first = wh.operands(ins)
        for part in first:
            part.append("junk")
        self.assertEqual(wh.operands(ins), old_operands(ins))

    def test_base_op_matches(self):
        for op in opcodes() + ["", "_e32", "v_x_e32_e64", "s_nop"]:
            self.assertEqual(wh.base_op(op), old_base_op(op), op)

    def test_regs_matches_for_every_register_width(self):
        for k, key in (("sgpr", "s"), ("vgpr", "v")):
            for n in (0, 1, 2, 3, 4, 8):
                a = {"k": k, "r": 7, "n": n}
                self.assertEqual(wh._regs(a), [(key, 7 + i) for i in range(n)])
            self.assertEqual(wh._regs({"k": k, "r": 7}), [(key, 7)])


class CachesAreBounded(unittest.TestCase):
    def test_cache_sizes_are_capped(self):
        for fn in (wh.base_op, wh._op_class):
            self.assertLessEqual(fn.cache_info().maxsize, 4096)

    def test_many_distinct_opcodes_do_not_grow_the_caches(self):
        for i in range(5000):
            op = f"v_fake_{i}_e32"
            ins = {"op": op, "args": [vg(1), vg(2)]}
            self.assertEqual(wh.operands(ins), old_operands(ins))
        for fn in (wh.base_op, wh._op_class):
            info = fn.cache_info()
            self.assertLessEqual(info.currsize, info.maxsize)
            self.assertLessEqual(info.currsize, 4096)


if __name__ == "__main__":
    unittest.main()

"""Parity and invariant tests for the dataflow fast paths in tools/image_bvh_census.py.

    python -I tests/test_image_bvh_census_fast.py -v

Checks _insn_meta, vgpr_access, sgpr_step, SiteState.write, site_transfer, and analyse against a verbatim
copy of the previous implementation across multi-site BVH traversal loops, narrowed EXEC branches, SDWA
high-half products, VOP3b carry outputs, and inner back-edges. Standard library only; no game data or GPU.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import image_bvh_census as C  # noqa: E402
import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402

ENC = " ; encoding: [0x00,0x00,0x00,0x00]"


def _listing(*lines):
    body, labels, pc = [], {}, 0
    for ln in lines:
        if ln.endswith(":"):
            labels[ln[:-1]] = pc
        else:
            body.append((pc, ln))
            pc += 4
    out = []
    for at, ln in body:
        if "@" in ln:
            op, target = ln.split("@")
            ln = f"{op}{(labels[target] - at - 4) // 4}"
        out.append("\t" + ln + ENC)
    return f"; cs_probe  code 0x0..0x{pc:x} (header), gfx1013, 0 invalid words\n" + "\n".join(out) + "\n"


def _load_insns(*lines):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cs_probe.s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(_listing(*lines))
        _, insns = C.load_program(path)
    return insns


def _ref_vgpr_access(ins):
    op, a, mods = ins["op"], ins["args"], ins.get("mods", [])
    if not a:
        return [], [], False
    bop = rdna2spirv.base_op(op)
    no_dst = op.startswith(C.NO_VGPR_DST) or op.startswith("v_cmpx")
    if bop.startswith(("global_atomic_", "flat_atomic_", "buffer_atomic_", "image_atomic_")) and "glc" not in mods:
        no_dst = True
    if bop.startswith("ds_") and not (bop.startswith(C.LDS_RETURNING) or "_rtn_" in bop):
        no_dst = True
    if op.startswith("s_"):
        return [], [], False
    sources = [n for n in range(len(a)) if no_dst or n > 0]
    reads = [(n, C.vregs(a[n])) for n in sources if C.vregs(a[n])]
    partial = rdna2spirv.sdwa_reads_dst(op, mods) or bop == "v_writelane_b32" or \
        ("_d16" in bop and bop.startswith(C.MEMORY))
    if (partial or bop in C.ACC_DST) and not no_dst and C.vregs(a[0]):
        reads.append((0, C.vregs(a[0])))
    writes = [] if no_dst else C.vregs(a[0])
    if bop == "v_swap_b32" and len(a) > 1:
        writes = C.vregs(a[0]) + C.vregs(a[1])
    return reads, writes, partial


def _ref_sgpr_step(sg, ins):
    op, a = ins["op"], ins["args"]
    bop = rdna2spirv.base_op(op)
    written = []
    if a and not op.startswith(C.SALU_NO_DST):
        written += C.sregs(a[0])
    if len(a) > 1 and ("_co_" in bop or bop.startswith("v_div_scale")):
        written += C.sregs(a[1])
    value = {}
    if bop in ("s_mov_b32", "s_movk_i32", "s_brev_b32") and len(a) == 2 and len(C.sregs(a[0])) == 1:
        v = C.imm32(a[1])
        if v is not None:
            if bop == "s_movk_i32":
                v = (v & 0xFFFF) - (0x10000 if v & 0x8000 else 0) & 0xFFFFFFFF
            elif bop == "s_brev_b32":
                v = int(f"{v:032b}"[::-1], 2)
            value[a[0]["r"]] = v
        elif bop == "s_mov_b32" and len(C.sregs(a[1])) == 1 and a[1]["r"] in sg and C.plain(a[1]):
            value[a[0]["r"]] = sg[a[1]["r"]]
    elif bop == "s_mov_b64" and len(a) == 2 and len(C.sregs(a[0])) == 2:
        r = a[0]["r"]
        if a[1]["k"] == "imm" and -16 <= a[1]["v"] <= 64:
            v = a[1]["v"] & 0xFFFFFFFFFFFFFFFF
            value = {r: v & 0xFFFFFFFF, r + 1: v >> 32}
        elif len(C.sregs(a[1])) == 2:
            value = {r + i: sg[a[1]["r"] + i] for i in (0, 1) if a[1]["r"] + i in sg}
    for r in written:
        sg.pop(r, None)
    sg.update(value)


def _ref_write(st, r, new, partial=False):
    node = st.exec
    old = st.vg.get(r, C.OTHER_ENTRIES)
    if node == C.ROOT and not partial:
        out = frozenset(new)
    else:
        mine = frozenset((lab, guard | {("in", node)}) for lab, guard in new)
        rest = old if partial else frozenset((lab, guard | {("out", node)}) for lab, guard in old)
        out = C.entries_union(frozenset(e for e in mine if C.live(e[1])), frozenset(e for e in rest if C.live(e[1])))
    if out == C.OTHER_ENTRIES:
        st.vg.pop(r, None)
    else:
        st.vg[r] = out


def _ref_site_transfer(call_pc, events, product_reads):
    def transfer(st, ins):
        a = ins["args"]
        bop = rdna2spirv.base_op(ins["op"])
        pc = ins["pc"]
        reads, writes, partial = _ref_vgpr_access(ins)
        copy = C.is_copy(ins)
        ev, pr = [], []
        for n, regs in reads if not copy else ():
            for r in regs:
                s = st.read(r)
                if C.H in s:
                    pr.append((C.product_reader(ins, n), "all" if s == {C.H} else "some"))
                words = sorted(x for x in s if x[0] == "w")
                if not words:
                    continue
                cat, reach = C.classify(st, ins, n, words)
                if len(words) != len(s):
                    reach = "some"
                ev += [(w[1], cat, reach, bop) for w in words]
        events[pc] = ev
        product_reads[pc] = pr
        if pc == call_pc:
            new = {r: {("w", k)} for k, r in enumerate(writes[:4])}
        elif copy:
            new = {r: st.read(a[1]["r"]) for r in writes[:1]}
            if bop == "v_swap_b32":
                new[a[1]["r"]] = st.read(a[0]["r"])
        elif bop == "v_rcp_f32" and writes and len(a) > 1 and len(C.vregs(a[1])) == 1:
            new = {writes[0]: C.derived(st.read(a[1]["r"]), C.R1)}
        elif writes and any(C.high_half_operand(ins, n) for n in (1, 2)):
            n = 1 if C.high_half_operand(ins, 1) else 2
            new = {writes[0]: C.derived(st.read(a[n]["r"]), C.H)}
        else:
            new = {r: {C.OTHER} for r in writes}
        for r, labs in new.items():
            _ref_write(st, r, [(lab, frozenset()) for lab in labs], partial)
        C.exec_step(st, ins)
    return transfer


def _ref_analyse(insns):
    blocks, succ = C.blocks_of(insns)
    where = {i["pc"]: (b, n) for b, blk in enumerate(blocks) for n, i in enumerate(blk)}
    calls = [i for i in insns if rdna2spirv.base_op(i["op"]) in C.FORMS]
    found_desc = {}

    def desc_transfer(sg, ins):
        if ins["pc"] in {c["pc"] for c in calls}:
            a = ins["args"]
            r = a[2]["r"] if len(a) > 2 and a[2]["k"] == "sgpr" else None
            old = found_desc.get(ins["pc"])
            now = [sg.get(r + i) if r is not None else None for i in range(4)]
            found_desc[ins["pc"]] = now if old is None else [x if x == y else None for x, y in zip(old, now)]
        _ref_sgpr_step(sg, ins)

    C.flow(blocks, succ, (0, 0), desc_transfer, C.join_sgprs, {})
    sites = []
    for c in calls:
        events, product_reads = {}, {}
        at_cut = C.flow(blocks, succ, where[c["pc"]], _ref_site_transfer(c["pc"], events, product_reads), C.join_site,
                        C.SiteState(), copy=C.SiteState.copy, cut=c["pc"], on_back=C.strip, idle=C.SiteState.idle)
        sites.append({"form": rdna2spirv.base_op(c["op"]), "pc": c["pc"], "reached": c["pc"] in found_desc,
                      "loop_carried": any(lab[0] == "w" for st in at_cut for s in st.vg.values() for lab, _ in s),
                      "desc": found_desc.get(c["pc"], [None] * 4),
                      "reads": [(pc,) + e + (any(c["pc"] < x["pc"] < pc for x in calls),)
                                for pc in sorted(events) for e in events[pc]],
                      "product_reads": [(pc,) + e for pc in sorted(product_reads) for e in product_reads[pc]]})
    return sites


class ImageBvhCensusFastTests(unittest.TestCase):
    def test_vgpr_access_and_insn_meta_parity(self):
        samples = [
            "s_mov_b32 s8, s2",
            "s_and_saveexec_b32 s40, vcc_lo",
            "s_andn2_wrexec_b32 s42, s41",
            "s_mov_b32 exec_lo, s40",
            "v_add_f32_e32 v10, v4, v5",
            "v_rcp_f32_e32 v20, v5",
            "v_mov_b32_e32 v12, v7",
            "v_swap_b32 v4, v5",
            "v_fmac_f32_e32 v9, v4, v5",
            "v_writelane_b32 v6, s2, 0",
            "v_cmpx_lt_u32_e32 2, v31",
            "v_cmp_gt_i32_e32 vcc_lo, 0, v24",
            "v_add_co_u32_e64 v1, s[12:13], v2, v3",
            "v_mul_i32_i24_sdwa v24, sext(v5), v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD",
            "v_mov_b32_sdwa v5, v6 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD",
            "global_atomic_add v[0:1], v2, off",
            "global_atomic_add v3, v[0:1], v2, off glc",
            "ds_write_b32 v0, v1",
            "ds_read_b32 v2, v0",
            "ds_add_rtn_u32 v3, v0, v1",
            "buffer_load_short_d16_hi v4, v0, s[0:3], 0 idxen",
            "image_bvh_intersect_ray v[4:7], v[0:10], s[8:11]",
            "image_bvh64_intersect_ray v[4:7], [v0, v1, v2, v3, v8, v9, v10, v11, v12, v13, v14, v15], s[12:15]",
        ]
        for idx, text in enumerate(samples):
            with self.subTest(idx=idx, text=text):
                ins = parse_line("\t" + text + ENC)
                ins["pc"] = idx * 4
                got_acc = C.vgpr_access(ins)
                want_acc = _ref_vgpr_access(ins)
                self.assertEqual(got_acc, want_acc)
                meta = C._insn_meta(ins)
                self.assertEqual(meta[0], ins["pc"])
                self.assertEqual(meta[1], rdna2spirv.base_op(ins["op"]))
                self.assertEqual(meta[2:5], got_acc)
                self.assertEqual(meta[5], C.is_copy(ins))

    def test_sgpr_step_non_sgpr_short_circuit_and_parity(self):
        seq = [
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s8, s2",
            "s_brev_b32 s9, 0x1fe",
            "s_movk_i32 s10, -2",
            "s_mov_b64 s[12:13], -1",
            "s_mov_b64 s[14:15], s[8:9]",
            "v_add_f32_e32 v4, v5, v6",
            "v_rcp_f32_e32 v20, v5",
            "v_add_co_u32_e64 v1, s8, v2, v3",
            "v_readfirstlane_b32 s9, v4",
            "s_load_dwordx2 s[12:13], s[0:1], 0x0",
        ]
        sg_got, sg_want = {}, {}
        for idx, text in enumerate(seq):
            with self.subTest(idx=idx, text=text):
                ins = parse_line("\t" + text + ENC)
                ins["pc"] = idx * 4
                if text.startswith("v_") and "s8" not in text and "s9" not in text:
                    orig_base_op = rdna2spirv.base_op
                    try:
                        def _forbid_base_op(_op):
                            raise AssertionError("sgpr_step must short-circuit pure-VGPR instructions before base_op")

                        rdna2spirv.base_op = _forbid_base_op
                        C.sgpr_step(sg_got, ins)
                    finally:
                        rdna2spirv.base_op = orig_base_op
                else:
                    C.sgpr_step(sg_got, ins)
                _ref_sgpr_step(sg_want, ins)
                self.assertEqual(sg_got, sg_want)
                self.assertEqual(C._insn_meta(ins)[0], ins["pc"])

    def test_sitestate_write_skips_untracked_other_under_narrowed_exec(self):
        node = ("n", 8, C.ROOT)
        with self.subTest(case="untracked_other_under_narrowed_exec"):
            st = C.SiteState(ex=node)
            st.write(32, [(C.OTHER, frozenset())], partial=False)
            st.write(33, [(C.OTHER, frozenset())], partial=True)
            self.assertEqual(st.vg, {})
            self.assertTrue(st.idle())

        with self.subTest(case="tracked_word_overwritten_in_same_narrowed_exec_pops_to_other"):
            st = C.SiteState(ex=node)
            st.write(5, [(("w", 1), frozenset())], partial=False)
            self.assertIn(5, st.vg)
            st.write(5, [(C.OTHER, frozenset())], partial=False)
            self.assertNotIn(5, st.vg)
            self.assertEqual(st.read(5), {C.OTHER})

    def test_analyse_parity_and_precomputed_meta_across_bvh_programs(self):
        desc = [
            "s_mov_b32 s8, s2",
            "s_brev_b32 s9, 0x1fe",
            "s_movk_i32 s10, -2",
            "s_mov_b32 s11, 0x52000123",
        ]
        call32 = "image_bvh_intersect_ray v[4:7], v[0:10], s[8:11]"
        triangle = [
            "v_rcp_f32_e32 v20, v5",
            "v_mul_f32_e32 v21, v4, v20",
            "v_mul_legacy_f32_e32 v22, v20, v6",
            "v_mov_b32_e32 v12, v7",
            "v_mul_f32_e32 v23, v12, v20",
            "v_mul_i32_i24_sdwa v24, sext(v5), v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD",
            "v_cmp_gt_i32_e32 vcc_lo, 0, v24",
        ]
        programs = [
            ("tri_and_alu_under_narrowed_exec", [
                *desc,
                "v_cmp_ne_u32_e32 vcc_lo, 0, v31",
                "s_and_saveexec_b32 s40, vcc_lo",
                call32,
                *triangle,
                "v_add_f32_e32 v32, v30, v31",
                "v_mul_f32_e32 v33, v32, v30",
                "s_andn2_b32 exec_lo, s40, exec_lo",
                "v_rcp_f32_e32 v25, v5",
                "s_mov_b32 exec_lo, s40",
                "v_add_f32_e32 v26, v5, v8",
                "s_endpgm",
            ]),
            ("multi_site_loop_and_swap", [
                *desc,
                "L_top:",
                call32,
                "v_swap_b32 v4, v5",
                "v_rcp_f32_e32 v20, v4",
                "v_mul_f32_e32 v21, v5, v20",
                "s_cbranch_scc1 @L_next",
                "v_mov_b32_e32 v6, 0",
                "L_next:",
                "v_lshlrev_b32_e32 v10, 3, v6",
                "image_bvh64_intersect_ray v[4:7], v[0:11], s[8:11]",
                "v_add_f32_e32 v9, v7, v8",
                "v_mul_f32_e32 v10, v9, v4",
                "v_add_f32_e32 v11, v10, v5",
                "s_cbranch_scc0 @L_top",
                "s_endpgm",
            ]),
            ("inner_loop_and_wrexec", [
                call32,
                "s_mov_b32 s41, exec_lo",
                "v_cmpx_lt_u32_e32 2, v31",
                "v_mov_b32_e32 v6, 0",
                "s_andn2_wrexec_b32 s42, s41",
                "v_cmp_eq_u32_e32 vcc_lo, v32, v6",
                "s_mov_b32 exec_lo, s41",
                "s_mov_b32 s50, exec_lo",
                "L_inner:",
                "v_readfirstlane_b32 s4, v9",
                "v_cmpx_eq_u32_e32 s4, v9",
                "v_rcp_f32_e32 v20, v5",
                "v_mov_b32_e32 v5, 0",
                "s_xor_b32 s50, s50, exec_lo",
                "s_mov_b32 exec_lo, s50",
                "s_cbranch_execnz @L_inner",
                "s_endpgm",
            ]),
        ]
        for name, lines in programs:
            with self.subTest(name=name):
                insns = _load_insns(*lines)
                want = _ref_analyse(insns)
                calls = [0]
                orig_vgpr_access = C.vgpr_access
                try:
                    def _counted_vgpr_access(ins):
                        calls[0] += 1
                        return orig_vgpr_access(ins)

                    C.vgpr_access = _counted_vgpr_access
                    got = C.analyse(insns)
                finally:
                    C.vgpr_access = orig_vgpr_access
                self.assertEqual(got, want)
                self.assertEqual(calls[0], len(insns))


if __name__ == "__main__":
    unittest.main()

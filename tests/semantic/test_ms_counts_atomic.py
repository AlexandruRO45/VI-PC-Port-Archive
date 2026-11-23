"""A mesh shader's vertex and primitive counts reach the Workgroup variable %ms_counts through an atomic store.

`s_sendmsg(MSG_GS_ALLOC_REQ)` is a scalar instruction, so every invocation of a wave that reaches it runs the
translated store. A plain OpStore there is a data race: many invocations store to one Workgroup word with no barrier
between them, whatever the values are (SHADER-766; GPU-assisted validation reports it on every mesh shader run as a
compute stage). The store is an OpAtomicStore, and the one load of the counts comes after the epilogue's barrier.

    python -m unittest discover -s tests/semantic -p "test_ms_counts_atomic.py"

Standard library only; the programs are hand-written.
"""
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import test_w64x_clear_single_writer as clears  # noqa: E402  (builds a mesh container; has a small mesh program)

rounds = clears.rounds
rdna2spirv = rounds.rdna2spirv

# The send as the corpus has it (the opcode shape only): one thread is picked, the waves without it branch past the
# send, and the wave with it builds M0 from scalars. Every invocation of that wave runs the send.
GUARDED_SEND = [
    "v_mbcnt_lo_u32_b32 v0, -1, 0",
    "v_mbcnt_hi_u32_b32 v0, -1, v0",
    "s_mov_b64 s[4:5], exec",
    "s_mov_b32 s8, 3",
    "s_mov_b32 s9, 1",
    "v_cmpx_eq_u32_e32 0, v0",
    "s_cbranch_execz skip",
    "s_lshl_b32 s9, s9, 12",
    "s_or_b32 s8, s8, s9",
    "s_mov_b32 m0, s8",
    "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)",
    "skip:",
    "s_mov_b64 exec, s[4:5]",
    "v_cvt_f32_u32_e32 v1, v0",
    "v_mov_b32_e32 v6, 0",
    "v_mov_b32_e32 v7, 1.0",
    "v_cmpx_gt_u32_e32 3, v0",
    "exp pos0 v1, v1, v6, v7 done",
    "exp param0 v1, v1, v6, v7",
    "v_cmpx_eq_u32_e32 0, v0",
    "v_mov_b32_e32 v9, 0x200400",
    "exp prim v9, off, off, off done",
    "s_endpgm",
]
# Two sends on two paths: the rule is per store, not "the one store of the module".
TWO_SENDS = [
    "v_mbcnt_lo_u32_b32 v0, -1, 0",
    "v_mbcnt_hi_u32_b32 v0, -1, v0",
    "v_readfirstlane_b32 s11, v0",
    "s_cmp_eq_u32 s11, 0",
    "s_cbranch_scc0 other",
    "s_mov_b32 m0, 0x1003",
    "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)",
    "s_branch join",
    "other:",
    "s_mov_b32 m0, 0x1003",
    "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)",
    "join:",
    "v_cvt_f32_u32_e32 v1, v0",
    "v_mov_b32_e32 v6, 0",
    "v_mov_b32_e32 v7, 1.0",
    "v_cmpx_gt_u32_e32 3, v0",
    "exp pos0 v1, v1, v6, v7 done",
    "exp param0 v1, v1, v6, v7",
    "v_cmpx_eq_u32_e32 0, v0",
    "v_mov_b32_e32 v9, 0x200400",
    "exp prim v9, off, off, off done",
    "s_endpgm",
]
PROGRAMS = {"plain": (clears.MESH_PROGRAM, 1), "guarded": (GUARDED_SEND, 1), "two sends": (TWO_SENDS, 2)}
MODES = [(emulate, as_compute) for emulate in (False, True) for as_compute in (False, True)]


def translate(program, threads, emulate, as_compute):
    """The SPIR-V text of a hand-written wave64 mesh program: natively or with wave64 emulation, as a mesh stage or
    as a compute stage (--mesh-compute)."""
    name = "ms_counts_%d" % threads
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n; mattr: +wavefrontsize64,-wavefrontsize32\n")
            for text, size in rounds._resolve_labels(program):
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(clears.mesh_container(threads))
        saved = rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE
        rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE = emulate, as_compute
        try:
            spv = rdna2spirv.translate(src)
            gen = rdna2spirv.translate.last_gen
        finally:
            rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE = saved
    assert gen.stage == "ms", gen.stage
    return spv


def uses(spv):
    """Every line of the module that names %ms_counts, in order, with its line number."""
    return [(i, ln.strip()) for i, ln in enumerate(spv.splitlines()) if re.search(r"%ms_counts\b", ln)]


class CountsAreStoredAtomically(unittest.TestCase):
    def check(self, spv, sends):
        lines = [ln.strip() for ln in spv.splitlines()]
        found = uses(spv)
        stores = [(i, ln) for i, ln in found if ln.startswith("OpAtomicStore %ms_counts ")]
        loads = [(i, ln) for i, ln in found if re.match(r"%\w+ = OpLoad %uint %ms_counts$", ln)]
        declared = [ln for _, ln in found if ln.startswith(("OpEntryPoint ", "%ms_counts = OpVariable ", "OpName "))]
        self.assertFalse([ln for _, ln in found if ln.startswith("OpStore ")], "a plain store to %ms_counts")
        self.assertEqual(len(stores), sends)
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(found), len(stores) + len(loads) + len(declared), [ln for _, ln in found])
        self.assertIn("%ms_counts = OpVariable %_ptr_WG_uint_ms Workgroup", declared)
        consts = dict(re.match(r"(%\w+) = OpConstant %uint (\d+)$", ln).groups()
                      for ln in lines if re.match(r"%\w+ = OpConstant %uint \d+$", ln))
        for _, ln in stores:
            _, _, scope, semantics, value = ln.split()
            self.assertEqual((consts[scope], consts[semantics]), ("2", "0"), ln)       # Workgroup scope, relaxed
            self.assertIn(f"{value} = OpLoad %uint %m0", lines, "the stored value is M0")
        # The load is the epilogue's: the barrier before it comes after every store, so no store is unordered with it.
        barriers = [i for i, ln in enumerate(lines) if ln.startswith("OpControlBarrier ") and i < loads[0][0]]
        self.assertTrue(barriers, "no barrier before the load of the counts")
        self.assertLess(max(i for i, _ in stores), barriers[-1])
        self.assertNotIn("OpAtomicStore", "".join(lines[barriers[-1]:loads[0][0]]))

    def test_every_send_is_one_atomic_store_before_the_barrier(self):
        for label, (program, sends) in PROGRAMS.items():
            for threads in (128, 40):
                for emulate, as_compute in MODES:
                    with self.subTest(program=label, threads=threads, emulate=emulate, as_compute=as_compute):
                        self.check(translate(program, threads, emulate, as_compute), sends)

    def test_the_stage_is_what_the_mode_says(self):
        for emulate, as_compute in MODES:
            with self.subTest(emulate=emulate, as_compute=as_compute):
                spv = translate(clears.MESH_PROGRAM, 128, emulate, as_compute)
                self.assertEqual("OpEntryPoint GLCompute" in spv, as_compute)
                self.assertEqual("OpEntryPoint MeshEXT" in spv, not as_compute)
                self.assertEqual("%x_buf = OpVariable" in spv, emulate)

    def test_the_check_reports_a_plain_store(self):
        spv = translate(clears.MESH_PROGRAM, 128, True, True)
        plain = re.sub(r"OpAtomicStore %ms_counts %\w+ %\w+ (%\w+)", r"OpStore %ms_counts \1", spv)
        self.assertNotEqual(plain, spv)
        with self.assertRaises(AssertionError):
            self.check(plain, 1)


if __name__ == "__main__":
    unittest.main()

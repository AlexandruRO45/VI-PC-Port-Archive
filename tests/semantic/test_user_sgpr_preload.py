"""Regression suite for live-in user-SGPR push-constant preloading (TEST-592).

translate() preloads only user SGPRs (s0..s31 excluding entry-written system SGPRs) that may be
read before being definitely written on every path from entry, falling back to all non-system
s0..s31 when s_movrels_* is present, while direct Gen() callers preload all 32. Standard library
only; spirv-val checks run when SPIRV-Tools are on PATH (required under LEONIDA_REQUIRE_SPIRV_TOOLS=1).
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from spirv_wave_interp import run_wave  # noqa: E402

MS_W64X_BALLOT = os.path.join(support.ROOT, "runtime", "test", "shaders", "ms_w64x_ballot.asm")


def _dxbc(chunks):
    hdr = 32 + 4 * len(chunks)
    offsets, body = [], bytearray()
    for tag, payload in chunks:
        offsets.append(hdr + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    out = bytearray(b"DXBC") + bytes(16) + struct.pack("<III", 1, hdr + len(body), len(chunks))
    for off in offsets:
        out += struct.pack("<I", off)
    return bytes(out) + bytes(body)


def _mesh_psv0(threads, max_vertices, max_primitives):
    r = bytearray(48)
    struct.pack_into("<HH", r, 12, max_vertices, max_primitives)
    r[24] = 13
    r[27] = 2
    struct.pack_into("<III", r, 36, threads, 1, 1)
    return struct.pack("<I", len(r)) + bytes(r)


def _make_fssm_cs(user_sgprs=4, tgid_mask=1, tid_cnt=0, threads=(64, 1, 1), wave32=True):
    buf = bytearray(0x160)
    buf[0xD4] = 0x20 if wave32 else 0x00
    rsrc2 = ((user_sgprs & 31) << 1) | ((tgid_mask & 7) << 7) | ((tid_cnt & 3) << 11)
    struct.pack_into("<IIII", buf, 0x138, rsrc2, threads[0], threads[1], threads[2])
    return bytes(buf)


def _encode_listing(name, body_lines):
    """Build a .s listing with valid relative branch encodings and ; mattr lines preserved."""
    lbl_idx = {}
    ins_count = 0
    for ln in body_lines:
        if not ln.startswith(";") and not ln.endswith(":"):
            ins_count += 1
        elif ln.endswith(":"):
            lbl_idx[ln[:-1]] = ins_count
    out = [f"; {name}  synthetic user-SGPR preload test"]
    idx = 0
    for ln in body_lines:
        if ln.endswith(":"):
            continue
        if ln.startswith(";"):
            out.append(ln)
            continue
        enc = 0
        if ln.startswith(("s_cbranch_", "s_branch ")):
            op, target = ln.split()
            rel = (lbl_idx[target] - idx - 1) & 0xFFFF
            enc = 0xBF880000 | rel
            signed = rel if rel < 0x8000 else rel - 0x10000
            ln = f"{op} {signed}"
        b = enc.to_bytes(4, "little")
        enc_s = ",".join(f"0x{x:02x}" for x in b)
        out.append(f"\t{ln:<44} ; encoding: [{enc_s}]")
        idx += 1
    return ("\n".join(out) + "\n").encode("utf-8")


def _validate_spirv(spvasm):
    sa = shutil.which("spirv-as")
    sv = shutil.which("spirv-val")
    if not (sa and sv):
        if os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS"):
            raise AssertionError("spirv-as / spirv-val required but not found on PATH")
        return
    with tempfile.TemporaryDirectory() as td:
        asm_p = os.path.join(td, "mod.spvasm")
        spv_p = os.path.join(td, "mod.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        subprocess.run([sa, "--target-env", "vulkan1.2", asm_p, "-o", spv_p], check=True)
        subprocess.run([sv, "--target-env", "vulkan1.2", spv_p], check=True)


def _preloaded_slots(spvasm):
    return tuple(int(m.group(1)) for m in re.finditer(r"^\s*%ud(\d+) = OpLoad %uint %ud\1p$", spvasm, re.M))


class UserSgprPreloadTests(unittest.TestCase):
    def test_emulated_wave64_mesh_no_live_in_user_sgprs(self):
        """ms_w64x_ballot has 0 live-in user SGPRs: no dead %ud* preloads across rounds while keeping ABI."""
        # Preserve "; mattr:" directive and instruction lines cleanly.
        with open(MS_W64X_BALLOT, encoding="utf-8") as f:
            raw_lines = []
            for ln in f:
                s = ln.strip()
                if s.startswith("; mattr:"):
                    raw_lines.append(s)
                else:
                    code = ln.split(";")[0].strip()
                    if code:
                        raw_lines.append(code)
        old_emu = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            spvasm = support.translate_lines(
                "ms_w64x_ballot_preload",
                raw_lines,
                dxbc=_dxbc([(b"PSV0", _mesh_psv0(128, 64, 64))]),
            )
            gen = rdna2spirv.translate.last_gen
            meta = rdna2spirv.shader_meta(gen, spvasm)
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emu

        _validate_spirv(spvasm)
        self.assertEqual(gen.control_flow, "emulated wave64 rounds")
        self.assertEqual(gen.user_sgpr_preload, ())
        self.assertEqual(_preloaded_slots(spvasm), ())
        self.assertIn("%UserData = OpTypeStruct %arr_ud", spvasm)
        self.assertEqual(meta["push_constant_bytes"], 144)

    def test_live_in_vs_scratch_user_sgprs(self):
        """Only s0 and s10 (read before write) preload; scratch s11 (written before read) does not."""
        lines = [
            "s_mov_b32 s11, 0x12345678",
            "s_add_u32 s12, s0, s10",
            "s_add_u32 s12, s12, s11",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s10",
            "v_mov_b32 v2, s11",
            "v_mov_b32 v3, s12",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_live_in_sgpr", lines)
        gen = rdna2spirv.translate.last_gen
        _validate_spirv(spvasm)
        self.assertEqual(gen.user_sgpr_preload, (0, 10))
        self.assertEqual(_preloaded_slots(spvasm), (0, 10))
        self.assertNotIn("%ud11 ", spvasm)

        pc = [0] * 32
        pc[0] = 0x11223344
        pc[10] = 0x55667788
        pc[11] = 0x99AABBCC
        got = support.mrt0_words(spvasm, push_constants=pc)
        self.assertEqual(got, [0x11223344, 0x55667788, 0x12345678, (0x11223344 + 0x55667788 + 0x12345678) & 0xFFFFFFFF])

    def test_conditional_single_arm_write_stays_live_in(self):
        """If only one branch arm writes s10 and merge reads s10, s10 remains preloaded."""
        body = [
            "s_cmp_eq_u32 s0, 0",
            "s_cbranch_scc1 BB_merge",
            "s_mov_b32 s10, 0x77777777",
            "BB_merge:",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, s0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_listing("ps_one_arm_s10", _encode_listing("ps_one_arm_s10", body))
        gen = rdna2spirv.translate.last_gen
        _validate_spirv(spvasm)
        self.assertIn(10, gen.user_sgpr_preload)
        self.assertIn(0, gen.user_sgpr_preload)

        pc_skip = [0] * 32
        pc_skip[0] = 0
        pc_skip[10] = 0xCAFE1234
        res_skip = run_wave(spvasm, wave=32, push_constants=pc_skip)
        self.assertEqual(tuple(res_skip["outputs"][0]["%out_mrt0"]), (0xCAFE1234, 0, 0, 1))

        pc_take = [0] * 32
        pc_take[0] = 1
        pc_take[10] = 0xCAFE1234
        res_take = run_wave(spvasm, wave=32, push_constants=pc_take)
        self.assertEqual(tuple(res_take["outputs"][0]["%out_mrt0"]), (0x77777777, 1, 0, 1))

    def test_conditional_both_arms_write_eliminates_preload(self):
        """If every path writes s10 before merge reads it, s10 is NOT preloaded."""
        body = [
            "s_cmp_eq_u32 s0, 0",
            "s_cbranch_scc1 BB_zero",
            "s_mov_b32 s10, 0x11111111",
            "s_branch BB_merge",
            "BB_zero:",
            "s_mov_b32 s10, 0x22222222",
            "BB_merge:",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, s0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_listing("ps_both_arms_s10", _encode_listing("ps_both_arms_s10", body))
        gen = rdna2spirv.translate.last_gen
        _validate_spirv(spvasm)
        self.assertEqual(gen.user_sgpr_preload, (0,))
        self.assertEqual(_preloaded_slots(spvasm), (0,))

        pc0 = [0] * 32
        pc0[10] = 0xDEAD0000
        self.assertEqual(tuple(run_wave(spvasm, wave=32, push_constants=pc0)["outputs"][0]["%out_mrt0"]), (0x22222222, 0, 0, 1))
        pc1 = [1] + [0] * 31
        pc1[10] = 0xDEAD0000
        self.assertEqual(tuple(run_wave(spvasm, wave=32, push_constants=pc1)["outputs"][0]["%out_mrt0"]), (0x11111111, 1, 0, 1))

    def test_rmw_ops_keep_preload(self):
        """RMW SGPR ops (s_addk_i32, s_cmov_b32, s_bitset1_b32) on s10 before a full write stay preloaded."""
        cases = (
            ("s_addk_i32", ["s_addk_i32 s10, 5"], lambda x, s0: (x + 5) & 0xFFFFFFFF),
            ("s_cmov_b32", ["s_cmp_eq_u32 s0, 1", "s_cmov_b32 s10, 0x99"], lambda x, s0: 0x99 if s0 == 1 else x),
            ("s_bitset1_b32", ["s_bitset1_b32 s10, 3"], lambda x, s0: x | (1 << 3)),
        )
        for label, extra, expected_fn in cases:
            with self.subTest(op=label):
                lines = extra + [
                    "v_mov_b32 v0, s10",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 1",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spvasm = support.translate_lines(f"ps_rmw_{label}", lines)
                gen = rdna2spirv.translate.last_gen
                _validate_spirv(spvasm)
                self.assertIn(10, gen.user_sgpr_preload)
                for s0 in (0, 1):
                    pc = [0] * 32
                    pc[0] = s0
                    pc[10] = 0x100
                    got = support.mrt0_words(spvasm, push_constants=pc)
                    self.assertEqual(got[0], expected_fn(0x100, s0))

    def test_v_cmpx_e64_does_not_mark_sgpr_written(self):
        """v_cmpx_*_e64 s10, v0, v1 updates EXEC only (rdna2spirv ignores a[0]), so s10 stays preloaded."""
        lines = [
            "v_mov_b32 v0, 7",
            "v_mov_b32 v1, 7",
            "v_cmpx_eq_u32_e64 s10, v0, v1",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_cmpx_s10", lines)
        gen = rdna2spirv.translate.last_gen
        _validate_spirv(spvasm)
        self.assertIn(10, gen.user_sgpr_preload)
        pc = [0] * 32
        pc[10] = 0xABCDEF01
        self.assertEqual(support.mrt0_words(spvasm, push_constants=pc)[0], 0xABCDEF01)

    def test_s_movrels_forces_full_non_system_preload(self):
        """Dynamic SGPR indexing (s_movrels_b32) preloads every non-system SGPR 0..31."""
        lines = [
            "s_mov_b32 m0, 10",
            "s_movrels_b32 s0, s0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_movrels", lines)
        gen = rdna2spirv.translate.last_gen
        _validate_spirv(spvasm)
        self.assertEqual(gen.user_sgpr_preload, tuple(range(32)))
        self.assertEqual(_preloaded_slots(spvasm), tuple(range(32)))
        pc = [0] * 32
        pc[10] = 0xDEADBEEF
        self.assertEqual(support.mrt0_words(spvasm, push_constants=pc)[0], 0xDEADBEEF)

    def test_compute_reads_past_rsrc2_user_and_skips_tgid(self):
        """Compute shader with user=4, tgid.x=s4: s8 still preloads, while system s4 is skipped."""
        vsharp_s = [0, 0, 0x1000, 0x31016FAC]
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_and_saveexec_b32 s5, vcc",
            "v_mov_b32 v1, s8",
            "v_mov_b32 v2, s4",
            "v_mov_b32 v3, 0",
            "v_mov_b32 v4, 4",
            "buffer_store_dword v1, v3, s[0:3], 0 offen",
            "buffer_store_dword v2, v4, s[0:3], 0 offen",
            "s_endpgm",
        ]
        fssm = _make_fssm_cs(user_sgprs=4, tgid_mask=1, tid_cnt=0, threads=(32, 1, 1), wave32=True)
        spvasm = support.translate_lines("cs_past_user", lines, fssm=fssm)
        gen = rdna2spirv.translate.last_gen
        _validate_spirv(spvasm)
        self.assertIn(8, gen.user_sgpr_preload)
        self.assertNotIn(4, gen.user_sgpr_preload)
        pc = vsharp_s + [0x11111111, 0, 0, 0, 0xCAFEBABE] + [0] * 23
        mem = si.run_workgroup(
            spvasm, 32, 32,
            inputs_for=lambda i: {"%wg_id": (7, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=pc, memory={},
        )
        self.assertEqual(mem[0], 0xCAFEBABE)
        self.assertEqual(mem[4], 7)

    def test_direct_gen_preloads_all_32(self):
        """Direct Gen() + assemble() without translate() defaults to preloading all 32 slots."""
        g = rdna2spirv.Gen("ps")
        spvasm = rdna2spirv.assemble(g)
        _validate_spirv(spvasm)
        self.assertIsNone(g.user_sgpr_preload)
        self.assertEqual(_preloaded_slots(spvasm), tuple(range(32)))


if __name__ == "__main__":
    unittest.main()

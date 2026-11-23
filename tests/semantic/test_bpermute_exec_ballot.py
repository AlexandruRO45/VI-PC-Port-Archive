"""Regression tests for SHADER-801: ds_lane_read ballots %active instead of pre-selecting data."""
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from compute_fixture import compute_header  # noqa: E402

VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]
HEAP_FILL = 0xEEEEEEEE
EXEC_LO = 0xFD25B8DB
EXEC_HI = 0x00000000
INIT_VDST = 0x11223344


class Heap(dict):
    def get(self, key, default=None):
        return super().get(key, HEAP_FILL)


def _header(threads, w64=True):
    asm_lines = [f"; group: {threads},1,1", "; rsrc2: 0x88"]
    if w64:
        asm_lines.append("; mattr: +wavefrontsize64,-wavefrontsize32")
    with tempfile.NamedTemporaryFile("w", suffix=".asm", delete=False, encoding="utf-8") as f:
        f.write("\n".join(asm_lines) + "\ns_endpgm\n")
        path = f.name
    try:
        return bytes(compute_header(path))
    finally:
        os.unlink(path)


def _run_shader(spv, threads, sg_size):
    mem = si.run_workgroup(
        spv, threads, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=VSHARP + [0] * 28,
        memory=Heap(),
    )
    return [mem.get(4 * k, HEAP_FILL) for k in range(threads)]


class PartialExecBallotLowering(unittest.TestCase):
    def _build_bpermute_program(self, w64=True):
        lines = [
            "s_mov_b64 exec, -1" if w64 else "s_mov_b32 exec_lo, -1",
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
        ]
        if w64:
            lines.append("v_mbcnt_hi_u32_b32 v0, -1, v0")
        lines += [
            f"v_mov_b32_e32 v22, {INIT_VDST:#x}",
            "v_add_nc_u32_e32 v1, 0x100, v0",
            f"s_mov_b32 exec_lo, {EXEC_LO:#x}",
        ]
        if w64:
            lines.append(f"s_mov_b32 exec_hi, {EXEC_HI:#x}")
        lines += [
            "v_mul_u32_u24_e32 v2, 27, v0",
            "v_add_nc_u32_e32 v2, 35, v2",
            "v_lshlrev_b32_e32 v2, 2, v2",
            "ds_bpermute_b32 v22, v2, v1",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b64 exec, -1" if w64 else "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v22, v3, s[0:3], 0 offen",
            "s_endpgm",
        ]
        return lines

    def test_wave64_and_wave32_ballot_active_directly(self):
        for w64, threads in ((True, 64), (False, 32)):
            with self.subTest(wave64=w64):
                lines = self._build_bpermute_program(w64=w64)
                spv = support.translate_lines(
                    f"cs_shader801_w{'64' if w64 else '32'}",
                    lines,
                    fssm=_header(threads, w64=w64),
                )
                self.assertNotIn("OpGroupNonUniformBallot %v4uint_bal %c3 %true", spv)
                m = re.search(
                    r"(%t\d+) = OpLoad %bool %active\n\s*"
                    r"(%t\d+) = OpGroupNonUniformBallot %v4uint_bal %c3 \1\n\s*"
                    r"(%t\d+) = OpGroupNonUniformBallotBitExtract %bool %c3 \2 (%t\d+)\n\s*"
                    r"(%t\d+) = OpSelect %uint \3 \4 %t\d+\n\s*"
                    r"%t\d+ = OpGroupNonUniformShuffle %uint %c3 (?P<dat>%t\d+) \5\n",
                    spv,
                )
                self.assertIsNotNone(m, "expected ds_lane_read to ballot %active and shuffle data directly")

                got = _run_shader(spv, threads, threads)
                for t in range(threads):
                    active = (EXEC_LO >> t) & 1 if t < 32 else (EXEC_HI >> (t - 32)) & 1
                    if active:
                        src = (27 * t + 35) % 32
                        src_active = (EXEC_LO >> src) & 1
                        want = (0x100 + src) if src_active else 0
                    else:
                        want = INIT_VDST
                    self.assertEqual(got[t], want, f"lane {t} mismatch")

    def test_sabotage_ballot_true_leaks_inactive_source_register(self):
        # Negative control: if OpGroupNonUniformBallot uses %true instead of %active (without pre-masking data),
        # lane 3 reads lane 20 (inactive in EXEC), leaking lane 20's v1 value (0x114) instead of 0.
        lines = self._build_bpermute_program(w64=True)
        spv = support.translate_lines("cs_shader801_neg", lines, fssm=_header(64, w64=True))
        got_ok = _run_shader(spv, 64, 64)
        self.assertEqual(got_ok[3], 0)

        bad, replaced = re.subn(
            r"OpGroupNonUniformBallot %v4uint_bal %c3 %t\d+",
            "OpGroupNonUniformBallot %v4uint_bal %c3 %true",
            spv,
        )
        self.assertEqual(replaced, 1)
        got_bad = _run_shader(bad, 64, 64)
        self.assertEqual(got_bad[3], 0x100 + 20)


if __name__ == "__main__":
    unittest.main()

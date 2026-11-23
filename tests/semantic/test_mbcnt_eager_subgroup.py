"""v_mbcnt_lo_u32_b32 / v_mbcnt_hi_u32_b32 mask reductions run in every lane under SPIRV-Cross (#6214, SHADER-529).

When v_mbcnt_lo_u32_b32 or v_mbcnt_hi_u32_b32 reads a lane mask (SGPR, SGPR pair, VCC or EXEC) while EXEC has been
narrowed, Gen.wave_mask reconstructs the 32-bit mask word with a single-use OpGroupNonUniformBitwiseOr reduction that
feeds straight into Gen.dst's `OpSelect %uint %active %r %old`. Without Gen.eager, SPIRV-Cross (MoltenVK 1.4.2) inlines
`simd_or(...)` inside `active ? (...) : old`, so lanes currently off in EXEC skip the reduction and drop their set mask
bits. Storing the selected mask word through Gen.eager makes it a statement that every lane runs before dst's select.

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.
"""
import os
import struct
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from test_dpp_eager_subgroup import conditional_subgroup_ops  # noqa: E402

rdna2spirv = support.rdna2spirv


def _fssm(threads: int) -> bytes:
    buf = bytearray(0x148)
    struct.pack_into("<4I", buf, 0x138, 0x8088, threads, 1, 1)
    return bytes(buf)


FSSM32 = _fssm(32)
FSSM64 = _fssm(64)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

W32_EXPECTED = [1000, 109, 112, 1008, 208, 213, 1002, 320]
W64_EXPECTED = [1005, 120, 132, 1035, 200, 212, 327, 411, 512, 611]


def _asm_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith(";")]


def _run_cs(spv: str, threads: int, count: int) -> list[int | None]:
    memory = si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})
    return [memory.get(4 * k) for k in range(count)]


class MbcntSubgroupOpsRunInEveryLane(unittest.TestCase):
    def test_cs_w32_mbcnt_exec_eager_and_values(self):
        spv = support.translate_fixture("cs_w32_mbcnt_exec", fssm=FSSM32)
        self.assertFalse(rdna2spirv.translate.last_gen.wave64)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        self.assertEqual(conditional_subgroup_ops(spv), [])
        self.assertEqual(_run_cs(spv, 32, len(W32_EXPECTED)), W32_EXPECTED)

    def test_cs_w64_mbcnt_exec_eager_and_values(self):
        lines = _asm_lines(Path(HERE) / "fixtures" / "mbcnt_exec_wave64.asm")
        spv = support.translate_lines("cs_w64_mbcnt_exec", lines, fssm=FSSM64)
        self.assertTrue(rdna2spirv.translate.last_gen.wave64)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        self.assertEqual(conditional_subgroup_ops(spv), [])
        self.assertEqual(_run_cs(spv, 64, len(W64_EXPECTED)), W64_EXPECTED)

    def test_cs_wavemask_has_no_conditional_subgroup_ops(self):
        spv = support.translate_fixture("cs_wavemask", fssm=FSSM64)
        self.assertEqual(conditional_subgroup_ops(spv), [])

    def test_vs_and_ps_wave32_and_wave64_have_no_conditional_subgroup_ops(self):
        vs_w32 = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_cmp_gt_u32_e64 s4, 8, v0",
            "v_cmpx_le_u32_e32 4, v0",
            "v_mbcnt_lo_u32_b32 v2, s4, 10",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32 v5, 0",
            "exp pos0 v5, v5, v5, v5 done",
            "exp param0 v2, v5, v5, v5",
            "s_endpgm",
        ]
        ps_w32 = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_cmp_gt_u32_e64 s4, 8, v0",
            "v_cmpx_le_u32_e32 4, v0",
            "v_mbcnt_lo_u32_b32 v2, s4, 10",
            "s_mov_b32 exec_lo, -1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        ps_w64 = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mbcnt_hi_u32_b32 v0, -1, v0",
            "v_cmp_gt_u32_e64 s[4:5], 40, v0",
            "v_cmpx_le_u32_e32 8, v0",
            "v_mbcnt_lo_u32_b32 v2, s[4:5], 10",
            "v_mbcnt_hi_u32_b32 v3, s[4:5], 20",
            "s_mov_b64 exec, -1",
            "exp mrt0 v2, v3, v2, v3 done vm",
            "s_endpgm",
        ]
        for name, lines in (("vs_mbcnt_w32", vs_w32), ("ps_mbcnt_w32", ps_w32), ("ps_mbcnt_w64", ps_w64)):
            with self.subTest(shader=name):
                spv = support.translate_lines(name, lines)
                self.assertEqual(conditional_subgroup_ops(spv), [])

    def test_plain_data_and_wave64_emulate_skip_eager_spill(self):
        # Plain immediate / data v_mbcnt uses no subgroup op and allocates no %eager variable.
        plain = support.translate_lines(
            "cs_mbcnt_imm",
            ["v_mbcnt_lo_u32_b32 v0, -1, 0", "v_mbcnt_lo_u32_b32 v1, 0x55555555, 7", "s_endpgm"],
            fssm=FSSM32,
        )
        self.assertNotIn("%eager0", plain)

        # --wave64-emulate exchanges masks through Workgroup memory rounds rather than subgroup ops.
        lines = _asm_lines(Path(HERE) / "fixtures" / "mbcnt_exec_wave64.asm")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            emulated = support.translate_lines("cs_w64_mbcnt_exec", lines, fssm=FSSM64)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        self.assertTrue(rdna2spirv.translate.last_gen.wave64)
        self.assertNotIn("%eager0", emulated)


if __name__ == "__main__":
    unittest.main()

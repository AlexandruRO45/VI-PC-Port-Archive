"""Semantic interpreter checks for v_add/sub/subrev_co_ci_u32 under _sdwa and _dpp.

Standard library only; no GPU, no LLVM, no game files.

Covers:
  - _sdwa source selects (src0_sel / src1_sel), sext(src0) / sext(src1), dst_sel / dst_unused,
    unsigned clamp with real carry/borrow preserved in VCC, destination aliasing, wave32 and wave64
  - _dpp DPP16 (quad_perm, row_shr, row_mask, bank_mask, bound_ctrl 0/1, fi 0/1) and DPP8 across
    all three ops (including subrev shuffling src0, not src1), plus null carry-out
  - carry-out scalar mask write under dpp_guard (disabled row/bank or bound_ctrl:0 invalid source
    writes 0 to VCC, never leaking pre-existing carry-in) and downstream VCC consumers
  - wave32, wave64 native, wave64 --wave64-emulate structure, and CPU verification of the GPU
    fixture runtime/test/shaders/cs_co_ci_sdwa_dpp.asm
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

rdna2spirv = support.rdna2spirv

M32 = 0xFFFFFFFF
VSHARP = [0, 0, M32, 0x31016FAC]
ENC4 = " ; encoding: [0x00,0x00,0x00,0x00]"


def _fssm(wave_size: int) -> bytes:
    """Compute header: 4 user SGPRs (output V#), TGID_X in s4, wave_size x 1 x 1 workgroup."""
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, 0x8088, wave_size, 1, 1)
    return bytes(blob)


def _translate_cs(name: str, body_lines: list[str], wave_size: int = 32, *,
                  emulate: bool = False, with_enc: bool = False) -> str:
    suf = ENC4 if with_enc else ""
    lines = [ln + suf if suf and ln.strip() and not ln.strip().startswith(";") else ln for ln in body_lines]
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        return support.translate_lines(name, lines, fssm=_fssm(wave_size))
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu


def _run_cs(name: str, body_lines: list[str], wave_size: int = 32, *, with_enc: bool = False) -> dict[int, int]:
    spv = _translate_cs(name, body_lines, wave_size=wave_size, with_enc=with_enc)
    return si.run_workgroup(
        spv,
        wave_size,
        wave_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


def _seed_vcc_mask32(mask: int) -> list[str]:
    """Set each lane L's bit of vcc_lo to (mask >> L) & 1 via v_cmp on v0 = lane."""
    return [
        "v_lshlrev_b32 v29, v0, 1",
        f"v_and_b32 v29, {mask:#x}, v29",
        "v_cmp_ne_u32_e32 vcc_lo, 0, v29",
    ]


GPU_EXPECTED = [
    0x0FFF1234,
    0xFFFE8000,
    0x00002000,
    0x00007777,
    0x0001000A,
    0x0001FFFA,
    0x00000006,
    0x0000ABCD,
]


class GpuFixtureTests(unittest.TestCase):
    def test_cs_co_ci_sdwa_dpp_matches_on_cpu(self):
        spv = support.translate_fixture("cs_co_ci_sdwa_dpp", fssm=_fssm(32))
        mem = si.run_workgroup(
            spv,
            32,
            32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28,
            memory={},
        )
        self.assertEqual([mem.get(4 * k) for k in range(8)], GPU_EXPECTED)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_co_ci_sdwa_dpp"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], GPU_EXPECTED)


class CoCiSdwaSemanticsTests(unittest.TestCase):
    def _run_ps(self, body_lines: list[str]) -> list[int]:
        lines = list(body_lines) + ["exp mrt0 v1, v2, v3, v4 done vm", "s_endpgm"]
        spv = support.translate_lines("ps_co_ci_sdwa", lines)
        return support.mrt0_words(spv)

    def test_sdwa_source_selects_sext_and_carry_chain(self):
        got = self._run_ps([
            "v_mov_b32 v1, 0xCAFE1234",
            "v_mov_b32 v5, 0x00FF0001",    # WORD_0=1, BYTE_2=0xFF -> sext=-1
            "v_mov_b32 v6, 0x00028000",    # WORD_1=2, BYTE_1=0x80 -> sext=0xFFFFFF80
            "s_mov_b32 vcc_lo, -1",
            # v1[31:16] <- WORD_0(v5) + BYTE_1(sext(v6)) + cin(1) = 1 + (-128) + 1 = 0xFFFFFF82 -> low 16 bits 0xFF82, carry=0
            "v_add_co_ci_u32_sdwa v1, vcc_lo, v5, sext(v6), vcc_lo dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:BYTE_1",
            "v_cndmask_b32_e64 v2, 10, 20, vcc_lo",
            # v3 <- sext(BYTE_2(v5)) - WORD_1(v6) - cin(0) = -1 - 2 - 0 = 0xFFFFFFFD, borrow=0
            "v_sub_co_ci_u32_sdwa v3, vcc_lo, sext(v5), v6, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:WORD_1",
            "v_cndmask_b32_e64 v4, 30, 40, vcc_lo",
        ])
        self.assertEqual(got, [0xFF821234, 10, 0xFFFFFFFD, 30])

    def test_sdwa_dst_unused_modes_and_clamp(self):
        # clamp saturates result to 0xFFFFFFFF (add) or 0 (sub/subrev) while still reporting real carry/borrow in vcc_lo,
        # and runs before dst_sel/dst_unused placement.
        for op, a, b, cin, unused, dst_sel, want_v1, want_carry in [
            ("v_add_co_ci_u32_sdwa", 0xFFFFFFFF, 5, -1, "UNUSED_PAD", "BYTE_2", 0x00FF0000, 1),
            ("v_sub_co_ci_u32_sdwa", 0x00000003, 10, 0, "UNUSED_SEXT", "BYTE_1", 0x00000000, 1),
            ("v_subrev_co_ci_u32_sdwa", 0x00000080, 0x0000007F, -1, "UNUSED_PRESERVE", "WORD_1", 0x00001234, 1),
            ("v_subrev_co_ci_u32_sdwa", 0x00000002, 0x00000085, 0, "UNUSED_SEXT", "BYTE_0", 0xFFFFFF83, 0),
        ]:
            with self.subTest(op=op, unused=unused, dst_sel=dst_sel):
                got = self._run_ps([
                    "v_mov_b32 v1, 0xDEAD1234",
                    f"v_mov_b32 v2, 0x{a:08x}",
                    f"v_mov_b32 v3, 0x{b:08x}",
                    f"s_mov_b32 vcc_lo, {cin}",
                    f"{op} v1, vcc_lo, v2, v3, vcc_lo clamp dst_sel:{dst_sel} dst_unused:{unused} src0_sel:DWORD src1_sel:DWORD",
                    "v_cndmask_b32_e64 v2, 0, 1, vcc_lo",
                    "v_mov_b32 v3, 0",
                    "v_mov_b32 v4, 0",
                ])
                self.assertEqual(got[:2], [want_v1, want_carry])

    def test_sdwa_aliasing_and_wave64(self):
        # Wave64 canonical spelling uses vcc (5-op), and dst aliasing src0/src1 preserves pre-op values.
        mem = _run_cs(
            "cs_w64_co_ci_sdwa",
            [
                "v_mov_b32 v1, 0x12340080",
                "v_mov_b32 v2, 0x000500FE",
                "s_mov_b64 vcc, -1",
                # dst == src0: WORD_1(v1)=0x1234 + sext(BYTE_0(v2))=-2 + cin(1) = 0x1233 -> WORD_0 preserving 0x1234
                "v_add_co_ci_u32_sdwa v1, vcc, v1, sext(v2), vcc dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:BYTE_0",
                "v_cndmask_b32_e64 v3, 0, 1, vcc",
                "v_cmpx_eq_u32_e32 0, v0",
                "v_mov_b32 v4, 0",
                "buffer_store_dword v1, v4, s[0:3], 0 offen",
                "v_mov_b32 v4, 4",
                "buffer_store_dword v3, v4, s[0:3], 0 offen",
                "s_endpgm",
            ],
            wave_size=64,
        )
        self.assertEqual([mem[0], mem[4]], [0x12341233, 1])


class CoCiDppSemanticsTests(unittest.TestCase):
    def _run_dpp_case(self, body_lines: list[str], wave_size: int = 32) -> tuple[list[int], list[int]]:
        """Run body_lines and store v10 (result) + v11 (vcc bit) for every lane into memory[0..2*wave_size*4]."""
        lines = list(body_lines) + [
            "s_mov_b32 exec_lo, -1" if wave_size == 32 else "s_mov_b64 exec, -1",
            "v_cndmask_b32_e64 v11, 0, 1, vcc_lo" if wave_size == 32 else "v_cndmask_b32_e64 v11, 0, 1, vcc",
            "v_lshlrev_b32 v12, 2, v0",
            "buffer_store_dword v10, v12, s[0:3], 0 offen",
            f"v_add_nc_u32 v13, {4 * wave_size}, v12",
            "buffer_store_dword v11, v13, s[0:3], 0 offen",
            "s_endpgm",
        ]
        mem = _run_cs(f"cs_co_ci_dpp_{wave_size}", lines, wave_size=wave_size)
        vals = [mem[4 * i] for i in range(wave_size)]
        vcc_bits = [mem[4 * (wave_size + i)] for i in range(wave_size)]
        return vals, vcc_bits

    def test_dpp_row_bank_bound_masks_update_vdst_and_zero_vcc(self):
        # Wave32 and wave64 native:
        # row_mask:0xa enables rows 1 & 3 (lanes 16..31 and 48..63); bank_mask:0x5 enables banks 0 & 2.
        # With row_shr:1 and omitted bound_ctrl (BC=0), row-boundary lanes (0, 16, 32, 48) are also disabled:
        #   - enabled & in-bounds lanes write sum and carry=1;
        #   - disabled lanes preserve old v10=0xCAFE0000|lane AND write 0 to VCC despite vcc starting all-ones.
        for wave_size in (32, 64):
            vcc = "vcc_lo" if wave_size == 32 else "vcc"
            vcc_init = "s_mov_b32 vcc_lo, -1" if wave_size == 32 else "s_mov_b64 vcc, -1"
            with self.subTest(wave_size=wave_size):
                vals, carries = self._run_dpp_case(
                    [
                        "v_add_nc_u32 v1, 10, v0",
                        "v_mov_b32 v2, 0xFFFFFFFF",
                        "v_or_b32 v10, 0xCAFE0000, v0",
                        vcc_init,
                        f"v_add_co_ci_u32_dpp v10, {vcc}, v1, v2, {vcc} row_shr:1 row_mask:0xa bank_mask:0x5",
                    ],
                    wave_size=wave_size,
                )
                for lane in range(wave_size):
                    row_on = (0xA >> (lane >> 4)) & 1
                    bank_on = (0x5 >> ((lane >> 2) & 3)) & 1
                    in_row = (lane & 15) >= 1
                    if row_on and bank_on and in_row:
                        total = 10 + (lane - 1) + 0xFFFFFFFF + 1
                        self.assertEqual(vals[lane], total & M32, f"lane {lane}")
                        self.assertEqual(carries[lane], int(total > M32), f"lane {lane}")
                    else:
                        self.assertEqual(vals[lane], 0xCAFE0000 | lane, f"lane {lane}")
                        self.assertEqual(carries[lane], 0, f"lane {lane}")

    def test_dpp_bound_ctrl_1_sub_and_subrev(self):
        # bound_ctrl:1 substitutes 0 for out-of-row src0; subrev shuffles src0 (right subtrahend: v2 - shuf(v1) - cin).
        for wave_size in (32, 64):
            vcc = "vcc_lo" if wave_size == 32 else "vcc"
            # Even lanes get carry-in = 1, odd lanes get carry-in = 0.
            cin_setup = (["v_and_b32 v29, 1, v0", "v_cmp_eq_u32_e32 vcc_lo, 0, v29"] if wave_size == 32
                         else ["s_mov_b64 vcc, 0x5555555555555555"])
            with self.subTest(wave_size=wave_size):
                vals_sub, borrow_sub = self._run_dpp_case(
                    [
                        "v_add_nc_u32 v1, 1, v0",
                        "v_mov_b32 v2, 10",
                        "v_mov_b32 v10, 0xBAD",
                        *cin_setup,
                        f"v_sub_co_ci_u32_dpp v10, {vcc}, v1, v2, {vcc} row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
                    ],
                    wave_size=wave_size,
                )
                for lane in range(wave_size):
                    cin = 1 if (lane % 2 == 0) else 0
                    src0 = 0 if (lane & 15) == 0 else (1 + lane - 1)
                    diff = src0 - 10 - cin
                    self.assertEqual(vals_sub[lane], diff & M32, f"sub wave{wave_size} lane {lane}")
                    self.assertEqual(borrow_sub[lane], int(diff < 0), f"sub borrow wave{wave_size} lane {lane}")

                # subrev with quad_perm:[3,2,1,0]: lane l reads v1[l^3] as S0, computing v2[l] - v1[l^3] - cin[l].
                vals_rev, borrow_rev = self._run_dpp_case(
                    [
                        "v_add_nc_u32 v1, 1, v0",
                        "v_add_nc_u32 v2, 10, v0",
                        "v_mov_b32 v10, 0xBAD",
                        *cin_setup,
                        f"v_subrev_co_ci_u32_dpp v10, {vcc}, v1, v2, {vcc} quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
                    ],
                    wave_size=wave_size,
                )
                for lane in range(wave_size):
                    cin = 1 if (lane % 2 == 0) else 0
                    diff = (10 + lane) - (1 + (lane ^ 3)) - cin
                    self.assertEqual(vals_rev[lane], diff & M32, f"subrev wave{wave_size} lane {lane}")
                    self.assertEqual(borrow_rev[lane], int(diff < 0), f"subrev borrow wave{wave_size} lane {lane}")

    def test_dpp8_fi_and_partial_exec_with_wave_consumers(self):
        # Partial EXEC: only even lanes are active; odd source lanes have v1[odd]=100+odd.
        # Under dpp8:[1,0,3,2,5,4,7,6] (pair swap), fi:0 reads src0=0 on odd-source; fi:1 reads v1[odd].
        # Verify s_bcnt1_i32_b32 and s_cbranch_vccnz when ALL active lanes compute 0xFFFFFFFF+src0+1, vs no-carry
        # when DPP bank_mask:0x0 disables every lane's write (VCC must be 0 so s_cbranch_vccz IS taken!).
        for fi, want_src0 in ((0, lambda l: 0), (1, lambda l: 100 + (l ^ 1))):
            with self.subTest(fi=fi):
                lines = [
                    "v_add_nc_u32 v1, 100, v0",
                    "v_mov_b32 v2, 0xFFFFFFFF",
                    "v_or_b32 v10, 0xBEEF0000, v0",
                    "s_mov_b32 vcc_lo, -1",
                    "s_mov_b32 exec_lo, 0x55555555",
                    f"v_add_co_ci_u32_dpp v10, vcc_lo, v1, v2, vcc_lo dpp8:[1,0,3,2,5,4,7,6] fi:{fi}",
                    "s_bcnt1_i32_b32 s9, vcc_lo",
                    "s_mov_b32 exec_lo, -1",
                    "v_lshlrev_b32 v12, 2, v0",
                    "buffer_store_dword v10, v12, s[0:3], 0 offen",
                    "v_mov_b32 v14, s9",
                    "v_cmpx_eq_u32_e32 0, v0",
                    "v_mov_b32 v15, 128",
                    "buffer_store_dword v14, v15, s[0:3], 0 offen",
                    "s_endpgm",
                ]
                mem = _run_cs(f"cs_co_ci_dpp8_fi{fi}", lines, wave_size=32)
                for lane in range(32):
                    if lane % 2 == 0:
                        total = want_src0(lane) + 0xFFFFFFFF + 1
                        self.assertEqual(mem[4 * lane], total & M32)
                    else:
                        self.assertEqual(mem[4 * lane], 0xBEEF0000 | lane)
                self.assertEqual(mem[128], 16)

        # s_cbranch_vccz must be taken when row_mask:0x0 disables all DPP writes despite carry-in=-1 and 0xFFFFFFFF+1.
        branch_lines = [
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_mov_b32 v10, 0x7777",
            "s_mov_b32 vcc_lo, -1",
            "s_mov_b32 s14, 0",
            "v_add_co_ci_u32_dpp v10, vcc_lo, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0xf",
            "s_cbranch_vccz 1",
            "s_mov_b32 s14, 1",
            "v_mov_b32 v8, s14",
            "v_lshlrev_b32 v9, 2, v0",
            "buffer_store_dword v8, v9, s[0:3], 0 offen",
            "s_endpgm",
        ]
        bmem = _run_cs("cs_co_ci_dpp_vccz", branch_lines, wave_size=32, with_enc=True)
        for lane in range(32):
            self.assertEqual(bmem[4 * lane], 0, f"lane {lane}: s_cbranch_vccz not taken")

    def test_dpp_null_carry_dst_keeps_vcc_and_respects_guard(self):
        # Null carry destination leaves VCC untouched while respecting dpp_guard on vdst.
        lines = _seed_vcc_mask32(0x12345678) + [
            "v_add_nc_u32 v1, 10, v0",
            "v_mov_b32 v2, 5",
            "v_or_b32 v10, 0xABCD0000, v0",
            "v_add_co_ci_u32_dpp v10, null, v1, v2, vcc_lo row_shr:1 row_mask:0xf bank_mask:0x1",
        ]
        vals, vcc_bits = self._run_dpp_case(lines, wave_size=32)
        for lane in range(32):
            expected_vcc = (0x12345678 >> lane) & 1
            self.assertEqual(vcc_bits[lane], expected_vcc)
            enabled = (lane & 15) >= 1 and ((0x1 >> ((lane >> 2) & 3)) & 1)
            if enabled:
                self.assertEqual(vals[lane], 10 + (lane - 1) + 5 + expected_vcc)
            else:
                self.assertEqual(vals[lane], 0xABCD0000 | lane)


class CoCiDppWave64EmulateTests(unittest.TestCase):
    """Verify --wave64-emulate translates 5-op _co_ci_u32_{sdwa,dpp} and routes DPP + guard into VCC."""

    def test_emulate_co_ci_dpp_and_sdwa(self):
        for name, body in [
            ("dpp", [
                "s_mov_b64 exec, -1",
                "v_add_nc_u32 v1, 10, v0",
                "v_mov_b32 v2, 0xFFFFFFFF",
                "s_mov_b64 vcc, -1",
                "v_add_co_ci_u32_dpp v10, vcc, v1, v2, vcc row_shr:1 row_mask:0xa bank_mask:0x5",
                "s_bcnt1_i32_b64 s14, vcc",
                "v_mov_b32 v8, s14",
                "v_lshlrev_b32 v9, 2, v0",
                "buffer_store_dword v8, v9, s[0:3], 0 offen",
                "s_endpgm",
            ]),
            ("sdwa", [
                "s_mov_b64 exec, -1",
                "v_mov_b32 v1, 0x10002",
                "v_mov_b32 v2, 0xFE00",
                "s_mov_b64 vcc, -1",
                "v_add_co_ci_u32_sdwa v10, vcc, v1, sext(v2), vcc dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:BYTE_1",
                "v_lshlrev_b32 v9, 2, v0",
                "buffer_store_dword v10, v9, s[0:3], 0 offen",
                "s_endpgm",
            ]),
        ]:
            with self.subTest(name=name):
                spv = _translate_cs(f"cs_emu_co_ci_{name}", body, wave_size=64, emulate=True)
                self.assertIn("OpStore %vcc", spv)
                if name == "dpp":
                    self.assertIn("OpStore %x_it", spv)
                    self.assertIn("OpControlBarrier", spv)
                else:
                    self.assertIn("OpBitFieldSExtract", spv)
                    self.assertIn("OpBitFieldUExtract", spv)
                    self.assertIn("OpBitFieldInsert", spv)


if __name__ == "__main__":
    unittest.main()

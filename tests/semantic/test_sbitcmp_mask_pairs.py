"""Semantic regression tests for s_bitcmp0/1_b64 (and b32) on mixed and non-canonical lane-mask pairs.

RDNA2 ISA (SOPC):
  - s_bitcmp1_b64 S0, S1: SCC = ((S0[63:0] >> (S1 & 63)) & 1) != 0
  - s_bitcmp0_b64 S0, S1: SCC = ((S0[63:0] >> (S1 & 63)) & 1) == 0
  - s_bitcmp1_b32 / s_bitcmp0_b32: same on S0[31:0] with shift S1 & 31.

Covers:
  - wave64 mixed mask/data SGPR pairs: (MASK_LO, data), (data, MASK_HI), (MASK_HI, data), (data, MASK_LO)
  - wave64 non-canonical mask SGPR pairs: (MASK_HI, 0), (0, MASK_LO), (MASK_HI, MASK_LO), (MASK_LO, MASK_LO),
    and partial-wave (40-thread) (MASK_LO, -1) testing unlaunched-lane bits
  - wave32 b64 SGPR pairs: (mask, data), (mask, -1), (mask, mask), (0, mask), plus wave64 plain-data vcc and null sources
  - lane-mask bit-index operand a[1] in s_bitcmp0/1_b32 and s_bitcmp0/1_b64
  - preservation of the single-wave_mask fast path for canonical wave64 (MASK_LO, MASK_HI) pairs and zero subgroup
    ops for plain-data pairs
  - --wave64-emulate barrier detection in tools/wave64_emulate_opt.py, including a mixed-pair s_bitcmp1_b64
    inside a s_cbranch_execz region.

Standard library only; CPU-only, no GPU, no game files.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402
import wave64_emulate_opt as wo  # noqa: E402

VSHARP = [0, 0, 0x1000, 0x31016FAC]


def make_fssm(wave64: bool, threads: int | None = None) -> bytes:
    n = threads if threads is not None else (64 if wave64 else 32)
    h = bytearray(0x148)
    h[0:4] = b"FSSM"
    struct.pack_into("<I", h, 0xD4, 0 if wave64 else 0x20)
    struct.pack_into("<4I", h, 0x138, 0x8088, n, 1, 1)
    return bytes(h)


def run_bitcmp64(
    wave64: bool,
    setup_lines: list[str],
    bit_idx: int | str,
    *,
    src: str = "s[10:11]",
    threads: int | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    n = threads if threads is not None else (64 if wave64 else 32)
    lines = [
        "v_mbcnt_lo_u32_b32 v1, -1, 0",
        *(["v_mbcnt_hi_u32_b32 v1, -1, v1"] if wave64 else []),
        *setup_lines,
        f"s_bitcmp1_b64 {src}, {bit_idx}",
        "s_cselect_b32 s20, 1, 0",
        f"s_bitcmp0_b64 {src}, {bit_idx}",
        "s_cselect_b32 s21, 1, 0",
        "v_mov_b32_e32 v20, s20",
        "v_mov_b32_e32 v21, s21",
        "v_mul_u32_u24_e32 v22, 8, v1",
        "buffer_store_dword v20, v22, s[0:3], 0 offen",
        "v_add_nc_u32_e32 v23, 4, v22",
        "buffer_store_dword v21, v23, s[0:3], 0 offen",
        "s_endpgm",
    ]
    spv = support.translate_lines("cs_sbitcmp_pair", lines, fssm=make_fssm(wave64, n))
    self_magic = spirv_asm.assemble(spv, target_env="vulkan1.2")[:4]
    assert self_magic == b"\x03\x02\x23\x07", f"bad SPIR-V magic: {self_magic!r}"
    mem = si.run_workgroup(
        spv,
        n,
        64 if wave64 else 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )
    return spv, [(mem[8 * lane], mem[8 * lane + 4]) for lane in range(n)]


# Wave64 reference mask: lanes 10..39
#   s14 (MASK_LO) = bits 10..31 set (0xFFFFFC00)
#   s15 (MASK_HI) = bits  0..7  set (0x000000FF)
W64_SETUP = [
    "v_cmp_le_u32_e64 s[14:15], 10, v1",
    "v_cmp_gt_u32_e64 s[18:19], 40, v1",
    "s_and_b64 s[14:15], s[14:15], s[18:19]",
]
W64_LO_BITS = sum(1 << l for l in range(10, 32))
W64_HI_BITS = sum(1 << (l - 32) for l in range(32, 40))

# Wave32 reference masks:
#   s10 = lanes 10..19 (0x000FFC00)
#   s14 = lanes  0..4  (0x0000001F)
W32_SETUP = [
    "v_cmp_le_u32_e64 s10, 10, v1",
    "v_cmp_gt_u32_e64 s12, 20, v1",
    "s_and_b32 s10, s10, s12",
    "v_cmp_gt_u32_e64 s14, 5, v1",
]
W32_S10_BITS = sum(1 << l for l in range(10, 20))
W32_S14_BITS = sum(1 << l for l in range(0, 5))


def expected_pair(lo: int, hi: int, bit_idx: int) -> tuple[int, int]:
    val64 = (lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32)
    b = (val64 >> (bit_idx & 63)) & 1
    return (b, 1 - b)


class BitcmpMaskPairTests(unittest.TestCase):
    def assertAllLanesMatch(self, got: list[tuple[int, int]], want: tuple[int, int], msg: str = "") -> None:
        bad = [(lane, g) for lane, g in enumerate(got) if g != want]
        self.assertFalse(bad, f"{msg}: {len(bad)}/{len(got)} lanes wrong, first={bad[:3]}, want={want}")

    def test_wave64_mixed_mask_and_data_pairs(self):
        cases = [
            ("lo_data", W64_SETUP + ["s_mov_b32 s10, s14", "s_mov_b32 s11, 5"], W64_LO_BITS, 5, (0, 10, 31, 32, 33, 34)),
            ("data_hi", W64_SETUP + ["s_mov_b32 s10, 5", "s_mov_b32 s11, s15"], 5, W64_HI_BITS, (0, 1, 2, 32, 39, 40)),
            ("hi_data", W64_SETUP + ["s_mov_b32 s10, s15", "s_mov_b32 s11, 5"], W64_HI_BITS, 5, (0, 7, 8, 32, 33, 34)),
            ("data_lo", W64_SETUP + ["s_mov_b32 s10, 5", "s_mov_b32 s11, s14"], 5, W64_LO_BITS, (0, 1, 2, 32, 42, 63)),
        ]
        for label, setup, lo_bits, hi_bits, bits in cases:
            for bit_idx in bits:
                with self.subTest(case=label, bit=bit_idx):
                    _, got = run_bitcmp64(True, setup, bit_idx)
                    self.assertAllLanesMatch(got, expected_pair(lo_bits, hi_bits, bit_idx), f"{label} bit={bit_idx}")

    def test_wave64_noncanonical_mask_pairs(self):
        cases = [
            ("hi_in_lo_with_0", W64_SETUP + ["s_mov_b32 s10, s15", "s_mov_b32 s11, 0"], W64_HI_BITS, 0, (0, 7, 8, 10, 32)),
            ("lo_in_hi_with_0", W64_SETUP + ["s_mov_b32 s10, 0", "s_mov_b32 s11, s14"], 0, W64_LO_BITS, (0, 10, 32, 42, 63)),
            ("swapped_hi_lo", W64_SETUP + ["s_mov_b32 s10, s15", "s_mov_b32 s11, s14"], W64_HI_BITS, W64_LO_BITS, (0, 8, 32, 42, 63)),
            ("dup_lo_lo", W64_SETUP + ["s_mov_b32 s10, s14", "s_mov_b32 s11, s14"], W64_LO_BITS, W64_LO_BITS, (0, 10, 32, 42, 63)),
        ]
        for label, setup, lo_bits, hi_bits, bits in cases:
            for bit_idx in bits:
                with self.subTest(case=label, bit=bit_idx):
                    _, got = run_bitcmp64(True, setup, bit_idx)
                    self.assertAllLanesMatch(got, expected_pair(lo_bits, hi_bits, bit_idx), f"{label} bit={bit_idx}")

    def test_wave64_partial_wave_mask_and_all_ones(self):
        # 40 launched threads: lanes 40..63 are unlaunched, yet s11 = -1 (EITHER) must keep all upper bits = 1.
        setup = W64_SETUP + ["s_mov_b32 s10, s14", "s_mov_b32 s11, -1"]
        for bit_idx in (0, 10, 32, 50, 63):
            with self.subTest(bit=bit_idx):
                _, got = run_bitcmp64(True, setup, bit_idx, threads=40)
                self.assertAllLanesMatch(got, expected_pair(W64_LO_BITS, 0xFFFFFFFF, bit_idx), f"partial bit={bit_idx}")

    def test_wave32_b64_mask_pairs(self):
        cases = [
            ("mask_data", W32_SETUP + ["s_mov_b32 s11, 5"], "s[10:11]", W32_S10_BITS, 5, (0, 10, 32, 33, 34)),
            ("mask_minus1", W32_SETUP + ["s_mov_b32 s11, -1"], "s[10:11]", W32_S10_BITS, 0xFFFFFFFF, (0, 10, 35, 63)),
            ("two_masks", W32_SETUP + ["s_mov_b32 s11, s14"], "s[10:11]", W32_S10_BITS, W32_S14_BITS, (10, 32, 36, 37)),
            ("zero_mask", W32_SETUP + ["s_mov_b32 s10, 0", "s_mov_b32 s11, s14"], "s[10:11]", 0, W32_S14_BITS, (0, 32, 33, 37)),
        ]
        for label, setup, src, lo_bits, hi_bits, bits in cases:
            for bit_idx in bits:
                with self.subTest(case=label, bit=bit_idx):
                    _, got = run_bitcmp64(False, setup, bit_idx, src=src)
                    self.assertAllLanesMatch(got, expected_pair(lo_bits, hi_bits, bit_idx), f"{label} bit={bit_idx}")

    def test_plain_data_vcc_and_null_sources(self):
        # Wave64 plain-data vcc (written with scalar constants, not a lane mask): vcc_hi bit 0 (bit 32) = 1.
        setup_vcc = ["s_mov_b32 vcc_lo, 4", "s_mov_b32 vcc_hi, 5"]
        for bit_idx in (0, 2, 32, 33, 34):
            with self.subTest(src="vcc", bit=bit_idx):
                _, got = run_bitcmp64(True, setup_vcc, bit_idx, src="vcc")
                self.assertAllLanesMatch(got, expected_pair(4, 5, bit_idx), f"plain vcc bit={bit_idx}")

        # null supplies 0 for all 64 bits.
        for bit_idx in (0, 32, 63):
            with self.subTest(src="null", bit=bit_idx):
                _, got = run_bitcmp64(True, [], bit_idx, src="null")
                self.assertAllLanesMatch(got, (0, 1), f"null bit={bit_idx}")

    def test_lane_mask_bit_index_operand(self):
        # Wave32: s10 = mask(lane == 2) => 0x00000004 (bit index 4). Bit 4 of 0x10 is 1 on all 32 lanes.
        lines32 = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_cmp_eq_u32_e64 s10, 2, v1",
            "s_mov_b32 s11, 16",
            "s_bitcmp1_b32 s11, s10",
            "s_cselect_b32 s20, 1, 0",
            "s_bitcmp0_b32 s11, s10",
            "s_cselect_b32 s21, 1, 0",
            "v_mov_b32_e32 v20, s20",
            "v_mov_b32_e32 v21, s21",
            "v_mul_u32_u24_e32 v22, 8, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v23, 4, v22",
            "buffer_store_dword v21, v23, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv32 = support.translate_lines("cs_bitcmp32_idx", lines32, fssm=make_fssm(False))
        self.assertEqual(spirv_asm.assemble(spv32, target_env="vulkan1.2")[:4], b"\x03\x02\x23\x07")
        mem32 = si.run_workgroup(
            spv32, 32, 32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28, memory={},
        )
        got32 = [(mem32[8 * l], mem32[8 * l + 4]) for l in range(32)]
        self.assertAllLanesMatch(got32, (1, 0), "w32 mask bit index")

        # Wave64: s[14:15] = mask(lane == 34) => s15 (MASK_HI) = 0x00000004 (bit index 4).
        # s[10:11] = (0x10, 0): bit 4 is 1 on all 64 lanes.
        setup64 = [
            "v_cmp_eq_u32_e64 s[14:15], 34, v1",
            "s_mov_b32 s10, 16",
            "s_mov_b32 s11, 0",
        ]
        _, got64 = run_bitcmp64(True, setup64, "s15")
        self.assertAllLanesMatch(got64, (1, 0), "w64 MASK_HI bit index")

    def test_canonical_pair_keeps_single_ballot_and_data_pair_stays_local(self):
        # Canonical wave64 (MASK_LO, MASK_HI): 2 instructions (bitcmp1 + bitcmp0), 1 wave_mask per op (2 BitwiseOr each).
        spv_canon, got_canon = run_bitcmp64(True, W64_SETUP, 35, src="s[14:15]")
        self.assertAllLanesMatch(got_canon, expected_pair(W64_LO_BITS, W64_HI_BITS, 35))
        self.assertEqual(spv_canon.count("OpGroupNonUniformBitwiseOr"), 4)

        # Plain-data pair: zero subgroup operations.
        spv_data, got_data = run_bitcmp64(True, ["s_mov_b32 s10, 5", "s_mov_b32 s11, 9"], 35)
        self.assertAllLanesMatch(got_data, expected_pair(5, 9, 35))
        self.assertEqual(spv_data.count("OpGroupNonUniform"), 0)


def _bitcmp_ins(op: str, src: dict, idx: dict | None = None) -> dict:
    return {"op": op, "args": [src, idx or {"k": "imm", "v": 10}], "mods": set()}


class BitcmpWaveEmulateOptTests(unittest.TestCase):
    PAIR = {"k": "sgpr", "r": 10, "n": 2}

    def test_mixed_pair_is_marked_cross_lane(self):
        for op in ("s_bitcmp0_b64", "s_bitcmp1_b64"):
            for lm in ({10: "lo"}, {11: "hi"}):
                with self.subTest(op=op, lm=lm):
                    ins = _bitcmp_ins(op, self.PAIR)
                    self.assertTrue(wo._insn_may_barrier(ins, {id(ins): lm}, set()))

    def test_data_pair_and_unrelated_b32_stay_lane_local(self):
        for op in ("s_bitcmp0_b64", "s_bitcmp1_b64"):
            ins = _bitcmp_ins(op, self.PAIR)
            self.assertFalse(wo._insn_may_barrier(ins, {id(ins): {}}, set()), op)
        for op in ("s_bitcmp0_b32", "s_bitcmp1_b32"):
            ins = _bitcmp_ins(op, {"k": "sgpr", "r": 10, "n": 1})
            self.assertFalse(wo._insn_may_barrier(ins, {id(ins): {11: "hi"}}, set()), op)

    def test_emulated_translation_with_mixed_pair_in_execz_branch(self):
        # Mixed-pair s_bitcmp1_b64 inside a divergent s_cbranch_execz block must not be folded as sync-free
        # (doing so used to raise AssertionError: x_split inside a folded region under --wave64-emulate).
        insns = [
            ("v_mbcnt_lo_u32_b32 v1, -1, 0", 4),
            ("v_mbcnt_hi_u32_b32 v1, -1, v1", 4),
            ("v_cmp_lt_u32_e64 s[10:11], v1, 20", 8),
            ("s_mov_b32 s11, 5", 4),
            ("v_cmp_lt_u32_e64 vcc, v1, 32", 8),
            ("s_and_saveexec_b64 s[12:13], vcc", 4),
            ("s_cbranch_execz 3", 4),
            ("s_bitcmp1_b64 s[10:11], 10", 4),
            ("s_cselect_b32 s20, 1, 0", 4),
            ("v_mov_b32_e32 v20, s20", 4),
            ("s_or_b64 exec, exec, s[12:13]", 4),
            ("v_mul_u32_u24_e32 v22, 4, v1", 4),
            ("buffer_store_dword v20, v22, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        lines = [
            f"{txt:52s} ; encoding: [{','.join(['0x00'] * nb)}]"
            for txt, nb in insns
        ]
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            spv = support.translate_lines("cs_w64x_bitcmp_mixed", lines, fssm=make_fssm(True))
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        self.assertGreater(spv.count("OpAtomicOr"), 0)
        self.assertNotIn("OpGroupNonUniform", spv)
        self.assertEqual(spirv_asm.assemble(spv, target_env="vulkan1.2")[:4], b"\x03\x02\x23\x07")


if __name__ == "__main__":
    unittest.main()

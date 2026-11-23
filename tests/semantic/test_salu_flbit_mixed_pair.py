"""64-bit leading-bit scans must reconstruct a mask word beside scalar data.

The expected values follow the SOP1 contract in docs/design.md: unsigned scans
count leading zeros, signed scans complement negative inputs first, zero returns
0xffffffff, and neither instruction writes SCC. All inputs are hand-written.
"""
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(HERE, "..", "..", "tools"))

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402
import test_salu64_mixed_mask_pair as mixed  # noqa: E402
from test_wave64_emulate_opt import _translate_encoded, run_emulated_workgroup  # noqa: E402

OPS = ("s_flbit_i32_b64", "s_flbit_i32_i64")


def leading_bits(value, signed):
    if signed and value & (1 << 63):
        value ^= (1 << 64) - 1
    return 64 - value.bit_length() if value else 0xFFFFFFFF


def program(op, word, mul, add, mask, data, scc=1, dest="s20", runtime_data=False):
    scalar = "s11" if word == "lo" else "s10"
    return [
        "v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1",
        f"v_mad_u32_u24 v3, v1, {mul}, {add}", f"v_and_b32_e32 v3, {mask}, v3",
        "v_cmp_eq_u32_e64 s[10:11], 0, v3",
        (f"s_buffer_load_dword {scalar}, s[0:3], 1024" if runtime_data else f"s_mov_b32 {scalar}, {data:#x}"),
        f"s_cmp_eq_u32 {scalar}, {data if scc else data ^ 1:#x}",
        f"{op} {dest}, s[10:11]", "s_cselect_b32 s24, 1, 0",
        "v_mul_u32_u24_e32 v21, 8, v1", f"v_mov_b32_e32 v20, {dest}",
        "buffer_store_dword v20, v21, s[0:3], 0 offen",
        "v_mov_b32_e32 v20, s24", "buffer_store_dword v20, v21, s[0:3], 0 offen offset:4",
        "s_endpgm",
    ]


def execute(lines, data=None, wave64=True):
    lanes = 64 if wave64 else 32
    header = bytearray(mixed.fssm(lanes))
    struct.pack_into("<I", header, 0xD4, 0 if wave64 else 0x20)
    spv = support.translate_lines("cs_flbit_mixed", lines, fssm=bytes(header))
    memory = si.run_workgroup(spv, lanes, lanes,
                             inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                             push_constants=mixed.VSHARP + [0] * 28,
                             memory={} if data is None else {1024: data})
    return {addr: value for addr, value in memory.items() if addr < 1024}


class MixedFlbitTests(unittest.TestCase):
    def test_reported_fixture(self):
        lines = (Path(HERE) / "fixtures" / "cs_flbit_mixed_pair.asm").read_text(encoding="utf-8").splitlines()
        memory = execute(lines)
        self.assertEqual(memory, {12 * lane + 4 * k: value
                                  for lane in range(64) for k, value in enumerate((7, 7, 1))})

    def test_both_word_orders_and_scc(self):
        for op in OPS:
            for word in ("lo", "hi"):
                for mul, add, mask in mixed.MASKS:
                    for data in mixed.DATA:
                        for scc in (0, 1):
                            with self.subTest(op=op, word=word, mask=(mul, add, mask), data=data, scc=scc):
                                mask64 = mixed.lane_mask(mul, add, mask)
                                value = ((mask64 & 0xFFFFFFFF) | (data << 32) if word == "lo"
                                         else data | (mask64 & (0xFFFFFFFF << 32)))
                                want = leading_bits(value, op.endswith("i64"))
                                # A runtime load remains data even when its value is zero. An immediate zero
                                # beside a mask is tracked as a whole mask and remains Unsupported (#210).
                                memory = execute(program(op, word, mul, add, mask, data, scc, runtime_data=True), data)
                                self.assertEqual(memory, {8 * lane + 4 * k: v for lane in range(64)
                                                          for k, v in enumerate((want, scc))})

    def test_destination_may_alias_either_source_word(self):
        for op in OPS:
            for dest in ("s10", "s11"):
                with self.subTest(op=op, dest=dest):
                    memory = execute(program(op, "hi", 1, 0, 7, 5, dest=dest))
                    self.assertEqual(memory, {8 * lane + 4 * k: v for lane in range(64)
                                              for k, v in enumerate((7, 1))})

    def test_copied_mask_word_keeps_its_original_half(self):
        # A mask's tag, rather than its destination SGPR number, selects the ballot half.
        # The same b64 scan can consume a wave32 mask in either word of its input pair.
        for wave64 in (False, True):
            lanes = 64 if wave64 else 32
            for half in ((0, 1) if wave64 else (0,)):
                for word in ("lo", "hi"):
                    for op in OPS:
                        for data in mixed.DATA:
                            with self.subTest(wave64=wave64, half=half, word=word, op=op, data=data):
                                pattern = (3, 1, 37)  # unequal, nontrivial low and high ballot words
                                lines = program(op, word, *pattern, data, runtime_data=True)
                                compare = "v_cmp_eq_u32_e64 s[10:11], 0, v3"
                                at = lines.index(compare)
                                mask_reg = "s10" if word == "lo" else "s11"
                                lines[at:at + 1] = [
                                    f"v_cmp_eq_u32_e64 {'s[14:15]' if wave64 else 's14'}, 0, v3",
                                    f"s_mov_b32 {mask_reg}, s{14 + half}",
                                ]
                                if not wave64:
                                    lines.remove("v_mbcnt_hi_u32_b32 v1, -1, v1")
                                mask_word = (mixed.lane_mask(*pattern) >> (32 * half)) & 0xFFFFFFFF
                                value = (mask_word | (data << 32) if word == "lo"
                                         else data | (mask_word << 32))
                                want = leading_bits(value, op.endswith("i64"))
                                self.assertEqual(execute(lines, data, wave64),
                                                 {8 * lane + 4 * k: v for lane in range(lanes)
                                                  for k, v in enumerate((want, 1))})

    def test_whole_masks_still_rejected(self):
        for op in OPS:
            for emulate in (False, True):
                with self.subTest(op=op, emulate=emulate), mock.patch.object(rdna2spirv, "WAVE64_EMULATE", emulate):
                    with self.assertRaisesRegex(rdna2spirv.Unsupported, "lane mask"):
                        support.translate_lines("cs_flbit_whole_mask", [f"{op} s20, exec", "s_endpgm"],
                                                fssm=mixed.fssm())

    def test_non_data_pairs_still_rejected(self):
        # Immediate 0/-1 is EITHER, unlike the runtime-loaded DATA above. Those
        # pairs and pairs of two masks retain the whole-mask rejection (#210),
        # including the noncanonical layouts reported in #7224's review.
        for wave64 in (False, True):
            pairs = (("s14", "0"), ("s14", "-1"), ("0", "s14"), ("s14", "s16"))
            if wave64:
                pairs += (("s15", "0"), ("s15", "s16"), ("s14", "s14"), ("s14", "s15"))
            setup = ["v_mbcnt_lo_u32_b32 v1, -1, 0"]
            if wave64:
                setup.append("v_mbcnt_hi_u32_b32 v1, -1, v1")
            setup += [f"v_cmp_gt_u32_e64 {'s[14:15]' if wave64 else 's14'}, 40, v1",
                      f"v_cmp_gt_u32_e64 {'s[16:17]' if wave64 else 's16'}, 5, v1"]
            for lo, hi in pairs:
                for op in OPS:
                    for emulate in (False, True):
                        with self.subTest(wave64=wave64, pair=(lo, hi), op=op, emulate=emulate):
                            lines = setup + [f"s_mov_b32 s10, {lo}", f"s_mov_b32 s11, {hi}",
                                             f"{op} s20, s[10:11]", "s_endpgm"]
                            with mock.patch.object(rdna2spirv, "WAVE64_EMULATE", emulate):
                                with self.assertRaisesRegex(rdna2spirv.Unsupported, "lane mask"):
                                    execute(lines, wave64=wave64)

    def test_emulated_branch_taken_by_one_wave(self):
        # The first wave enters, the second skips. A mixed-pair scan needs a
        # workgroup exchange even though the complete pair is not a lane mask.
        for op in OPS:
            for word in ("lo", "hi"):
                with self.subTest(op=op, word=word):
                    scalar = "s11" if word == "lo" else "s10"
                    insns = [
                        ("v_and_b32_e32 v1, 7, v0", 4),
                        ("v_cmp_eq_u32_e64 s[10:11], 0, v1", 8),
                        (f"s_mov_b32 {scalar}, 5", 4),
                        ("v_lshlrev_b32_e32 v6, 2, v0", 4),
                        ("v_readfirstlane_b32 s20, v0", 4), ("s_cmp_eq_u32 s20, 0", 4),
                        ("s_cbranch_scc0 4", 4), (f"{op} s9, s[10:11]", 4),
                        ("v_mov_b32_e32 v4, s9", 4),
                        ("buffer_store_dword v4, v6, s[0:3], 0 offen", 8),
                        ("s_mov_b64 exec, -1", 4), ("s_endpgm", 4),
                    ]
                    spv, _ = _translate_encoded("cs_flbit_mixed_branch", insns, (128, 1, 1), emulate=True)
                    for reverse in (False, True):
                        memory, _ = run_emulated_workgroup(spv, (128, 1, 1),
                                                           push_constants=mixed.VSHARP + [0] * 28,
                                                           reverse_order=reverse)
                        self.assertEqual(memory, {4 * lane: (29 if word == "lo" else 7) for lane in range(64)})


if __name__ == "__main__":
    unittest.main()

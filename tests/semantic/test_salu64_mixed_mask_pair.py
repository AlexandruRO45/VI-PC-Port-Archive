"""SHADER-823: s_bcnt0/1_i32_b64 and s_ff0/1_i32_b64 of an SGPR pair that holds a lane mask in one word and data in
the other, translated by tools/rdna2spirv.py (support.translate_lines) and run on every lane of a wave64 in
spirv_mini_interp.run_workgroup.

RDNA2 ISA, SOP1: s_bcnt1_i32_b64 D = popcount(S0[63:0]), s_bcnt0_i32_b64 D = popcount(~S0[63:0]), both with
SCC = (D != 0); s_ff1_i32_b64 / s_ff0_i32_b64 D = the lowest set / clear bit of S0[63:0], or -1 when there is none.
S0 is one 64-bit value whatever wrote its words. Here v_cmp_eq_u32_e64 s[10:11] writes a lane mask and s_mov_b32 then
puts a scalar in one of the two words, so the pair is (lanes 0-31 of the mask, data) or (data, lanes 32-63).

The translator keeps a lane mask as each lane's own bit, so it has to collect the whole wave's word before it counts
(Gen.mixed_pair_words). Until SHADER-823 it loaded both words as data here: every lane counted its own 0 / ~0 plus the
scalar, so lanes inside and outside the mask got different answers. Every lane stores its four results and SCC, and
every lane must agree with the model.

Under --wave64-emulate that collection is a cross-lane exchange, so tools/wave64_emulate_opt.py must not treat the
instruction as lane-local; WaveEmulateOptTests checks its predicates and that the emulated translation exchanges.
The emulated SPIR-V needs Workgroup memory, which the interpreter rejects, so it is not executed here.

Standard library only; no GPU, no game files.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402
import wave64_emulate_opt as wo  # noqa: E402

VSHARP = [0, 0, 0x1000, 0x31016FAC]
ALL64 = (1 << 64) - 1
OPS = ("s_bcnt1_i32_b64", "s_bcnt0_i32_b64", "s_ff1_i32_b64", "s_ff0_i32_b64")
SLOT = 4 * (len(OPS) + 1)          # bytes per lane: the four results, then SCC after s_bcnt1

# (multiplier, addend, and-mask): the mask is the lanes l with ((l * multiplier + addend) & and-mask) == 0.
MASKS = ((1, 0, 7), (1, 0, 8), (3, 1, 5), (1, 0, 0), (1, 0, 127), (1, 32, 32), (7, 2, 12))
DATA = (5, 0, 0xFFFFFFFF, 0x80000001)


def fssm(threads=64):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def lane_mask(mul, add, mask):
    return sum(1 << lane for lane in range(64) if ((lane * mul + add) & mask) == 0)


def model(word, mask64, data):
    """The pair's 64 bits and the expected (bcnt1, bcnt0, ff1, ff0, scc) from the ISA."""
    value = (mask64 & 0xFFFFFFFF) | (data << 32) if word == "lo" else data | (mask64 & (0xFFFFFFFF << 32))
    low = lambda v: (v & -v).bit_length() - 1 if v else 0xFFFFFFFF  # noqa: E731
    ones = bin(value).count("1")
    return value, (ones, 64 - ones, low(value), low(~value & ALL64), int(ones != 0))


def program(word, mul, add, mask, data):
    keep = "s_mov_b32 s11" if word == "lo" else "s_mov_b32 s10"
    lines = ["v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1",
             f"v_mad_u32_u24 v3, v1, {mul}, {add}", f"v_and_b32_e32 v3, {mask}, v3",
             "v_cmp_eq_u32_e64 s[10:11], 0, v3", f"{keep}, {data:#x}",
             "s_bcnt1_i32_b64 s20, s[10:11]", "s_cselect_b32 s24, 1, 0", "s_bcnt0_i32_b64 s21, s[10:11]",
             "s_ff1_i32_b64 s22, s[10:11]", "s_ff0_i32_b64 s23, s[10:11]",
             f"v_mul_u32_u24_e32 v21, {SLOT}, v1"]
    for k, reg in enumerate(("s20", "s21", "s22", "s23", "s24")):
        lines += [f"v_mov_b32_e32 v20, {reg}", f"v_add_nc_u32_e32 v22, {4 * k}, v21",
                  "buffer_store_dword v20, v22, s[0:3], 0 offen"]
    return lines + ["s_endpgm"]


def run(word, mul, add, mask, data):
    spv = support.translate_lines("cs_salu64_mixed_pair", program(word, mul, add, mask, data), fssm=fssm())
    memory = si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                 "%sg_lane": i},
                              push_constants=VSHARP + [0] * 28, memory={})
    return [tuple(memory[SLOT * lane + 4 * k] for k in range(len(OPS) + 1)) for lane in range(64)]


class MixedPairEveryLaneTests(unittest.TestCase):
    def test_model_by_hand(self):
        # The finder's case: a comparison mask beside the scalar 5. Lanes 0, 8, 16, 24 in the low word, 5 above.
        self.assertEqual(model("lo", lane_mask(1, 0, 7), 5),
                         (0x5_01010101, (6, 58, 0, 1, 1)))
        self.assertEqual(model("hi", lane_mask(1, 0, 7), 5), (0x01010101_00000005, (6, 58, 0, 1, 1)))
        self.assertEqual(model("lo", 0, 0), (0, (0, 64, 0xFFFFFFFF, 0, 0)))

    def test_every_lane_agrees_with_the_isa(self):
        for word in ("lo", "hi"):
            for mul, add, mask in MASKS:
                for data in DATA:
                    with self.subTest(word=word, mask=(mul, add, mask), data=hex(data)):
                        value, want = model(word, lane_mask(mul, add, mask), data)
                        got = run(word, mul, add, mask, data)
                        bad = [lane for lane, g in enumerate(got) if g != want]
                        self.assertFalse(bad, f"pair {value:#018x}: lane {bad[:1]} got {got[bad[0]] if bad else ''}, "
                                              f"want (bcnt1, bcnt0, ff1, ff0, scc) = {want}")


def _ins(op, src):
    return {"op": op, "args": [{"k": "sgpr", "r": 20, "n": 1}, src], "mods": set()}


class WaveEmulateOptTests(unittest.TestCase):
    PAIR = {"k": "sgpr", "r": 10, "n": 2}

    def test_mixed_pair_is_cross_lane(self):
        for op in OPS:
            for lm in ({10: "lo"}, {11: "hi"}):
                with self.subTest(op=op, mask_words=sorted(lm)):
                    ins = _ins(op, self.PAIR)
                    self.assertTrue(wo._insn_may_barrier(ins, {id(ins): lm}, set()))

    def test_data_pair_and_b32_stay_lane_local(self):
        for op in OPS:
            ins = _ins(op, self.PAIR)
            self.assertFalse(wo._insn_may_barrier(ins, {id(ins): {}}, set()), op)
        for op in ("s_bcnt1_i32_b32", "s_ff1_i32_b32"):
            ins = _ins(op, {"k": "sgpr", "r": 10, "n": 1})
            self.assertFalse(wo._insn_may_barrier(ins, {id(ins): {11: "hi"}}, set()), op)

    def test_whole_mask_is_still_cross_lane(self):
        ins = _ins("s_bcnt1_i32_b64", self.PAIR)
        self.assertTrue(wo._insn_may_barrier(ins, {id(ins): {10: "lo", 11: "hi"}}, set()))

    def test_emulated_translation_exchanges_for_the_mixed_pair(self):
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            mixed = support.translate_lines("cs_w64x_mixed_pair", program("lo", 1, 0, 7, 5), fssm=fssm())
            data = support.translate_lines("cs_w64x_data_pair",
                                           [line for line in program("lo", 1, 0, 7, 5)
                                            if not line.startswith("v_cmp")], fssm=fssm())
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        # Each lane ORs its bit into the wave's mask word in %x_buf (Gen.x_round); a data pair needs no exchange, and
        # emulation uses no subgroup operation at all.
        self.assertGreater(mixed.count("OpAtomicOr"), 0)
        self.assertEqual(data.count("OpAtomicOr"), 0)
        self.assertNotIn("OpGroupNonUniform", mixed)


if __name__ == "__main__":
    unittest.main()

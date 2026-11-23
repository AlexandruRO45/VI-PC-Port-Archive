"""Lane masks in a partial wave: the bits of lanes that were never launched.

A compute group whose size is not a multiple of the wave size ends in a partial wave: lanes c..W-1 of its last
wave (c = group size mod W) were never launched. In a scalar register those lanes are bits like any other. A VALU
compare and EXEC have them 0, but a SALU op computes them with the rest of the word, so s_not_b64 of VCC sets them:
SCC is 1, s_bcnt1 counts them, and s_cmp with 0 sees a non-zero mask. The translator keeps one bit per launched lane
and a shadow word per scalar register for the rest (docs/design.md, "Partial-wave lane masks"). These programs run
every launched lane on the mini interpreter (tests/semantic/spirv_mini_interp.run_workgroup) and compare each wave's
words with a model of the hardware registers: the expected values follow the RDNA2 ISA SALU definitions (bitwise
ops on all W bits, SCC = D != 0) with the never-launched lanes 0 after a VALU compare. They are not hardware
measurements; runtime/test/shaders/cs_w64x_partial_not.asm and cs_w32_partial_not.asm are the GPU fixtures.

No llvm-mc, GPU or game data: the listings are hand-written, the branch listing with its encodings.
"""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import test_wave64_scmp_masks as h  # noqa: E402  (tools path, translate_cs, VSHARP, STUB_DXBC, make_fssm)
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import test_loop_ballot_barrier as lbb  # noqa: E402  (run_compute_workgroup: the interpreter --wave64-emulate needs)

M32, M64 = (1 << 32) - 1, (1 << 64) - 1


def popcount(x):
    return bin(x).count("1")


def wqm(x):
    """s_wqm: every quad (4 consecutive bits) with a bit set becomes 1111."""
    return sum(0xF << q for q in range(0, 64, 4) if (x >> q) & 0xF)


def prologue(wave):
    """v6 = this wave's block of 16 result dwords; vcc = every launched lane (a VALU compare: 0 elsewhere)."""
    shift = 6 if wave == 64 else 5
    text = "v_mbcnt_lo_u32_b32 v1, -1, 0\n" + ("v_mbcnt_hi_u32_b32 v1, -1, v1\n" if wave == 64 else "")
    text += f"v_lshrrev_b32_e32 v6, {shift}, v0\nv_lshlrev_b32_e32 v6, 6, v6\n"
    return text + ("v_cmp_eq_u32_e32 vcc, v1, v1\n" if wave == 64 else "v_cmp_eq_u32_e32 vcc_lo, v1, v1\n")


def store(slot, sreg):
    return (f"v_mov_b32_e32 v4, {sreg}\nv_add_nc_u32_e32 v5, {slot * 4}, v6\n"
            "buffer_store_dword v4, v5, s[0:3], 0 offen\n")


def store_scc(slot):
    return "s_cselect_b32 s20, 1, 0\n" + store(slot, "s20")


def run_waves(spv, threads, wave, s4=0, s5=0, emulate=False):
    pc = h.VSHARP + [s4, s5] + [0] * 26
    if emulate:
        # --wave64-emulate exchanges through workgroup memory and runs on any subgroup size: 32 here, as on NVIDIA
        return lbb.run_compute_workgroup(spv, threads, 32, push_constants=pc, memory={})
    return si.run_workgroup(
        spv, threads, wave,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % wave},
        push_constants=pc, memory={})


def launched(threads, wave):
    """The launched-lane mask of each wave of the group."""
    return [(1 << min(wave, threads - w * wave)) - 1 for w in range(-(-threads // wave))]


def translate(name, asm, threads, emulate=False):
    prev = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        return h.translate_cs(name, asm, threads=threads)
    finally:
        rdna2spirv.WAVE64_EMULATE = prev


def translate_encoded(name, lines, threads):
    """Translate a listing whose lines carry their encodings (branch offsets are placed with them)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n" + "".join(f"\t{ln}\n" for ln in lines))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(h.STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(h.make_fssm(threads))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
            gc.collect()
        return spv


# Each case: (name, wave, body, model). model(L) gives the expected words of a wave whose launched lanes are L.
NOT = lambda L: ~L & M64  # noqa: E731
CASES = [
    ("not_b64", 64, "s_not_b64 s[12:13], vcc\n" + store_scc(0) + "s_bcnt1_i32_b64 s21, s[12:13]\n" + store(1, "s21")
     + "s_bcnt0_i32_b64 s21, s[12:13]\n" + store(2, "s21"),
     lambda L: [int(NOT(L) != 0), popcount(NOT(L)), 64 - popcount(NOT(L))]),
    ("orn2_b64", 64, "s_orn2_b64 s[12:13], 0, vcc\n" + store_scc(0) + "s_bcnt1_i32_b64 s21, s[12:13]\n" + store(1, "s21"),
     lambda L: [int(NOT(L) != 0), popcount(NOT(L))]),
    ("nand_b64", 64, "s_nand_b64 s[12:13], vcc, exec\n" + store_scc(0) + "s_bcnt1_i32_b64 s21, s[12:13]\n"
     + store(1, "s21"), lambda L: [int(NOT(L) != 0), popcount(NOT(L))]),
    ("nor_b64", 64, "s_nor_b64 s[12:13], vcc, 0\n" + store_scc(0) + "s_bcnt1_i32_b64 s21, s[12:13]\n" + store(1, "s21"),
     lambda L: [int(NOT(L) != 0), popcount(NOT(L))]),
    ("xnor_b64", 64, "s_xnor_b64 s[12:13], vcc, vcc\n" + store_scc(0) + "s_bcnt1_i32_b64 s21, s[12:13]\n"
     + store(1, "s21"), lambda L: [1, 64]),
    ("xor_minus1_b64", 64, "s_xor_b64 vcc, vcc, -1\n" + store_scc(0) + "s_bcnt1_i32_b64 s21, vcc\n" + store(1, "s21"),
     lambda L: [int(NOT(L) != 0), popcount(NOT(L))]),
    # a copy keeps the bits; s_cmp with 0 sees them; AND with EXEC clears them; a second NOT gives the mask back
    ("copy_cmp_and_not", 64, "s_not_b64 s[12:13], vcc\ns_mov_b64 s[14:15], s[12:13]\ns_cmp_lg_u64 s[14:15], 0\n"
     + store_scc(0) + "s_and_b64 s[16:17], s[14:15], exec\ns_bcnt1_i32_b64 s21, s[16:17]\n" + store(1, "s21")
     + "s_not_b64 s[18:19], s[14:15]\ns_bcnt1_i32_b64 s21, s[18:19]\n" + store(2, "s21")
     + "s_ff0_i32_b64 s21, s[16:17]\n" + store(3, "s21"),
     lambda L: [int(NOT(L) != 0), 0, popcount(L), 0]),
    # a VALU compare into VCC after the NOT leaves the never-launched bits 0 again (both words)
    ("vcmp_overwrites", 64, "s_not_b64 vcc, vcc\nv_cmp_eq_u32_e32 vcc, v1, v1\ns_bcnt1_i32_b64 s21, vcc\n"
     + store(0, "s21") + "s_cmp_eq_u64 vcc, exec\n" + store_scc(1), lambda L: [popcount(L), 1]),
    # s_cselect picks one source's bits, never-launched ones included
    ("cselect", 64, "s_not_b64 s[12:13], vcc\ns_cmp_eq_u32 s4, 0\ns_cselect_b64 s[14:15], s[12:13], vcc\n"
     "s_bcnt1_i32_b64 s21, s[14:15]\n" + store(0, "s21"), lambda L: [popcount(NOT(L))]),
    # the constant -1 in a mask register has every bit, never-launched ones too
    ("minus1_andn2", 64, "s_mov_b64 s[12:13], -1\ns_andn2_b64 s[14:15], s[12:13], vcc\n" + store_scc(0)
     + "s_bcnt1_i32_b64 s21, s[14:15]\n" + store(1, "s21"), lambda L: [int(NOT(L) != 0), popcount(NOT(L))]),
    # a 32-bit half of a wave64 mask
    ("not_hi_half", 64, "s_not_b32 s13, vcc_hi\n" + store_scc(0) + "s_bcnt1_i32_b32 s21, s13\n" + store(1, "s21"),
     lambda L: [int(~(L >> 32) & M32 != 0), popcount(~(L >> 32) & M32)]),
    ("w32_not", 32, "s_not_b32 s12, vcc_lo\n" + store_scc(0) + "s_bcnt1_i32_b32 s21, s12\n" + store(1, "s21")
     + "s_cmp_lg_u32 s12, 0\n" + store_scc(2), lambda L: [int(~L & M32 != 0), popcount(~L & M32), int(~L & M32 != 0)]),
    ("w32_nand_xnor", 32, "s_nand_b32 s12, vcc_lo, exec_lo\n" + store_scc(0) + "s_bcnt1_i32_b32 s21, s12\n"
     + store(1, "s21") + "s_xnor_b32 s13, vcc_lo, vcc_lo\ns_bcnt1_i32_b32 s21, s13\n" + store(2, "s21"),
     lambda L: [int(~L & M32 != 0), popcount(~L & M32), 32]),
    # a mask parked in VGPR lanes (v_writelane / v_readlane, a shader out of SGPRs) keeps its never-launched bits
    ("parked", 64, "s_not_b64 s[12:13], vcc\nv_writelane_b32 v40, s12, 3\nv_writelane_b32 v40, s13, 4\n"
     "s_mov_b64 s[12:13], 0\nv_readlane_b32 s14, v40, 3\nv_readlane_b32 s15, v40, 4\n"
     "s_bcnt1_i32_b64 s21, s[14:15]\n" + store(0, "s21"), lambda L: [popcount(NOT(L))]),
    # a wave32 vcc_hi is plain data, never a mask word
    ("w32_vcc_hi_data", 32, "s_mov_b32 vcc_hi, 0\ns_not_b32 vcc_hi, vcc_hi\n" + store(0, "vcc_hi") + store_scc(1),
     lambda L: [M32, 1]),
]
# group sizes: one partial wave, a full wave then a partial one, a full wave (the control)
GROUPS = {64: (40, 96, 64), 32: (20, 52, 32)}


class PartialWaveInvertedMasks(unittest.TestCase):
    def check(self, name, wave, body, model, emulate=False):
        for threads in GROUPS[wave]:
            spv = translate(f"cs_pwm_{name}", prologue(wave) + body + "s_endpgm\n", threads, emulate)
            mem = run_waves(spv, threads, wave, emulate=emulate)
            for w, L in enumerate(launched(threads, wave)):
                want = model(L)
                got = [mem.get(w * 64 + 4 * k) for k in range(len(want))]
                with self.subTest(case=name, threads=threads, wave=w, emulate=emulate):
                    self.assertEqual(got, want)

    def test_cases(self):
        for name, wave, body, model in CASES:
            self.check(name, wave, body, model)

    def test_cases_wave64_emulate(self):
        for name, wave, body, model in CASES:
            if wave == 64:
                self.check(name, wave, body, model, emulate=True)

    def test_full_wave_module_unchanged(self):
        """A group of whole waves has no never-launched lane: no shadow, the same module as before."""
        for wave, threads in ((64, 64), (64, 128), (32, 32), (32, 96)):
            body = "s_not_b64 s[12:13], vcc\n" if wave == 64 else "s_not_b32 s12, vcc_lo\n"
            spv = translate("cs_pwm_full", prologue(wave) + body + store_scc(0) + "s_endpgm\n", threads)
            with self.subTest(wave=wave, threads=threads):
                self.assertNotIn("%sgpr_s", spv)
                self.assertNotIn("%vcc_s", spv)


# A join and a loop: s4 != 0 inverts VCC, s5 counts how many times s[12:13] is inverted. Lines with encodings
# (llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64); each invocation stores s21 at 4*i and s22 at 0x200 + 4*i.
BRANCHES = [
    "v_mbcnt_lo_u32_b32 v1, -1, 0 ; encoding: [0x01,0x00,0x65,0xd7,0xc1,0x00,0x01,0x02]",
    "v_mbcnt_hi_u32_b32 v1, -1, v1 ; encoding: [0x01,0x00,0x66,0xd7,0xc1,0x02,0x02,0x02]",
    "v_cmp_eq_u32_e32 vcc, v1, v1 ; encoding: [0x01,0x03,0x84,0x7d]",
    "s_and_b64 s[12:13], vcc, exec ; encoding: [0x6a,0x7e,0x8c,0x87]",
    "s_cmp_eq_u32 s4, 0 ; encoding: [0x04,0x80,0x06,0xbf]",
    "s_cbranch_scc1 1 ; encoding: [0x01,0x00,0x85,0xbf]",
    "s_not_b64 vcc, vcc ; encoding: [0x6a,0x08,0xea,0xbe]",
    "s_bcnt1_i32_b64 s21, vcc ; encoding: [0x6a,0x10,0x95,0xbe]",
    "s_mov_b32 s24, s5 ; encoding: [0x05,0x03,0x98,0xbe]",
    "s_cmp_eq_u32 s24, 0 ; encoding: [0x18,0x80,0x06,0xbf]",
    "s_cbranch_scc1 3 ; encoding: [0x03,0x00,0x85,0xbf]",
    "s_not_b64 s[12:13], s[12:13] ; encoding: [0x0c,0x08,0x8c,0xbe]",
    "s_sub_u32 s24, s24, 1 ; encoding: [0x18,0x81,0x98,0x80]",
    "s_branch -5 ; encoding: [0xfb,0xff,0x82,0xbf]",
    "s_bcnt1_i32_b64 s22, s[12:13] ; encoding: [0x0c,0x10,0x96,0xbe]",
    "v_mov_b32_e32 v4, s21 ; encoding: [0x15,0x02,0x08,0x7e]",
    "v_lshlrev_b32_e32 v5, 2, v0 ; encoding: [0x82,0x00,0x0a,0x34]",
    "buffer_store_dword v4, v5, s[0:3], 0 offen ; encoding: [0x00,0x10,0x70,0xe0,0x05,0x04,0x00,0x80]",
    "v_mov_b32_e32 v4, s22 ; encoding: [0x16,0x02,0x08,0x7e]",
    "v_add_nc_u32_e32 v5, 0x200, v5 ; encoding: [0xff,0x0a,0x0a,0x4a,0x00,0x02,0x00,0x00]",
    "buffer_store_dword v4, v5, s[0:3], 0 offen ; encoding: [0x00,0x10,0x70,0xe0,0x05,0x04,0x00,0x80]",
    "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
]


class PartialWaveWholeQuadMode(unittest.TestCase):
    """s_wqm fills a quad from all four bits. Where the launch ends inside a quad (42 lanes: 40 and 41 launched, 42
    and 43 not) the quad has launched and never-launched bits. In a partial-wave module s_wqm of a lane mask is taken
    from the whole word, with no quad operation, so the stock mini interpreter runs it; a full-wave module keeps
    its quad operations (run_compute_workgroup has OpGroupNonUniformQuadSwap)."""

    # out[0..1]: launched bits fill their quad, never-launched lanes included (group 42: 44); out[2]: the reverse,
    # and out[3] the launched lanes' own bits of it (v_cndmask reads this lane's bit); out[4..5]: a constant leaves
    # launched-lane bits of ~vcc in the shadow, which must not reach the never-launched lanes of their quad
    # (group 10 gives SCC 0 and no bit, group 9 SCC 1 and 4 bits); out[6]: s_wqm_b32 of
    # vcc_hi, the high half.
    BODY64 = ("s_wqm_b64 s[12:13], vcc\n" + store_scc(0) + "s_bcnt1_i32_b64 s21, s[12:13]\n" + store(1, "s21")
              + "s_not_b64 s[14:15], vcc\ns_wqm_b64 s[16:17], s[14:15]\ns_bcnt1_i32_b64 s21, s[16:17]\n"
              + store(2, "s21") + "s_wqm_b64 s[18:19], s[14:15]\nv_cndmask_b32_e64 v7, 0, 1, s[18:19]\n"
              + "v_cmp_ne_u32_e32 vcc, 0, v7\ns_bcnt1_i32_b64 s21, vcc\n" + store(3, "s21")
              + "s_and_b64 s[22:23], s[14:15], 0x300\ns_wqm_b64 s[24:25], s[22:23]\n" + store_scc(4)
              + "s_bcnt1_i32_b64 s21, s[24:25]\n" + store(5, "s21")
              + "v_cmp_eq_u32_e32 vcc, v1, v1\ns_wqm_b32 s26, vcc_hi\ns_bcnt1_i32_b32 s21, s26\n" + store(6, "s21"))
    BODY32 = ("s_wqm_b32 s12, vcc_lo\n" + store_scc(0) + "s_bcnt1_i32_b32 s21, s12\n" + store(1, "s21")
              + "s_not_b32 s14, vcc_lo\ns_wqm_b32 s16, s14\ns_bcnt1_i32_b32 s21, s16\n" + store(2, "s21"))

    @staticmethod
    def want64(L):
        x = NOT(L) & 0x300
        return [1, popcount(wqm(L)), popcount(wqm(NOT(L))), popcount(wqm(NOT(L)) & L), int(wqm(x) != 0),
                popcount(wqm(x)), popcount(wqm(L) >> 32)]

    @staticmethod
    def want32(L):
        return [1, popcount(wqm(L) & M32), popcount(wqm(~L & M32))]

    def run_group(self, wave, body, want, threads, emulate):
        spv = translate(f"cs_pwm_wqm{wave}", prologue(wave) + body + "s_endpgm\n", threads, emulate)
        pc = h.VSHARP + [0] * 28
        if threads % wave and not emulate:
            self.assertNotIn("OpGroupNonUniformQuad", spv)
            mem = run_waves(spv, threads, wave)          # the stock interpreter: no quad operation is left
        else:
            mem = lbb.run_compute_workgroup(spv, threads, 32 if emulate else wave, push_constants=pc, memory={})
        for w, L in enumerate(launched(threads, wave)):
            n = len(want(L))
            with self.subTest(wave_size=wave, threads=threads, wave=w, emulate=emulate):
                self.assertEqual([mem.get(w * 64 + 4 * k) for k in range(n)], want(L))

    def test_wqm_wave64(self):
        for emulate in (False, True):
            for threads in (42, 10, 9, 8, 40, 96, 64):
                self.run_group(64, self.BODY64, self.want64, threads, emulate)

    def test_wqm_wave32(self):
        for threads in (18, 20, 52, 32):
            self.run_group(32, self.BODY32, self.want32, threads, False)

    def test_split_quad_values(self):
        """The translator's words on the stock interpreter, spelled out rather than taken from want64: group 10 ->
        SCC 0 and 0 bits, group 9 -> SCC 1 and 4 bits (out[4..5]), group 42 -> 44 bits (out[1])."""
        for threads, first, want in ((10, 4, [0, 0]), (9, 4, [1, 4]), (42, 1, [44])):
            spv = translate("cs_pwm_wqm64", prologue(64) + self.BODY64 + "s_endpgm\n", threads, False)
            mem = run_waves(spv, threads, 64)
            with self.subTest(threads=threads):
                self.assertEqual([mem.get(4 * (first + k)) for k in range(len(want))], want)


def mask_words(src):
    """out[0] SCC, then the 64-bit mask `src`: its bit count (1) and those of its low (2) and high (3) words."""
    return (store_scc(0) + f"s_bcnt1_i32_b64 s21, {src}\n" + store(1, "s21") + f"s_mov_b64 s[24:25], {src}\n"
            + "s_bcnt1_i32_b32 s22, s24\n" + store(2, "s22") + "s_bcnt1_i32_b32 s23, s25\n" + store(3, "s23"))


def words_of(scc_word, value):
    """The model of mask_words: SCC is s_wqm's (its result != 0, as `scc_word`), `value` the mask read back."""
    return [int(scc_word != 0), popcount(value), popcount(value & M32), popcount(value >> 32)]


def lo(L):
    return L & M32


def hi(L):
    return L >> 32


class PartialWaveWqmHalves(unittest.TestCase):
    """s_wqm of a lane mask whose source is not the half, or the pair, its destination is. A 32-bit s_wqm reads the
    word its source names and writes the lanes its destination names: s_wqm_b32 vcc_lo, vcc_hi fills lanes 0-31
    from lanes 32-63. A 64-bit s_wqm of an SGPR pair put together from 32-bit copies reads each word from its own
    register, so s12 = vcc_hi, s13 = vcc_lo is the mask with its halves swapped. In a partial-wave module the
    result is taken from the gathered words (PartialWaveWholeQuadMode), which at first read the low word whenever
    source and destination named different halves and took every pair as (lanes 0-31, lanes 32-63): in a 40-thread
    wave64 s_wqm_b32 vcc_lo, vcc_hi left vcc_lo = 0xffffffff (0xff by the ISA definition), and in a 20-thread one
    it gave SCC 1 for vcc_hi = 0. Only partial-wave modules: a group of whole waves goes through Gen.wqm."""

    # (name, body, model of a wave whose launched lanes are L)
    CASES = [
        ("b32_vcc_lo_from_vcc_hi", "s_wqm_b32 vcc_lo, vcc_hi\n" + mask_words("vcc"),
         lambda L: words_of(wqm(hi(L)), wqm(hi(L)) | hi(L) << 32)),
        ("b32_vcc_lo_from_copy_of_hi", "s_mov_b32 s15, vcc_hi\ns_wqm_b32 vcc_lo, s15\n" + mask_words("vcc"),
         lambda L: words_of(wqm(hi(L)), wqm(hi(L)) | hi(L) << 32)),
        ("b32_vcc_hi_from_vcc_lo", "s_wqm_b32 vcc_hi, vcc_lo\n" + mask_words("vcc"),
         lambda L: words_of(wqm(lo(L)), lo(L) | wqm(lo(L)) << 32)),
        ("b32_vcc_hi_from_copy_of_lo", "s_mov_b32 s14, vcc_lo\ns_wqm_b32 vcc_hi, s14\n" + mask_words("vcc"),
         lambda L: words_of(wqm(lo(L)), lo(L) | wqm(lo(L)) << 32)),
        ("b32_exec_lo_from_vcc_hi", "s_mov_b64 s[30:31], exec\ns_wqm_b32 exec_lo, vcc_hi\ns_cselect_b32 s20, 1, 0\n"
         "s_bcnt1_i32_b64 s21, exec\ns_bcnt1_i32_b32 s22, exec_lo\ns_bcnt1_i32_b32 s23, exec_hi\n"
         "s_mov_b64 exec, s[30:31]\n" + store(0, "s20") + store(1, "s21") + store(2, "s22") + store(3, "s23"),
         lambda L: words_of(wqm(hi(L)), wqm(hi(L)) | hi(L) << 32)),
        ("b32_vcc_lo_from_not_of_hi", "s_not_b64 s[12:13], vcc\ns_wqm_b32 vcc_lo, s13\n" + mask_words("vcc"),
         lambda L: words_of(wqm(~hi(L) & M32), wqm(~hi(L) & M32) | hi(L) << 32)),
        ("b32_same_half", "s_wqm_b32 vcc_hi, vcc_hi\n" + mask_words("vcc"),
         lambda L: words_of(wqm(hi(L)), lo(L) | wqm(hi(L)) << 32)),
        ("b64_swapped_halves", "s_mov_b32 s12, vcc_hi\ns_mov_b32 s13, vcc_lo\ns_wqm_b64 s[14:15], s[12:13]\n"
         + mask_words("s[14:15]"), lambda L: words_of(L, wqm(hi(L) | lo(L) << 32))),
        ("b64_hi_and_zero", "s_mov_b32 s12, vcc_hi\ns_mov_b32 s13, 0\ns_wqm_b64 s[14:15], s[12:13]\n"
         + mask_words("s[14:15]"), lambda L: words_of(hi(L), wqm(hi(L)))),
        ("b64_zero_and_lo", "s_mov_b32 s12, 0\ns_mov_b32 s13, vcc_lo\ns_wqm_b64 s[14:15], s[12:13]\n"
         + mask_words("s[14:15]"), lambda L: words_of(lo(L), wqm(lo(L) << 32))),
        ("b64_lo_twice", "s_mov_b32 s12, vcc_lo\ns_mov_b32 s13, vcc_lo\ns_wqm_b64 s[14:15], s[12:13]\n"
         + mask_words("s[14:15]"), lambda L: words_of(lo(L), wqm(lo(L) | lo(L) << 32))),
        ("b64_lo_and_minus1", "s_mov_b32 s12, vcc_lo\ns_mov_b32 s13, -1\ns_wqm_b64 s[14:15], s[12:13]\n"
         + mask_words("s[14:15]"), lambda L: words_of(1, wqm(lo(L) | M32 << 32))),
    ]
    # a partial wave ending in the high half (on a quad edge, and inside a quad) and in the low half; a full wave
    # then a partial one
    GROUPS = (40, 42, 20, 10, 96)

    def run_case(self, name, body, threads, emulate):
        spv = translate(f"cs_pwm_{name}", prologue(64) + body + "s_endpgm\n", threads, emulate)
        if emulate:
            return lbb.run_compute_workgroup(spv, threads, 32, push_constants=h.VSHARP + [0] * 28, memory={})
        self.assertNotIn("OpGroupNonUniformQuad", spv)
        return run_waves(spv, threads, 64)

    def test_cases(self):
        for emulate in (False, True):
            for name, body, model in self.CASES:
                for threads in self.GROUPS:
                    mem = self.run_case(name, body, threads, emulate)
                    for w, L in enumerate(launched(threads, 64)):
                        with self.subTest(case=name, threads=threads, wave=w, emulate=emulate):
                            self.assertEqual([mem.get(w * 64 + 4 * k) for k in range(4)], model(L))

    def test_values(self):
        """The same words spelled out, not taken from the model. Group 40: vcc_hi = 0xff, so s_wqm_b32 vcc_lo,
        vcc_hi gives vcc_lo = 0xff (SCC 1, 16 bits in VCC); group 20: vcc_hi = 0, so vcc_lo = 0 and SCC 0."""
        by_name = {name: body for name, body, _model in self.CASES}
        for name, threads, want in (
                ("b32_vcc_lo_from_vcc_hi", 40, [1, 16, 8, 8]), ("b32_vcc_lo_from_vcc_hi", 20, [0, 0, 0, 0]),
                ("b32_vcc_lo_from_copy_of_hi", 40, [1, 16, 8, 8]), ("b32_vcc_lo_from_copy_of_hi", 20, [0, 0, 0, 0]),
                ("b32_vcc_hi_from_vcc_lo", 40, [1, 64, 32, 32]), ("b32_vcc_hi_from_vcc_lo", 20, [1, 40, 20, 20]),
                ("b64_swapped_halves", 40, [1, 40, 8, 32]), ("b64_swapped_halves", 20, [1, 20, 0, 20]),
                ("b64_hi_and_zero", 40, [1, 8, 8, 0]), ("b64_hi_and_zero", 20, [0, 0, 0, 0])):
            for emulate in (False, True):
                mem = self.run_case(name, by_name[name], threads, emulate)
                with self.subTest(case=name, threads=threads, emulate=emulate):
                    self.assertEqual([mem.get(4 * k) for k in range(4)], want)


class PartialWaveControlFlow(unittest.TestCase):
    def test_join_and_loop(self):
        """The shadow is stored next to the register, so it follows each path through a join and a loop."""
        for threads in (40, 96):
            spv = translate_encoded("cs_pwm_branches", BRANCHES, threads)
            for s4 in (0, 1):
                for s5 in (0, 1, 2, 3):
                    mem = run_waves(spv, threads, 64, s4=s4, s5=s5)
                    for w, L in enumerate(launched(threads, 64)):
                        first = 4 * 64 * w
                        want = [popcount(NOT(L) if s4 else L), popcount(NOT(L) if s5 % 2 else L)]
                        with self.subTest(threads=threads, s4=s4, s5=s5, wave=w):
                            self.assertEqual([mem.get(first), mem.get(0x200 + first)], want)


if __name__ == "__main__":
    unittest.main()

"""ds_swizzle_b32 run across every lane of the translated SPIR-V, on full and on partial waves (#618).

Programs are written here (no game data) and translated by tools/rdna2spirv.py; spirv_mini_interp.run_workgroup runs
one workgroup. Each lane stores its result to out[lane]. The expected source lane of each pattern is computed here
from what llvm-mc's swizzle() macros mean (tools/lds_fuzz.py's swizzle_src has the same rules for the encoded
offset), not taken from the translator:
  * QUAD_PERM,a,b,c,d: lane 4q + k reads lane 4q + (a, b, c, d)[k];
  * BITMASK_PERM,"54321": one character per lane-index bit, bit 4 first: p keeps the bit, i inverts it, 0 and 1
    force it; bit 5 (the half of a wave64) is kept;
  * SWAP,n: lane ^ n. REVERSE,n: lane ^ (n - 1). BROADCAST,n,k: lane k of the lane's group of n.

A source lane that is not running reads 0. That is the case in the missing lanes of a partial wave (a group whose
thread count isn't a multiple of the wave), and for the upper lanes where a driver runs a pixel shader narrower
than the wave (#618; a pixel shader compares with SubgroupSize instead of taking a ballot, see Translation).
SPIR-V leaves a shuffle from such a lane undefined, and the interpreter raises when an undefined value is stored:
with the shuffle as it was before (no ballot of the running lanes) every PartialWave case here ends in "OpStore
uses an undefined value". The --wave64-emulate output needs Workgroup memory, which the interpreter rejects, so it
is not run here.
"""
import unittest

from support import translate_listing
from test_ds_permute import translate_text, run

OLD = 99


def data(lane):
    return 7 * lane + 1


def program(pattern):
    """v2 = 7 * lane + 1 is the data; v10 starts at OLD; the result of lane l is stored at out[l]."""
    return ["v_mad_u32_u24 v2, v0, 7, 1", f"v_mov_b32_e32 v10, {OLD}",
            f"ds_swizzle_b32 v10, v2 offset:swizzle({pattern})", "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0", "buffer_store_dword v10, v21, s[0:3], 0 offen", "s_endpgm"]


def bitmask(chars):
    def src(lane):
        out = lane & ~31
        for k, ch in enumerate(chars):
            bit = 1 << (4 - k)
            if ch == "1" or (ch == "p" and lane & bit) or (ch == "i" and not lane & bit):
                out |= bit
        return out
    return src


# name -> (the swizzle() arguments, the source lane of a lane)
PATTERNS = {
    "quad_reverse": ("QUAD_PERM,3,2,1,0", lambda l: (l & ~3) | (3, 2, 1, 0)[l & 3]),
    "quad_mixed": ("QUAD_PERM,1,1,2,0", lambda l: (l & ~3) | (1, 1, 2, 0)[l & 3]),
    "bitmask_pippi": ('BITMASK_PERM,"pippi"', bitmask("pippi")),
    "bitmask_10p0i": ('BITMASK_PERM,"10p0i"', bitmask("10p0i")),
    "bitmask_0pppp": ('BITMASK_PERM,"0pppp"', bitmask("0pppp")),
    "swap_8": ("SWAP,8", lambda l: l ^ 8),
    "reverse_8": ("REVERSE,8", lambda l: l ^ 7),
    "reverse_32": ("REVERSE,32", lambda l: l ^ 31),
    "broadcast_8_3": ("BROADCAST,8,3", lambda l: (l & ~7) | 3),
}


def want(src, threads):
    """Every lane is on in EXEC: a lane gets its source lane's data, or 0 when that lane isn't running."""
    return [data(src(l)) if src(l) < threads else 0 for l in range(threads)]


class FullWave(unittest.TestCase):
    def test_wave32(self):
        for name, (pattern, src) in PATTERNS.items():
            with self.subTest(pattern=name):
                got = run(translate_text("cs_" + name, program(pattern), 32), 32, 32)
                self.assertEqual(got, want(src, 32))

    def test_wave64_swizzles_inside_each_half(self):
        for name in ("bitmask_10p0i", "reverse_32", "quad_mixed"):
            pattern, src = PATTERNS[name]
            with self.subTest(pattern=name):
                got = run(translate_text("cs_" + name, program(pattern), 64), 64, 64)
                self.assertEqual(got, want(src, 64))
                self.assertTrue(all((src(l) ^ l) < 32 for l in range(64)))


class PartialWave(unittest.TestCase):
    """The last wave of a group is short: its missing lanes are not running."""

    def test_a_source_lane_that_is_not_running_reads_zero(self):
        # 16 threads: "10p0i" reads lanes 16-31 only, as on a driver that runs the stage 16 wide (#618).
        # 18 and 20 threads: some lanes of the last quad, or of the upper half, are there and some are not.
        for threads in (16, 18, 20, 31):
            for name, (pattern, src) in PATTERNS.items():
                with self.subTest(threads=threads, pattern=name):
                    got = run(translate_text("cs_" + name, program(pattern), threads), threads, 32)
                    self.assertEqual(got, want(src, threads))

    def test_the_patterns_reach_missing_lanes(self):
        # the test above would say nothing if no pattern left the running lanes
        for threads in (16, 18, 20, 31):
            missing = {name for name, (_, src) in PATTERNS.items() if any(src(l) >= threads for l in range(threads))}
            self.assertIn("reverse_32", missing)
            self.assertGreaterEqual(len(missing), 2, (threads, missing))

    def test_a_short_second_wave(self):
        # 40 threads in 32-lane waves: wave 0 is full, wave 1 has lanes 0-7
        pattern, src = PATTERNS["swap_8"]
        got = run(translate_text("cs_swap", program(pattern), 40), 40, 32)
        self.assertEqual(got[:32], [data(l ^ 8) for l in range(32)])
        self.assertEqual(got[32:], [0] * 8)

    def test_wave64_partial(self):
        # 40 threads of one wave64: the upper half has lanes 32-39
        for name in ("bitmask_10p0i", "quad_reverse", "reverse_8"):
            pattern, src = PATTERNS[name]
            with self.subTest(pattern=name):
                got = run(translate_text("cs_" + name, program(pattern), 40), 40, 64)
                self.assertEqual(got, want(src, 40))


class Translation(unittest.TestCase):
    def test_the_shuffle_is_asked_only_for_running_lanes(self):
        spv = translate_text("cs_guard", program('BITMASK_PERM,"10p0i"'), 32)
        lines = [l.split() for l in spv.splitlines()]
        shuffles = [l for l in lines if "OpGroupNonUniformShuffle" in l]
        self.assertEqual(len(shuffles), 1)
        index = shuffles[0][-1]
        select = next(l for l in lines if l[0] == index)
        self.assertEqual(select[2], "OpSelect")
        extract = next(l for l in lines if l[0] == select[4])
        self.assertEqual(extract[2], "OpGroupNonUniformBallotBitExtract")
        self.assertIn("OpCapability GroupNonUniformBallot", spv)

    def test_a_pixel_shader_asks_the_subgroup_size_not_a_ballot(self):
        # The quad swizzles of a derivative read helper pixels, and whether a helper invocation counts in a ballot
        # is the implementation's choice before SPIR-V 1.6: a ballot could zero a neighbour the hardware reads.
        # A lane at or past SubgroupSize is the one case that is certainly not there (a driver that runs the wave32
        # pixel shader 8 or 16 wide), so that is what a pixel shader asks.
        lines = ["v_mbcnt_lo_u32_b32 v0, -1, 0", 'ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM,"10p0i")',
                 "ds_swizzle_b32 v2, v0 offset:swizzle(QUAD_PERM,1,0,3,2)", "v_mov_b32_e32 v3, 0",
                 "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
        text = "; ps_swizzle  hand-written\n" + "".join("\t" + line + "\n" for line in lines)
        spv = translate_listing("ps_swizzle", text.encode())
        self.assertNotIn("OpGroupNonUniformBallot", spv)
        self.assertIn("OpDecorate %sg_size BuiltIn SubgroupSize", spv)
        self.assertIn("OpDecorate %sg_size Flat", spv)
        rows = [l.split() for l in spv.splitlines()]
        entry = next(r for r in rows if r[:1] == ["OpEntryPoint"])
        self.assertIn("%sg_size", entry)
        shuffles = [r for r in rows if "OpGroupNonUniformShuffle" in r]
        self.assertEqual(len(shuffles), 2)
        for shuffle in shuffles:
            select = next(r for r in rows if r[0] == shuffle[-1])
            self.assertEqual(select[2], "OpSelect")
            bound = next(r for r in rows if r[0] == select[4])
            self.assertEqual(bound[2], "OpULessThan")
            size = next(r for r in rows if r[0] == bound[-1])
            self.assertEqual(size[2:], ["OpLoad", "%uint", "%sg_size"])

    def test_a_compute_shader_does_not_declare_the_subgroup_size(self):
        self.assertNotIn("%sg_size", translate_text("cs_guard", program("SWAP,8"), 32))


if __name__ == "__main__":
    unittest.main()

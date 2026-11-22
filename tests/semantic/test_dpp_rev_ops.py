"""DPP on the reversed integer VOP2 ops selects the second source operand.

Measured on RDNA2 (gfx1036, raw wave32 and wave64 kernels through tools/hw_oracle; row_shr, quad_perm and row_mirror,
with and without bound_ctrl): for v_subrev_nc_u32, v_lshlrev_b32, v_lshrrev_b32 and v_ashrrev_i32 the DPP lane select,
and the zero bound_ctrl:1 puts in for an invalid source, apply to S1, the operand the op reads as its left-hand
value (D = S1 - S0, D = S1 shifted by S0). v_sub_nc_u32 and the float v_subrev_f32 select S0 like every other op.

The two GPU fixtures (runtime/test/shaders/cs_dpp_rev.asm, cs_dpp_rev_w64.asm) run here across every lane in
spirv_mini_interp.run_workgroup against the constants derived by hand in their headers and against a plain-Python
reference; a second set of programs checks each op alone under three DPP controls.
"""
import struct
import unittest

import support
import spirv_mini_interp as si

M = 0xFFFFFFFF
VSHARP = [0, 0, 0x1000, 0x31016FAC]
# The words derived in the .asm headers.
REV32 = [0x00DD00EE, 0xE0640227, 0xEE300829, 0xF617182B, 0xA8B0042D, 0xD0540A2F, 0xE6281831, 0xF2133833]
REV64 = [0x000500EE, 0x602C4267, 0xAE2D8869, 0xD62E186B, 0xA82F246D, 0x50304A6F, 0xA6319871, 0xD2323873]


def fssm(threads):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def run(spv, lanes):
    return si.run_workgroup(
        spv, lanes, lanes,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})


def s32(x):
    x &= M
    return x - (1 << 32) if x >> 31 else x


def shr1(lane):
    """row_shr:1: (source lane, valid)."""
    return lane - 1, lane % 16 >= 1


def ref32(lane):
    """cs_dpp_rev, one lane: the ISA ops with the lane select on the second source operand."""
    sl, ok = shr1(lane)
    S = lane & 3
    d1 = ((40 + 3 * sl) - lane) & M if ok else 0xEE
    d2 = (((sl + 1) if ok else 0) << S) & M
    d3 = (200 - 8 * sl) >> S if ok else 0xDD
    d4 = (s32(-(64 + 8 * sl)) if ok else 0) >> S
    return (d1 | (d2 & 0xFF) << 8 | d3 << 16 | (d4 & 0xFF) << 24) & M


def ref64(lane):
    """cs_dpp_rev_w64, one lane; v_subrev_f32_dpp is the control that selects the first operand."""
    sl, ok = shr1(lane)
    S = lane & 3
    d1 = ((40 + 3 * sl) - lane) & M if ok else 0xEE
    d2 = (((sl + 1) if ok else 0) << S) & M
    f = int((2 * lane + 10.0) - float(sl)) if ok else 5
    d4 = (s32(-(64 + 8 * sl)) if ok else 0) >> S
    return (d1 | (d2 & 0xFF) << 8 | f << 16 | (d4 & 0xFF) << 24) & M


class GpuFixtures(unittest.TestCase):
    def test_reference_gives_the_hand_derived_words(self):
        self.assertEqual([ref32(l) for l in range(8)], REV32)
        self.assertEqual([ref64(l) for l in range(32, 40)], REV64)

    def test_cs_dpp_rev(self):
        mem = run(support.translate_fixture("cs_dpp_rev", fssm=fssm(32)), 32)
        got = [mem.get(4 * l) for l in range(32)]
        self.assertEqual(got[:8], REV32)
        self.assertEqual(got, [ref32(l) for l in range(32)])

    def test_cs_dpp_rev_w64(self):
        mem = run(support.translate_fixture("cs_dpp_rev_w64", fssm=fssm(64)), 64)
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])       # only lanes 32-39 store
        self.assertEqual([mem[4 * k] for k in range(8)], REV64)


CONTROLS = {
    "row_shr:3": lambda l: (l - 3, l % 16 >= 3),
    "quad_perm:[2,3,0,1]": lambda l: ((l & ~3) | ((l & 3) ^ 2), True),
    "row_mirror": lambda l: ((l & ~15) | (15 - (l & 15)), True),
}
# name -> (D from the two source operands in assembler order, which operand the lane select applies to)
OPS = {
    "v_subrev_nc_u32": (lambda a, b: (b - a) & M, 1),
    "v_lshlrev_b32": (lambda a, b: (b << (a & 31)) & M, 1),
    "v_lshrrev_b32": (lambda a, b: (b & M) >> (a & 31), 1),
    "v_ashrrev_i32": (lambda a, b: (s32(b) >> (a & 31)) & M, 1),
    "v_sub_nc_u32": (lambda a, b: (a - b) & M, 0),
    "v_add_nc_u32": (lambda a, b: (a + b) & M, 0),
}
OLD = 0x7777


def a_of(lane):
    return (lane * 5 + 1) & 31          # also a shift count


def b_of(lane):
    return (0x90000000 + 0x01010101 * lane) & M


class EachOp(unittest.TestCase):
    def check(self, name, control, bound, wave):
        fn, which = OPS[name]
        lanes = wave
        lines = ["v_mbcnt_lo_u32_b32 v0, -1, 0"] + (["v_mbcnt_hi_u32_b32 v0, -1, v0"] if wave == 64 else [])
        lines += ["v_mad_u32_u24 v1, v0, 5, 1", "v_and_b32 v1, 31, v1", "v_mad_u32_u24 v2, v0, 0x10101, 0",
                  "v_lshlrev_b32 v3, 24, v0", "v_add_nc_u32 v2, v2, v3", "v_add_nc_u32 v2, 0x90000000, v2",
                  f"v_mov_b32 v10, {OLD:#x}",
                  f"{name}_dpp v10, v1, v2 {control} row_mask:0xf bank_mask:0xf" + (" bound_ctrl:1" if bound else ""),
                  "v_lshlrev_b32 v8, 2, v0", "buffer_store_dword v10, v8, s[0:3], 0 offen", "s_endpgm"]
        mem = run(support.translate_lines("cs_dpp_rev_each", lines, fssm=fssm(lanes)), lanes)
        want = []
        for lane in range(lanes):
            sl, ok = CONTROLS[control](lane)
            if not ok and not bound:
                want.append(OLD)
                continue
            a, b = a_of(lane), b_of(lane)
            if which == 0:
                a = a_of(sl) if ok else 0
            else:
                b = b_of(sl) if ok else 0
            want.append(fn(a, b))
        self.assertEqual([mem.get(4 * l) for l in range(lanes)], want, f"{name} {control} bound_ctrl:{int(bound)}")

    def test_wave32(self):
        for name in OPS:
            for control in CONTROLS:
                for bound in (False, True):
                    with self.subTest(op=name, control=control, bound=bound):
                        self.check(name, control, bound, 32)

    def test_wave64(self):
        for name in OPS:
            with self.subTest(op=name):
                self.check(name, "row_shr:3", True, 64)

    def test_operand_data(self):
        """The reference's b_of is what the program builds (lane * 0x01010101 + 0x90000000)."""
        self.assertEqual(b_of(3), 0x93030303)
        self.assertEqual(a_of(7), 4)


if __name__ == "__main__":
    unittest.main()

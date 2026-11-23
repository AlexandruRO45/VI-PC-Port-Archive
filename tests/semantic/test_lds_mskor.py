"""Value tests for RDNA2 ds_mskor_b32 / ds_mskor_rtn_b32, run through spirv_mini_interp on the CPU.

RDNA2 ISA, "Data Share Instructions": tmp = M[A]; M[A] = (tmp & ~DATA) | DATA2; the _rtn_ form returns tmp in vdst.
The contention tests compare two lanes' return values and final LDS word with both serial orders.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_atomics_values import run_atomic  # noqa: E402  (also puts tools/ on sys.path)
from test_lds_atomics_cas import run_cs  # noqa: E402

M32 = 0xFFFFFFFF
VALUES = [0, M32, 0x12345678, 0xF0F0F0F0, 0x0000FFFF, 0x80000001]


def mskor(m, mask, data):
    return (m & ~mask | data) & M32


def run_pair(initial, masks, data, returning, active=3):
    """Two lanes update LDS[0]. Each publishes its v5 and the word after both updates."""
    op = "ds_mskor_rtn_b32 v5, v1, v3, v4" if returning else "ds_mskor_b32 v1, v3, v4"
    body = [
        "v_mov_b32_e32 v1, 0",
        f"v_mov_b32_e32 v2, {initial:#x}",
        "s_mov_b32 exec_lo, 1",  # Only lane 0 initializes the shared word.
        "ds_write_b32 v1, v2",
        "s_waitcnt lgkmcnt(0)",
        "s_mov_b32 exec_lo, 3",
        "s_barrier",
        "v_cmp_eq_u32_e32 vcc_lo, 1, v0",
        f"v_mov_b32_e32 v3, {masks[0]:#x}",
        f"v_mov_b32_e32 v6, {masks[1]:#x}",
        "v_cndmask_b32_e32 v3, v3, v6, vcc_lo",
        f"v_mov_b32_e32 v4, {data[0]:#x}",
        f"v_mov_b32_e32 v6, {data[1]:#x}",
        "v_cndmask_b32_e32 v4, v4, v6, vcc_lo",
        "v_mov_b32_e32 v5, 0xdead",
        f"s_mov_b32 exec_lo, {active}",
        op,
        "s_waitcnt lgkmcnt(0)",
        "s_mov_b32 exec_lo, 3",
        "s_barrier",
        "ds_read_b32 v7, v1",
        "s_waitcnt lgkmcnt(0)",
        "v_lshlrev_b32_e32 v6, 2, v0",
        "global_store_dword v6, v5, s[0:1]",
        "global_store_dword v6, v7, s[0:1] offset:8",
    ]
    memory = run_cs(body, nlanes=2)
    return (memory[0], memory[4]), (memory[8], memory[12])


class LdsMskor(unittest.TestCase):
    def test_contended_updates_match_a_serial_order(self):
        for initial, masks, data in (
            (0, (3, 3), (1, 2)),  # ORing both values would give 3, impossible for a serial replacement.
            (0x34, (0x20, 0x30), (0x21, 3)),  # Unequal masks and set bits outside the masks.
        ):
            for returning in (False, True):
                with self.subTest(initial=initial, returning=returning):
                    allowed = set()
                    for first, second in ((0, 1), (1, 0)):
                        middle = mskor(initial, masks[first], data[first])
                        final = mskor(middle, masks[second], data[second])
                        old = [0xdead, 0xdead]
                        if returning:
                            old[first], old[second] = initial, middle
                        allowed.add((tuple(old), (final, final)))
                    self.assertIn(run_pair(initial, masks, data, returning), allowed)

    def test_inactive_lane_preserves_destination_and_lds(self):
        for returning in (False, True):
            with self.subTest(returning=returning):
                old, final = run_pair(7, (3, M32), (1, 0), returning, active=1)
                self.assertEqual(old, (7 if returning else 0xdead, 0xdead))
                self.assertEqual(final, (5, 5))

    def test_returning_form(self):
        for m in VALUES:
            for mask in VALUES[::2]:
                for data in VALUES[1::2]:
                    with self.subTest(m=hex(m), mask=hex(mask), data=hex(data)):
                        got = run_atomic("ds_mskor_rtn_b32 v5, v1, v3, v4", m, data=(mask, data))
                        self.assertEqual(got, (m, mskor(m, mask, data)))

    def test_non_returning_form_leaves_vdst_alone(self):
        for m, mask, data in ((0x12345678, 0x0000FF00, 0x0000AB00), (M32, M32, 0), (0, 0, 0x55)):
            with self.subTest(m=hex(m)):
                self.assertEqual(run_atomic("ds_mskor_b32 v1, v3, v4", m, data=(mask, data)),
                                 (0xDEAD, mskor(m, mask, data)))

    def test_data_outside_the_mask_and_offset(self):
        # DATA2 bits outside DATA are ORed in too: (tmp & ~DATA) | DATA2, not a masked insert.
        self.assertEqual(run_atomic("ds_mskor_rtn_b32 v5, v1, v3, v4", 0x00FF00FF, data=(0x0000FFFF, 0xFFFF0000)),
                         (0x00FF00FF, 0xFFFF0000 | 0x00FF0000))
        self.assertEqual(run_atomic("ds_mskor_rtn_b32 v5, v1, v3, v4", 7, addr=0x20, offset=0x24, data=(3, 8)),
                         (7, 12))

    def test_seeded(self):
        rng = random.Random(0x3502)
        for _ in range(20):
            m, mask, data = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(m=hex(m), mask=hex(mask), data=hex(data)):
                self.assertEqual(run_atomic("ds_mskor_rtn_b32 v5, v1, v3, v4", m, data=(mask, data)),
                                 (m, mskor(m, mask, data)))


if __name__ == "__main__":
    unittest.main()

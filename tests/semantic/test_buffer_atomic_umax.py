"""Value tests for RDNA2 buffer_atomic_umax, run through spirv_mini_interp on the CPU.

RDNA2 MUBUF atomic: M = max(M, DATA) (unsigned 32-bit) at the MUBUF element address.
With 'glc' the pre-op value lands in vdata; without 'glc' vdata is preserved.
Active OOB lanes (offset + 4 > NUM_RECORDS for RAW / OOB_SELECT=3) leave memory
untouched and write 0 with glc. Inactive lanes (EXEC=0) never touch memory and
keep vdata. Standard library only; no GPU, no LLVM, no game files.
"""
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)   # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)

# Buffer layout: element at ELEM, store-back destination at OUT.  RAW descriptor (OOB_SELECT=3).
ELEM, OUT, RECORDS = 0x40, 0x80, 0x100


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_bumax.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_bumax.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_bumax.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_bumax\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(body, memory):
    """Run `body` as one invocation (lane 0) over `memory` (buffer base = 0)."""
    spv = translate(body)
    mem = dict(memory)
    si.run_workgroup(spv, 1, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def buf_atomic_umax(m, d, glc=True, offset=ELEM, records=RECORDS):
    """Translate and execute buffer_atomic_umax over memory word `m` with operand `d`.

    Returns (returned_in_vdata, memory_after_op) as (int|None, int|None).
    """
    body = [
        "s_mov_b32 s8, 0", "s_mov_b32 s9, 0",
        f"s_mov_b32 s10, {records:#x}", "s_mov_b32 s11, 0x31016fac",
        f"v_mov_b32_e32 v1, {offset:#x}", f"v_mov_b32_e32 v2, {d:#x}",
        f"buffer_atomic_umax v2, v1, s[8:11], 0 offen" + (" glc" if glc else ""),
        "s_waitcnt vmcnt(0)",
        f"v_mov_b32_e32 v4, {OUT:#x}", "s_mov_b32 s10, 0x1000",
        "buffer_store_dword v2, v4, s[8:11], 0 offen",
    ]
    mem = run(body, {offset: m})
    return mem.get(OUT), mem.get(offset)


# Key test values covering the unsigned 32-bit domain
VALUES = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, M32]


class BufferAtomicUmaxReturning(unittest.TestCase):
    """Test buffer_atomic_umax with glc=True: pre-op value is returned in vdata."""

    def test_basic_values(self):
        for m in VALUES:
            for d in VALUES[::2]:
                with self.subTest(m=hex(m), d=hex(d)):
                    expected_ret = m          # pre-op value returned
                    expected_mem = max(m, d)  # unsigned max written
                    ret, mem_after = buf_atomic_umax(m, d, glc=True)
                    self.assertEqual(ret, expected_ret)
                    self.assertEqual(mem_after, expected_mem)

    def test_idempotent_when_data_equal(self):
        """max(M, M) leaves memory unchanged and returns M."""
        for v in VALUES:
            with self.subTest(v=hex(v)):
                ret, mem_after = buf_atomic_umax(v, v, glc=True)
                self.assertEqual(ret, v)
                self.assertEqual(mem_after, v)

    def test_unsigned_comparison_at_sign_boundary(self):
        """0x80000000 is larger than 0x7FFFFFFF in unsigned arithmetic."""
        m, d = 0x7FFFFFFF, 0x80000000
        ret, mem_after = buf_atomic_umax(m, d, glc=True)
        self.assertEqual(ret, m)
        self.assertEqual(mem_after, 0x80000000)

        # Reverse: memory already holds the larger unsigned value
        ret2, mem_after2 = buf_atomic_umax(0x80000000, 0x7FFFFFFF, glc=True)
        self.assertEqual(ret2, 0x80000000)
        self.assertEqual(mem_after2, 0x80000000)

    def test_max_u32_always_wins(self):
        """M32 is the fixed point: max(M32, d) == M32 for any d."""
        for d in VALUES:
            with self.subTest(d=hex(d)):
                ret, mem_after = buf_atomic_umax(M32, d, glc=True)
                self.assertEqual(ret, M32)
                self.assertEqual(mem_after, M32)

    def test_zero_always_loses(self):
        """max(0, d) == d for any d."""
        for d in VALUES:
            with self.subTest(d=hex(d)):
                ret, mem_after = buf_atomic_umax(0, d, glc=True)
                self.assertEqual(ret, 0)
                self.assertEqual(mem_after, d)

    def test_seeded_random(self):
        rng = random.Random(0xBEEFCAFE)
        for _ in range(16):
            m = rng.getrandbits(32)
            d = rng.getrandbits(32)
            with self.subTest(m=hex(m), d=hex(d)):
                ret, mem_after = buf_atomic_umax(m, d, glc=True)
                self.assertEqual(ret, m)
                self.assertEqual(mem_after, max(m, d))


class BufferAtomicUmaxNonReturning(unittest.TestCase):
    """Test buffer_atomic_umax without glc: vdata is preserved, memory is still updated."""

    def test_vdata_preserved_when_no_glc(self):
        """Without glc, v2 keeps the original DATA value (d); memory still updated."""
        m, d = 0x00001234, 0x00005678
        ret, mem_after = buf_atomic_umax(m, d, glc=False)
        # vdata preserved as d (the value we stored to OUT from v2)
        self.assertEqual(ret, d)
        self.assertEqual(mem_after, max(m, d))

    def test_no_glc_grid(self):
        for m in VALUES[::2]:
            for d in VALUES[::2]:
                with self.subTest(m=hex(m), d=hex(d)):
                    ret, mem_after = buf_atomic_umax(m, d, glc=False)
                    self.assertEqual(ret, d)   # vdata unchanged
                    self.assertEqual(mem_after, max(m, d))


class BufferAtomicUmaxBounds(unittest.TestCase):
    """OOB and boundary behaviour for RAW descriptor (OOB_SELECT=3, num_records=RECORDS)."""

    def test_exactly_in_range(self):
        """offset + 4 == num_records: in-range (last valid 4-byte slot)."""
        m, d = 0x00001111, 0xAAAAAAAA
        offset = RECORDS - 4
        ret, mem_after = buf_atomic_umax(m, d, glc=True, offset=offset, records=RECORDS)
        self.assertEqual(ret, m)
        self.assertEqual(mem_after, max(m, d))

    def test_oob_one_byte_past(self):
        """offset + 4 > num_records: OOB — memory unchanged, glc lane gets 0."""
        m, d = 0xDEADBEEF, 0x12345678
        offset = RECORDS - 3   # only 3 bytes remain: OOB for 4-byte access
        ret, mem_after = buf_atomic_umax(m, d, glc=True, offset=offset, records=RECORDS)
        self.assertEqual(ret, 0)          # OOB glc return
        self.assertEqual(mem_after, m)    # memory untouched

    def test_oob_at_num_records(self):
        """offset == num_records: clearly OOB."""
        m, d = 0x55555555, 0xAAAAAAAA
        ret, mem_after = buf_atomic_umax(m, d, glc=True, offset=RECORDS, records=RECORDS)
        self.assertEqual(ret, 0)
        self.assertEqual(mem_after, m)

    def test_oob_no_glc_vdata_preserved(self):
        """OOB without glc: v2 keeps d, memory untouched."""
        m, d = 0xCAFE0000, 0x0000BABE
        offset = RECORDS  # OOB
        ret, mem_after = buf_atomic_umax(m, d, glc=False, offset=offset, records=RECORDS)
        self.assertEqual(ret, d)   # vdata preserved
        self.assertEqual(mem_after, m)  # memory untouched


class BufferAtomicUmaxSPIRVShape(unittest.TestCase):
    """Check that the translator emits OpAtomicUMax (not OpAtomicSMax or similar)."""

    def test_emits_opatomicumax(self):
        body = [
            "s_mov_b32 s8, 0", "s_mov_b32 s9, 0",
            "s_mov_b32 s10, 0x100", "s_mov_b32 s11, 0x31016fac",
            "v_mov_b32_e32 v1, 0x40", "v_mov_b32_e32 v2, 0x1234",
            "buffer_atomic_umax v2, v1, s[8:11], 0 offen glc",
            "s_waitcnt vmcnt(0)",
        ]
        spv = translate(body)
        self.assertIn("OpAtomicUMax", spv, "translator must emit OpAtomicUMax for buffer_atomic_umax")
        self.assertNotIn("OpAtomicSMax", spv, "must not emit OpAtomicSMax for an unsigned atomic")

    def test_unsupported_forms_rejected(self):
        """buffer_atomic_umax_x2 is not a valid RDNA2 instruction; translator must raise."""
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(["v_mov_b32_e32 v1, 0x40",
                       "buffer_atomic_umax_x2 v[2:3], v1, s[8:11], 0 offen glc"])


if __name__ == "__main__":
    unittest.main()

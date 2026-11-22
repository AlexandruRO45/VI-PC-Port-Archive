"""Translation tests for RDNA2 image_atomic_swap / image_atomic_inc / image_atomic_dec.

RDNA2 ISA (MIMG): swap is an exchange; inc wraps M to 0 when M >= DATA, dec wraps it to DATA when M == 0 or
M > DATA — the same rules as ds_inc_u32/ds_dec_u32 and global_atomic_{inc,dec}. None of the three has an
operand that leaves memory unchanged, so (like image_atomic_cmpswap) they run only in lanes EXEC leaves on;
with glc the texel's pre-op value lands in vdata. CPU-checked by this file; no GPU, no LLVM, no game files.
"""
import os
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)


def translate(body, *, emulate=False):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_cas.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_cas.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_cas.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_cas\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(src)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev


def body(op, dim="2D", addr="v[1:2]", mods="unorm", glc=True):
    return [
        "v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0", "v_mov_b32_e32 v3, 0",
        f"{op} v5, {addr}, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_{dim} {mods}" + (" glc" if glc else ""),
        "s_waitcnt vmcnt(0)",
    ]


class ImageAtomicSwapTests(unittest.TestCase):
    def test_swap_2d_and_3d(self):
        for dim, addr in (("2D", "v[1:2]"), ("3D", "v[1:3]")):
            for glc in (True, False):
                spv = translate(body("image_atomic_swap", dim, addr, glc=glc))
                self.assertIn("OpImageTexelPointer", spv)
                self.assertEqual(spv.count("OpAtomicExchange"), 1, (dim, glc))
                self.assertNotIn("OpAtomicCompareExchange", spv)

    def test_swap_runs_only_in_active_lanes(self):
        # No neutral operand exists for an exchange: the op must sit inside the EXEC check's region —
        # an OpBranchConditional guard before it, not the OpSelect neutral-operand form the other atomics use.
        spv = translate(body("image_atomic_swap"))
        xchg = spv.find("OpAtomicExchange")
        self.assertIn("OpSelectionMerge", spv[:xchg])
        self.assertIn("OpBranchConditional", spv[:xchg])

    def test_swap_glc_writes_vdata_back(self):
        # glc: the pre-op texel value lands in the data VGPR (v5 here); without glc it is not written.
        spv = translate(body("image_atomic_swap"))
        xchg, store = spv.find("OpAtomicExchange"), spv.find("OpStore", spv.find("OpAtomicExchange"))
        self.assertGreater(store, xchg)


class ImageAtomicIncDecTests(unittest.TestCase):
    def test_inc_cas_loop(self):
        # inc lowers to a CAS loop: M = (M >= DATA) ? 0 : M + 1 (unsigned compare).
        spv = translate(body("image_atomic_inc"))
        self.assertIn("OpLoopMerge", spv)
        self.assertIn("OpUGreaterThanEqual", spv)
        self.assertIn("OpAtomicCompareExchange", spv)
        self.assertNotIn("OpAtomicExchange %", spv.replace("CompareExchange", ""))

    def test_dec_cas_loop(self):
        # dec: M = (M == 0 || M > DATA) ? DATA : M - 1.
        spv = translate(body("image_atomic_dec"))
        self.assertIn("OpLoopMerge", spv)
        self.assertIn("OpIEqual", spv)
        self.assertIn("OpUGreaterThan %bool", spv)
        self.assertIn("OpLogicalOr", spv)
        self.assertIn("OpAtomicCompareExchange", spv)

    def test_inc_dec_3d(self):
        for op in ("image_atomic_inc", "image_atomic_dec"):
            spv = translate(body(op, "3D", "v[1:3]", glc=False))
            self.assertIn("OpCompositeConstruct %v3int", spv)
            self.assertIn("OpAtomicCompareExchange", spv)

    def test_inc_dec_only_in_active_lanes(self):
        for op in ("image_atomic_inc", "image_atomic_dec"):
            spv = translate(body(op, glc=False))
            head = spv.split("OpAtomicCompareExchange", 1)[0]
            self.assertIn("OpBranchConditional", head)


class ImageAtomicCasA16Tests(unittest.TestCase):
    def test_swap_and_inc_a16(self):
        # a16 packed coordinates unpack like the plain atomics (low half first).
        for op in ("image_atomic_swap", "image_atomic_inc"):
            spv = translate(body(op, mods="unorm a16", glc=False))
            self.assertIn("OpShiftRightLogical %uint", spv)
            self.assertIn("OpImageTexelPointer", spv)


if __name__ == "__main__":
    unittest.main()

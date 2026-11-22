"""cs_w32_ds_wrxchg_b64: GPU test fixture for 64-bit and st64 LDS atomic exchanges, checked on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_ds_wrxchg_b64.asm exercises ds_wrxchg_rtn_b64 (with offset: and vdst aliasing DATA),
ds_wrxchg2_rtn_b64 (8-byte steps and same-address ordering with vdst aliasing both DATA pairs),
ds_wrxchg2st64_rtn_b64 (512-byte steps), and ds_wrxchg2st64_rtn_b32 (256-byte steps), folding the old and read-back
values into eight output dwords on lane 0. This runs the generated fixture (tests/semantic/regen_fixtures.py) through
spirv_mini_interp, checks a negative control that mutates a st64 offset, and verifies that check.py and
scripts/run_gpu_tests.py expect the exact same words on the GPU.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF

# 4 user SGPRs (the output V#), TGID_X in s4, 2048 bytes of LDS (LDS_SIZE = 4 -> 0x20088), 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x20088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def mix(lo: int, hi: int) -> int:
    """Fold two 32-bit words asymmetrically: (lo ^ (3 * hi)) mod 2^32."""
    return (lo ^ ((3 * hi) & M32)) & M32


def mix2(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> int:
    """Fold four 32-bit words via pairwise mix."""
    return mix(mix(a_lo, a_hi), mix(b_lo, b_hi))


def _run_shader(spv: str) -> dict[int, int]:
    """Execute one 32-lane workgroup of the translated compute fixture on the CPU interpreter."""
    return si.run_workgroup(
        spv,
        1,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


EXPECTED = [
    mix(0x81223344, 0x55667788),
    mix(0x99AABBCC, 0xDDEEFF01),
    mix2(0x10203040, 0x90A0B0C0, 0x13579BDF, 0x2468ACE0),
    mix2(0xCAFEBABE, 0xDEADBEEF, 0xFEEDFACE, 0x8BADF00D),
    mix2(0x01020304, 0x85060708, 0xA1A2A3A4, 0xB5B6B7B8),
    mix2(0x31415926, 0x53589793, 0x27182818, 0x28459045),
    mix2(0x12345678, 0x87654321, 0xAAAABBBB, 0xCCCCDDDD),
    mix(mix2(0x11112222, 0x33334444, 0x55556666, 0x77778888), mix(0x9999AAAA, 0xBBBBCCCC)),
]


class LdsWrxchgB64Fixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(
            EXPECTED,
            [0x811155DC, 0x006646CF, 0xDA8AFA7D, 0x4F5B06C8, 0x0F839BB8, 0xD411551A, 0xE27FF99F, 0x1113777E],
        )

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_ds_wrxchg_b64", fssm=FSSM)
        mem = _run_shader(spv)
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_negative_control_offset_mismatch(self):
        path = os.path.join(support.FIXTURE_DIR, "cs_w32_ds_wrxchg_b64.s")
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        target = "\tds_wrxchg2st64_rtn_b64 v[16:19], v1, v[12:13], v[14:15] offset0:1 offset1:2\n"
        self.assertEqual(text.count(target), 1)
        mutated = text.replace(target, target.replace("offset1:2", "offset1:1"), 1)
        spv = support.translate_listing("cs_w32_ds_wrxchg_b64", mutated.encode("utf-8"), fssm=FSSM)
        mem = _run_shader(spv)
        self.assertNotEqual(mem[16], EXPECTED[4])
        self.assertNotEqual(mem[20], EXPECTED[5])

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_ds_wrxchg_b64"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()

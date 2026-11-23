"""ds_read_i8_d16 followed by ds_write_b16 of the same register: the half written is the sign-extended byte.

Standard library only; no GPU, no LLVM, no game files.

    python -m unittest discover -s tests/semantic -p "test_lds_i8_d16_write.py" -v

The input is the GPU test runtime/test/shaders/cs_w64_lds_i8_d16_write.asm (fixture from regen_fixtures.py). It was
found by tools/lds_fuzz.py --seed 9950 (program 95 at 64 lanes) on AMD's Windows driver (gfx1036, 32.0.21042.62),
which wrote the half zero-extended (0x0090 for the byte 0x90) while the register itself was right, and while the
same SPIR-V module gave the right half on an NVIDIA RTX 4080. The driver loses the extension when the read is
lowered as OpBitFieldSExtract at a run-time offset, so the signed sub-dword LDS reads sign-extend as
(field ^ sign) - sign instead. Two things are checked here:

  * the words: the translated shader, run as one invocation (without its s_barrier lines, which only order the
    lanes against each other), stores the eight words of the .asm header (the same
    words check.py / run_gpu_tests.py expect), derived below from the ISA rule
    ds_read_i8_d16 D[15:0] = sign-extend 8 -> 16 of MEM[ADDR][7:0], D[31:16] unchanged;
  * the lowering: no signed sub-dword LDS read is emitted with OpBitFieldSExtract any more. This is the part a CPU
    run can't see (both lowerings give the same words here); it fails if the old form comes back.
"""
import os
import sys
import unittest

import support
import spirv_mini_interp as si

LDS_BYTES = (0xA5, 0x90, 0xCC, 0x80)         # bytes at byte address 0..3 of LDS dword 0 = 0x80CC90A5
OLD = 0xBEEF1234                             # every read's destination before the read


def sext16(b):
    return b | 0xFF00 if b & 0x80 else b


# In the order the shader stores them: dwords 4..7 (0 before the 16-bit write), then the four registers.
EXPECTED = [sext16(b) for b in LDS_BYTES] + [(OLD & 0xFFFF0000) | sext16(b) for b in LDS_BYTES]

SIGNED_READS = ("ds_read_i8", "ds_read_i16", "ds_read_i8_d16", "ds_read_i8_d16_hi")


def fixture_header():
    sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
    from compute_fixture import compute_header
    asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_w64_lds_i8_d16_write.asm")
    return bytes(compute_header(asm))


def translated(barriers=True):
    """The fixture's SPIR-V. spirv_mini_interp has no OpControlBarrier and runs one invocation here, where the
    barriers (they order the lanes' writes against each other) change nothing, so the run leaves them out."""
    if barriers:
        return support.translate_fixture("cs_w64_lds_i8_d16_write", fssm=fixture_header())
    with open(os.path.join(support.FIXTURE_DIR, "cs_w64_lds_i8_d16_write.s"), encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines(keepends=True) if ln.strip() != "s_barrier"]
    return support.translate_listing("cs_w64_lds_i8_d16_write", "".join(lines).encode("utf-8"), fssm=fixture_header())


def run():
    spv = translated(barriers=False)
    memory = si.run_workgroup(
        spv, 1, 64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})
    stray = sorted(a for a in memory if not 0 <= a < 32 or a % 4)
    if stray:
        raise AssertionError(f"stores outside out[0..7]: {stray}")
    return [memory.get(4 * k) for k in range(8)]


class LdsI8D16Write(unittest.TestCase):
    def test_reference_values(self):
        self.assertEqual(EXPECTED, [0x0000FFA5, 0x0000FF90, 0x0000FFCC, 0x0000FF80,
                                    0xBEEFFFA5, 0xBEEFFF90, 0xBEEFFFCC, 0xBEEFFF80])

    def test_translated_shader(self):
        got = run()
        for k, (g, w) in enumerate(zip(got, EXPECTED)):
            with self.subTest(word=k):
                self.assertEqual(g, w, f"out[{k}] = {g if g is None else hex(g)}, want {hex(w)}")

    def test_signed_reads_are_not_lowered_with_a_signed_field_extract(self):
        self.assertNotIn("OpBitFieldSExtract", translated())
        for op in SIGNED_READS:
            with self.subTest(op=op):
                spv = support.translate_lines("cs_lds_signed_read", [
                    "v_mov_b32_e32 v1, 0", f"{op} v2, v1 offset:1", "s_waitcnt lgkmcnt(0)",
                    "v_mov_b32_e32 v3, 0", "buffer_store_dword v2, v3, s[0:3], 0 offen", "s_endpgm"],
                    fssm=fixture_header())
                self.assertNotIn("OpBitFieldSExtract", spv)
                self.assertIn("OpBitFieldUExtract", spv)


if __name__ == "__main__":
    unittest.main()

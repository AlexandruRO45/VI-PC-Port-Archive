"""Four docs/design.md rules that were only CPU-checked on RDNA2, pinned to the words an RDNA2 GPU returned.

Every row was run natively on an AMD Radeon RDNA2 iGPU (gfx1036, Ryzen 7 9800X3D, Windows AMD driver 32.0.21042.62)
through tools/hw_oracle with AMD clang 23 (rocm-sdk-core 10.0.0), one wave32 thread, in the MODE inferred for the game
(IEEE 0, DX10_CLAMP 1, f16 denormals kept). Each row sets its registers (VGPRs with v_mov_b32, SGPRs with s_mov_b32),
seeds the output VGPR with 0xA5A5A5A5 when the row does not set it, runs its lines, and gives the word the output VGPR
held afterwards. No game files: every word is synthetic.

- v_cndmask_b32 source modifiers (row "v_cndmask_b32 source modifiers"): abs/neg on either source are sign-bit moves,
  so signalling NaNs stay signalling and denormals are not flushed. All 180 rows gave the same words with IEEE 1.
- null in 64-bit integer compares (row "Null sources in 64-bit integer VALU compares"): null reads 0 in either
  position; the compare's SGPR goes through v_cndmask_b32 to 0/1.
- v_fmamk_f16 / v_fmaak_f16 (row "v_fmamk_f16, v_fmaak_f16"), until now measured on gfx1100 only: the operand
  order, one rounding, the kept high half, denormals and infinities. 8 further measured rows are left out: a quiet
  NaN input 0x7e01 comes back as 0x7e01 on the hardware and as the canonical 0x7e00 from the translator, the known NaN
  payload difference of every f16 arithmetic op (row "f16 transcendentals").
- v_dot2_f32_f16 and v_dot2c_f32_f16 (row "Packed integer dot products"), until now CPU-checked only: products and
  accumulation in f32 over f16 halves including denormals, infinities and a NaN, and f32 accumulators at the extremes
  (VOP3P S2, or the old vdst for the VOP2 form). AMD clang rejects the VOP2 v_dot*c_* integer forms for gfx1036, so
  they stay unmeasured.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402

SEED = 0xA5A5A5A5

# (lines, ((register, value), ...), output VGPR, the word gfx1036 left in it)
CNDMASK = [
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800001),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800005),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0xFFC00002),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0x40000000),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0xFF800000),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800001),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800005),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0xFFC00002),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0x40000000),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0xFF800000),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800005),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0x40000000),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0x7F800005),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0x40000000),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800001),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0xFFC00002),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0xFF800000),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800001),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000000),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000000),), 'v2', 0xFFC00002),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000000),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000000),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000000),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000000),), 'v2', 0xFF800000),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, v1, v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0x40000000),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0xFF800000),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0xFF800001),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0x7F800005),
    ('v_cndmask_b32_e64 v2, v1, -v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0xFFC00002),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0x40000000),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0x7F800005),
    ('v_cndmask_b32_e64 v2, v1, |v4|, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0xFF800000),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0xFF800001),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, v1, -|v4|, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0xFFC00002),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, -v1, v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0x40000000),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0xFF800000),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0xFF800001),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0x7F800005),
    ('v_cndmask_b32_e64 v2, -v1, -v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0xFFC00002),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, |v1|, v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0x40000000),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0x7F800005),
    ('v_cndmask_b32_e64 v2, |v1|, |v4|, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x00000000),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x00000001),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0x7F800000),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0x7F800001),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, -|v1|, v4, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0x7FC00002),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x7F800001), ('v4', 0x80000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0xFF800005), ('v4', 0x00000000), ('vcc_lo', 0x00000001),), 'v2', 0x80000000),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x7FC00002), ('v4', 0xC0000000), ('vcc_lo', 0x00000001),), 'v2', 0xC0000000),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x80000000), ('v4', 0x00000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x00000000), ('v4', 0x80000001), ('vcc_lo', 0x00000001),), 'v2', 0x80000001),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0xC0000000), ('v4', 0x7F800000), ('vcc_lo', 0x00000001),), 'v2', 0xFF800000),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x00000001), ('v4', 0x7F800001), ('vcc_lo', 0x00000001),), 'v2', 0xFF800001),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x80000001), ('v4', 0xFF800005), ('vcc_lo', 0x00000001),), 'v2', 0xFF800005),
    ('v_cndmask_b32_e64 v2, -|v1|, -|v4|, vcc_lo', (('v1', 0x7F800000), ('v4', 0x7FC00002), ('vcc_lo', 0x00000001),), 'v2', 0xFFC00002),
]

CMP64_NULL = [
    ('v_cmp_eq_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_eq_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_eq_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_eq_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_eq_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000000),
    ('v_cmp_eq_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000000),
    ('v_cmp_eq_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000000),
    ('v_cmp_eq_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000000),
    ('v_cmp_ne_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_ne_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_ne_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_ne_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_ne_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000001),
    ('v_cmp_ne_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000001),
    ('v_cmp_ne_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000001),
    ('v_cmp_ne_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000001),
    ('v_cmp_lt_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_lt_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_lt_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_lt_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_lt_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000000),
    ('v_cmp_lt_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000001),
    ('v_cmp_lt_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000000),
    ('v_cmp_lt_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000001),
    ('v_cmp_gt_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_gt_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_gt_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_gt_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_gt_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000000),
    ('v_cmp_gt_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000001),
    ('v_cmp_gt_u64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000000),
    ('v_cmp_gt_u64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000001),
    ('v_cmp_le_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_le_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_le_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_le_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_le_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000000),
    ('v_cmp_le_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000001),
    ('v_cmp_le_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000000),
    ('v_cmp_le_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000001),
    ('v_cmp_ge_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_ge_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_ge_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000000),
    ('v_cmp_ge_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000001), ('v2', 0x00000000),), 'v3', 0x00000001),
    ('v_cmp_ge_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000001),
    ('v_cmp_ge_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0x00000000), ('v2', 0x80000000),), 'v3', 0x00000000),
    ('v_cmp_ge_i64_e64 s20, null, v[1:2] ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000001),
    ('v_cmp_ge_i64_e64 s20, v[1:2], null ; v_cndmask_b32_e64 v3, 0, 1, s20', (('v1', 0xFFFFFFFF), ('v2', 0xFFFFFFFF),), 'v3', 0x00000000),
]

FMAK16 = [
    ('v_fmamk_f16 v2, v1, 0x3e00, v4', (('v1', 0x12343C00), ('v4', 0x56780001),), 'v2', 0xA5A53E00),
    ('v_fmamk_f16 v2, v1, 0x3e00, v4', (('v1', 0x1234C000), ('v4', 0x56783555),), 'v2', 0xA5A5C155),
    ('v_fmamk_f16 v2, v1, 0x3e00, v4', (('v1', 0x12340001), ('v4', 0x56780000),), 'v2', 0xA5A50002),
    ('v_fmamk_f16 v2, v1, 0x3e00, v4', (('v1', 0x123483FF), ('v4', 0x5678C000),), 'v2', 0xA5A5C000),
    ('v_fmamk_f16 v2, v1, 0x3e00, v4', (('v1', 0x12347BFF), ('v4', 0x56787BFF),), 'v2', 0xA5A57C00),
    ('v_fmamk_f16 v2, v1, 0x3e00, v4', (('v1', 0x12347C00), ('v4', 0x56783C00),), 'v2', 0xA5A57C00),
    ('v_fmamk_f16 v2, v1, 0x3e00, v4', (('v1', 0x12340000), ('v4', 0x56787C00),), 'v2', 0xA5A57C00),
    ('v_fmamk_f16 v2, v1, 0x3e00, v4', (('v1', 0x12348000), ('v4', 0x56788000),), 'v2', 0xA5A58000),
    ('v_fmaak_f16 v2, v1, v4, 0x4000', (('v1', 0x12343C00), ('v4', 0x56780001),), 'v2', 0xA5A54000),
    ('v_fmaak_f16 v2, v1, v4, 0x4000', (('v1', 0x1234C000), ('v4', 0x56783555),), 'v2', 0xA5A53D56),
    ('v_fmaak_f16 v2, v1, v4, 0x4000', (('v1', 0x12340001), ('v4', 0x56780000),), 'v2', 0xA5A54000),
    ('v_fmaak_f16 v2, v1, v4, 0x4000', (('v1', 0x123483FF), ('v4', 0x5678C000),), 'v2', 0xA5A54000),
    ('v_fmaak_f16 v2, v1, v4, 0x4000', (('v1', 0x12347BFF), ('v4', 0x56787BFF),), 'v2', 0xA5A57C00),
    ('v_fmaak_f16 v2, v1, v4, 0x4000', (('v1', 0x12347C00), ('v4', 0x56783C00),), 'v2', 0xA5A57C00),
    ('v_fmaak_f16 v2, v1, v4, 0x4000', (('v1', 0x12340000), ('v4', 0x56787C00),), 'v2', 0xA5A5FE00),
    ('v_fmaak_f16 v2, v1, v4, 0x4000', (('v1', 0x12348000), ('v4', 0x56788000),), 'v2', 0xA5A54000),
    ('v_fmamk_f16 v2, v1, 0x0001, v4', (('v1', 0x12343C00), ('v4', 0x56780001),), 'v2', 0xA5A50002),
    ('v_fmamk_f16 v2, v1, 0x0001, v4', (('v1', 0x1234C000), ('v4', 0x56783555),), 'v2', 0xA5A53555),
    ('v_fmamk_f16 v2, v1, 0x0001, v4', (('v1', 0x12340001), ('v4', 0x56780000),), 'v2', 0xA5A50000),
    ('v_fmamk_f16 v2, v1, 0x0001, v4', (('v1', 0x123483FF), ('v4', 0x5678C000),), 'v2', 0xA5A5C000),
    ('v_fmamk_f16 v2, v1, 0x0001, v4', (('v1', 0x12347BFF), ('v4', 0x56787BFF),), 'v2', 0xA5A57BFF),
    ('v_fmamk_f16 v2, v1, 0x0001, v4', (('v1', 0x12347C00), ('v4', 0x56783C00),), 'v2', 0xA5A57C00),
    ('v_fmamk_f16 v2, v1, 0x0001, v4', (('v1', 0x12340000), ('v4', 0x56787C00),), 'v2', 0xA5A57C00),
    ('v_fmamk_f16 v2, v1, 0x0001, v4', (('v1', 0x12348000), ('v4', 0x56788000),), 'v2', 0xA5A58000),
    ('v_fmaak_f16 v2, v1, v4, 0xfc00', (('v1', 0x12343C00), ('v4', 0x56780001),), 'v2', 0xA5A5FC00),
    ('v_fmaak_f16 v2, v1, v4, 0xfc00', (('v1', 0x1234C000), ('v4', 0x56783555),), 'v2', 0xA5A5FC00),
    ('v_fmaak_f16 v2, v1, v4, 0xfc00', (('v1', 0x12340001), ('v4', 0x56780000),), 'v2', 0xA5A5FC00),
    ('v_fmaak_f16 v2, v1, v4, 0xfc00', (('v1', 0x123483FF), ('v4', 0x5678C000),), 'v2', 0xA5A5FC00),
    ('v_fmaak_f16 v2, v1, v4, 0xfc00', (('v1', 0x12347BFF), ('v4', 0x56787BFF),), 'v2', 0xA5A5FC00),
    ('v_fmaak_f16 v2, v1, v4, 0xfc00', (('v1', 0x12347C00), ('v4', 0x56783C00),), 'v2', 0xA5A5FE00),
    ('v_fmaak_f16 v2, v1, v4, 0xfc00', (('v1', 0x12340000), ('v4', 0x56787C00),), 'v2', 0xA5A5FE00),
    ('v_fmaak_f16 v2, v1, v4, 0xfc00', (('v1', 0x12348000), ('v4', 0x56788000),), 'v2', 0xA5A5FC00),
]

DOT2_F32_F16 = [
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xBC003C00), ('v4', 0x00017BFF), ('v5', 0x3F800000),), 'v2', 0x477FE100),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x8001BC00), ('v4', 0xC5007E00), ('v5', 0x00000000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7E007BFF), ('v4', 0xFC00BC00), ('v5', 0x80000000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xC5000001), ('v4', 0x7BFFFC00), ('v5', 0x7F7FFFFF),), 'v2', 0xFF800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7BFF8001), ('v4', 0x04003C00), ('v5', 0x00000001),), 'v2', 0x407FE000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7C007C00), ('v4', 0x7C007C00), ('v5', 0xBF800000),), 'v2', 0x7F800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x3555FC00), ('v4', 0xBC00C500), ('v5', 0x3F800000),), 'v2', 0x7F800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x3C007E00), ('v4', 0x35558001), ('v5', 0x00000000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x00013555), ('v4', 0x80010400), ('v5', 0x80000000),), 'v2', 0x37AAA000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xFC000400), ('v4', 0x3C000001), ('v5', 0x7F7FFFFF),), 'v2', 0xFF800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x0400C500), ('v4', 0x7E003555), ('v5', 0x00000001),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xBC003C00), ('v4', 0x00017BFF), ('v5', 0xBF800000),), 'v2', 0x477FDF00),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x8001BC00), ('v4', 0xC5007E00), ('v5', 0x3F800000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7E007BFF), ('v4', 0xFC00BC00), ('v5', 0x00000000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xC5000001), ('v4', 0x7BFFFC00), ('v5', 0x80000000),), 'v2', 0xFF800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7BFF8001), ('v4', 0x04003C00), ('v5', 0x7F7FFFFF),), 'v2', 0x7F7FFFFF),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7C007C00), ('v4', 0x7C007C00), ('v5', 0x00000001),), 'v2', 0x7F800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x3555FC00), ('v4', 0xBC00C500), ('v5', 0xBF800000),), 'v2', 0x7F800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x3C007E00), ('v4', 0x35558001), ('v5', 0x3F800000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x00013555), ('v4', 0x80010400), ('v5', 0x00000000),), 'v2', 0x37AAA000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xFC000400), ('v4', 0x3C000001), ('v5', 0x80000000),), 'v2', 0xFF800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x0400C500), ('v4', 0x7E003555), ('v5', 0x7F7FFFFF),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xBC003C00), ('v4', 0x00017BFF), ('v5', 0x00000001),), 'v2', 0x477FE000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x8001BC00), ('v4', 0xC5007E00), ('v5', 0xBF800000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7E007BFF), ('v4', 0xFC00BC00), ('v5', 0x3F800000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xC5000001), ('v4', 0x7BFFFC00), ('v5', 0x00000000),), 'v2', 0xFF800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7BFF8001), ('v4', 0x04003C00), ('v5', 0x80000000),), 'v2', 0x407FE000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7C007C00), ('v4', 0x7C007C00), ('v5', 0x7F7FFFFF),), 'v2', 0x7F800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x3555FC00), ('v4', 0xBC00C500), ('v5', 0x00000001),), 'v2', 0x7F800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x3C007E00), ('v4', 0x35558001), ('v5', 0xBF800000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x00013555), ('v4', 0x80010400), ('v5', 0x3F800000),), 'v2', 0x3F8000AB),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xFC000400), ('v4', 0x3C000001), ('v5', 0x00000000),), 'v2', 0xFF800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x0400C500), ('v4', 0x7E003555), ('v5', 0x80000000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xBC003C00), ('v4', 0x00017BFF), ('v5', 0x7F7FFFFF),), 'v2', 0x7F7FFFFF),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x8001BC00), ('v4', 0xC5007E00), ('v5', 0x00000001),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7E007BFF), ('v4', 0xFC00BC00), ('v5', 0xBF800000),), 'v2', 0x7FC00000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0xC5000001), ('v4', 0x7BFFFC00), ('v5', 0x3F800000),), 'v2', 0xFF800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7BFF8001), ('v4', 0x04003C00), ('v5', 0x00000000),), 'v2', 0x407FE000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x7C007C00), ('v4', 0x7C007C00), ('v5', 0x80000000),), 'v2', 0x7F800000),
    ('v_dot2_f32_f16 v2, v1, v4, v5', (('v1', 0x3555FC00), ('v4', 0xBC00C500), ('v5', 0x7F7FFFFF),), 'v2', 0x7F800000),
]

DOT2C_F32_F16 = [
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xBC003C00), ('v4', 0x00017BFF), ('v2', 0x3F800000)), 'v2', 0x477FE100),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x8001BC00), ('v4', 0xC5007E00), ('v2', 0x00000000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7E007BFF), ('v4', 0xFC00BC00), ('v2', 0x80000000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xC5000001), ('v4', 0x7BFFFC00), ('v2', 0x7F7FFFFF)), 'v2', 0xFF800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7BFF8001), ('v4', 0x04003C00), ('v2', 0x00000001)), 'v2', 0x407FE000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7C007C00), ('v4', 0x7C007C00), ('v2', 0xBF800000)), 'v2', 0x7F800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x3555FC00), ('v4', 0xBC00C500), ('v2', 0x3F800000)), 'v2', 0x7F800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x3C007E00), ('v4', 0x35558001), ('v2', 0x00000000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x00013555), ('v4', 0x80010400), ('v2', 0x80000000)), 'v2', 0x37AAA000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xFC000400), ('v4', 0x3C000001), ('v2', 0x7F7FFFFF)), 'v2', 0xFF800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x0400C500), ('v4', 0x7E003555), ('v2', 0x00000001)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xBC003C00), ('v4', 0x00017BFF), ('v2', 0xBF800000)), 'v2', 0x477FDF00),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x8001BC00), ('v4', 0xC5007E00), ('v2', 0x3F800000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7E007BFF), ('v4', 0xFC00BC00), ('v2', 0x00000000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xC5000001), ('v4', 0x7BFFFC00), ('v2', 0x80000000)), 'v2', 0xFF800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7BFF8001), ('v4', 0x04003C00), ('v2', 0x7F7FFFFF)), 'v2', 0x7F7FFFFF),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7C007C00), ('v4', 0x7C007C00), ('v2', 0x00000001)), 'v2', 0x7F800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x3555FC00), ('v4', 0xBC00C500), ('v2', 0xBF800000)), 'v2', 0x7F800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x3C007E00), ('v4', 0x35558001), ('v2', 0x3F800000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x00013555), ('v4', 0x80010400), ('v2', 0x00000000)), 'v2', 0x37AAA000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xFC000400), ('v4', 0x3C000001), ('v2', 0x80000000)), 'v2', 0xFF800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x0400C500), ('v4', 0x7E003555), ('v2', 0x7F7FFFFF)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xBC003C00), ('v4', 0x00017BFF), ('v2', 0x00000001)), 'v2', 0x477FE000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x8001BC00), ('v4', 0xC5007E00), ('v2', 0xBF800000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7E007BFF), ('v4', 0xFC00BC00), ('v2', 0x3F800000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xC5000001), ('v4', 0x7BFFFC00), ('v2', 0x00000000)), 'v2', 0xFF800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7BFF8001), ('v4', 0x04003C00), ('v2', 0x80000000)), 'v2', 0x407FE000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7C007C00), ('v4', 0x7C007C00), ('v2', 0x7F7FFFFF)), 'v2', 0x7F800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x3555FC00), ('v4', 0xBC00C500), ('v2', 0x00000001)), 'v2', 0x7F800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x3C007E00), ('v4', 0x35558001), ('v2', 0xBF800000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x00013555), ('v4', 0x80010400), ('v2', 0x3F800000)), 'v2', 0x3F8000AB),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xFC000400), ('v4', 0x3C000001), ('v2', 0x00000000)), 'v2', 0xFF800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x0400C500), ('v4', 0x7E003555), ('v2', 0x80000000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xBC003C00), ('v4', 0x00017BFF), ('v2', 0x7F7FFFFF)), 'v2', 0x7F7FFFFF),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x8001BC00), ('v4', 0xC5007E00), ('v2', 0x00000001)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7E007BFF), ('v4', 0xFC00BC00), ('v2', 0xBF800000)), 'v2', 0x7FC00000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0xC5000001), ('v4', 0x7BFFFC00), ('v2', 0x3F800000)), 'v2', 0xFF800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7BFF8001), ('v4', 0x04003C00), ('v2', 0x00000000)), 'v2', 0x407FE000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x7C007C00), ('v4', 0x7C007C00), ('v2', 0x80000000)), 'v2', 0x7F800000),
    ('v_dot2c_f32_f16_e32 v2, v1, v4', (('v1', 0x3555FC00), ('v4', 0xBC00C500), ('v2', 0x7F7FFFFF)), 'v2', 0x7F800000),
]

GROUPS = {"cndmask": CNDMASK, "cmp64null": CMP64_NULL, "fmamk_fmaak_f16": FMAK16, "dot2_f32_f16": DOT2_F32_F16,
          "dot2c_f32_f16": DOT2C_F32_F16}


def run(lines, regs, out):
    prog = []
    for reg, value in regs:
        prog.append(f"{'s' if not reg.startswith('v') else 'v'}_mov_b32 {reg}, {value:#010x}")
    if out not in dict(regs):
        prog.append(f"v_mov_b32 {out}, {SEED:#010x}")
    prog += lines.split(" ; ") + [f"exp mrt0 {out}, {out}, {out}, {out} done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_gfx1036_rows", prog))[0]


class Gfx1036MeasuredRows(unittest.TestCase):
    def check(self, rows):
        for lines, regs, out, word in rows:
            with self.subTest(lines=lines, regs=" ".join(f"{r}={v:#x}" for r, v in regs)):
                got = run(lines, regs, out)
                self.assertEqual(got, word, f"translator {got:#010x}, gfx1036 {word:#010x}")

    def test_cndmask_source_modifiers(self):
        self.check(CNDMASK)

    def test_cmp64_null_sources(self):
        self.check(CMP64_NULL)

    def test_fmamk_fmaak_f16(self):
        self.check(FMAK16)

    def test_dot2_f32_f16(self):
        self.check(DOT2_F32_F16)

    def test_dot2c_f32_f16(self):
        self.check(DOT2C_F32_F16)

    def test_rows_are_distinct(self):
        for name, rows in GROUPS.items():
            keys = [(lines, regs) for lines, regs, _, _ in rows]
            self.assertEqual(len(keys), len(set(keys)), name)


if __name__ == "__main__":
    unittest.main()

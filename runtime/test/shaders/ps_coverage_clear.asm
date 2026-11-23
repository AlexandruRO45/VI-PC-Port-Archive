; Writes the vk_harness clear sentinel (-1,-1,-1,-1) on every covered pixel. Run with ms_coverage to prove
; VKH_COVERAGE_MASK's stencil still marks those pixels covered (TEST-362, editrice's colour-collision case).
; Expected: the 52x52 mask is unchanged and the whole RGBA32F readback is the clear value.
v_mov_b32_e32 v0, -1.0
exp mrt0 v0, v0, v0, v0 done
s_endpgm

; Negative control: declare both targets, then discard every fragment with an empty valid mask.
; Both attachments must retain their distinct clear colors.
v_mov_b32 v2, 1.0
v_mov_b32 v3, 2.0
v_mov_b32 v4, 0x40400000
v_mov_b32 v5, 4.0
exp mrt0 v2, v3, v4, v5
s_waitcnt expcnt(0)
exp mrt1 v5, v4, v3, v2
s_waitcnt expcnt(0)
v_mov_b32 v1, 1
v_cmpx_eq_u32 0, v1
exp null off, off, off, off done vm
s_endpgm

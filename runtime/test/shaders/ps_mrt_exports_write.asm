; Positive control: both targets receive distinct RGBA values at every pixel.
v_mov_b32 v2, 1.0
v_mov_b32 v3, 2.0
v_mov_b32 v4, 0x40400000
v_mov_b32 v5, 4.0
exp mrt0 v2, v3, v4, v5 vm
s_waitcnt expcnt(0)
exp mrt1 v5, v4, v3, v2 done
s_waitcnt expcnt(0)
s_endpgm

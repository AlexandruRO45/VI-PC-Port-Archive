; Two eight-word T# keys in the synthetic heap. Convert both fetched XY word pairs to float as signed integers.
; SINT bytes sign extend and float16 components widen to float32 bits. Each coordinate addresses a different texel in
; either 8x8 image. With SINT then float, expect (-126, 34, 1061158912, -1073741824).
; With keys exchanged, expect (1048576000, -1082130432, 119, -8).
s_load_dwordx8 s[8:15], s[0:1], 0
s_load_dwordx8 s[16:23], s[0:1], 32
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v0, 1
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v2, 3
v_mov_b32_e32 v3, 4
image_load v[4:5], v[0:1], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D
image_load v[6:7], v[2:3], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_cvt_f32_i32_e32 v4, v4
v_cvt_f32_i32_e32 v5, v5
v_cvt_f32_i32_e32 v6, v6
v_cvt_f32_i32_e32 v7, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm

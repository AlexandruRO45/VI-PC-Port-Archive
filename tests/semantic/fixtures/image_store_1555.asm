; image_store to a T# the shader reads from memory, so a test picks its FORMAT (word 1 [28:20]).
; s[0:1] points at the T# (32 bytes) followed by the four components' f32 bits (16 bytes).
; A 1_5_5_5_UNORM T# (134) stores the float texel through the float view (binding 8) and the same texel packed
; as A1R5G5B5 (component 0 in bits 14:10, 1 in 9:5, 2 in 4:0, 3 in bit 15) through the uint view (binding 9),
; both at texel (5, 3); any other float format stores through the float view only.
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v0, s16
v_mov_b32_e32 v1, s17
v_mov_b32_e32 v2, s18
v_mov_b32_e32 v3, s19
v_mov_b32_e32 v4, 5
v_mov_b32_e32 v5, 3
image_store v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm

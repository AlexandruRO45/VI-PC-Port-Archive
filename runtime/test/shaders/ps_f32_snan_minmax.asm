; Synthetic GPU regression for signaling NaNs as operand 0 of v_min_f32 / v_max_f32 / v_med3_f32.
; The first three cases use inline literals; the final med3 case reads the same values from the Vulkan heap.
; Run with vk_graphics_test using VKH_SGPRS=0x1000,0,16,0 and VKH_HEAP below.
; AMD Windows 26.11.2 returned NaN for the literal cases before the translator workaround.
; Expected MRT0 words: 0x3f800000, 0x40000000, 0x3f800000, 0x3f800000.
s_buffer_load_dwordx4 s[4:7], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_min_f32 v4, 0x7f800001, 1.0
v_max_f32 v5, 0x7fa00000, 2.0
v_med3_f32 v6, 0x7f800001, 1.0, 2.0
v_med3_f32 v7, v2, v1, v3
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm

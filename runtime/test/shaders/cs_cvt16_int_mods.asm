; EGEN-CVT16-INT-MODS: the VOP3 clamp and omod on v_cvt_i16_f16 / v_cvt_u16_f16 leave the integer result unchanged.
; Measured natively on RDNA2 (gfx1036, tools/hw_oracle raw kernels, wave32 and wave64, the game's MODE with IEEE
; off and the default IEEE on: 1,456 programs, all equal to the translation without the modifiers). The conversion
; truncates toward zero and saturates to the 16-bit range; only D[15:0] is written. Lane 0 runs 8 cases, D starts
; as 0xdeadbeef, case i stores D at out[i]; out[8..63] stay 0xeeeeeeee.
;   0 v_cvt_i16_f16 clamp of 1.5 (0xabcd3e00, high half ignored): 1                    -> 0xdead0001
;   1 v_cvt_i16_f16 clamp of -2.5 (0xc100): -2 (a float clamp to [0, 1] would give 0)  -> 0xdeadfffe
;   2 v_cvt_u16_f16 clamp of 100.0 (0x5640): 100 = 0x64 (a float clamp would give 1)    -> 0xdead0064
;   3 v_cvt_i16_f16 mul:4 of 2.5 (0x4100): 2 (an omod would give 8 or 10)               -> 0xdead0002
;   4 v_cvt_u16_f16 div:2 of 100.0: 100 (an omod would give 50)                         -> 0xdead0064
;   5 v_cvt_i16_f16 clamp mul:2 of -inf (0xfc00): saturates to -32768 = 0x8000          -> 0xdead8000
;   6 v_cvt_u16_f16 -|v2| clamp of 1.0: -1.0 saturates to 0                             -> 0xdead0000
;   7 v_cvt_u16_f16 clamp of 65504.0 (0x7bff): 65504 = 0xffe0                           -> 0xdeadffe0
; Run: vk_compute_test cs_cvt16_int_mods.spv 1 dump -> out[0..7] (check.py CVT16_INT_MODS_WORDS)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0xabcd3e00
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_i16_f16_e64 v8, v2 clamp
buffer_store_dword v8, off, s[0:3], 0 offset:0
v_mov_b32_e32 v2, 0x0000c100
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_i16_f16_e64 v8, v2 clamp
buffer_store_dword v8, off, s[0:3], 0 offset:4
v_mov_b32_e32 v2, 0x00005640
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_u16_f16_e64 v8, v2 clamp
buffer_store_dword v8, off, s[0:3], 0 offset:8
v_mov_b32_e32 v2, 0x00004100
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_i16_f16_e64 v8, v2 mul:4
buffer_store_dword v8, off, s[0:3], 0 offset:12
v_mov_b32_e32 v2, 0x00005640
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_u16_f16_e64 v8, v2 div:2
buffer_store_dword v8, off, s[0:3], 0 offset:16
v_mov_b32_e32 v2, 0x0000fc00
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_i16_f16_e64 v8, v2 clamp mul:2
buffer_store_dword v8, off, s[0:3], 0 offset:20
v_mov_b32_e32 v2, 0x00003c00
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_u16_f16_e64 v8, -|v2| clamp
buffer_store_dword v8, off, s[0:3], 0 offset:24
v_mov_b32_e32 v2, 0x00007bff
v_mov_b32_e32 v8, 0xdeadbeef
v_cvt_u16_f16_e64 v8, v2 clamp
buffer_store_dword v8, off, s[0:3], 0 offset:28
s_waitcnt vmcnt(0)
s_endpgm

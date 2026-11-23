; Chained scalar constant folding (docs/design.md, row "Chained scalar constant folding (PERF)"): a multi-step SGPR
; constant calculation folds at translation time, and a later instruction uses the folded value. The row was
; confirmed by tests/semantic only (#445 #448 #449 #544 #577 added GPU fixtures for single constant ops; this one
; covers the chains and the memory addresses they build).
; Lane 0 of a wave32 group seeds heap bytes 0x40..0x7f with 0x5eed0000 + byte offset, builds a V# in s[4:7] over
; those bytes (base 0x40, stride 0, NUM_RECORDS 0x40, word3 0x31016fac) and loads through offsets a SALU chain folds:
;   out[0]: min(((0x16 - 2) << 1), 0x30)         = 0x28; heap byte 0x68 -> 0x5eed0028
;   out[1]: s_bfe_u32(5 + 0x180, 0x00080004)     = (0x185 >> 4) & 0xff = 0x18; heap byte 0x58 -> 0x5eed0018
;   out[2]: 0x10 << 2                            = 0x40, past NUM_RECORDS -> 0
; and five values the same folding leaves in SGPRs, written out unchanged:
;   out[3]: s_pack_ll_b32_b16(0x1234, 0xabcd)    = 0xabcd1234
;   out[4]: s_bfm_b32(12, 20)                    = 0xfff00000
;   out[5]: s_andn2_b32(0xff, 0xbf)              = 0x40
;   out[6]: ((0x1000 >> 4) * 3) + 7              = 0x307
;   out[7]: s_sext_i32_i8(0x80)                  = 0xffffff80
; group: 32,1,1
; Run: vk_compute_test cs_salu_const_fold.spv 1 dump -> out[0..7] = 0x5eed0028 0x5eed0018 0 0xabcd1234 0xfff00000 0x40 0x307 0xffffff80
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x5eed0000
v_mov_b32 v2, 0x5eed0004
v_mov_b32 v3, 0x5eed0008
v_mov_b32 v4, 0x5eed000c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:64
v_mov_b32 v1, 0x5eed0010
v_mov_b32 v2, 0x5eed0014
v_mov_b32 v3, 0x5eed0018
v_mov_b32 v4, 0x5eed001c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:80
v_mov_b32 v1, 0x5eed0020
v_mov_b32 v2, 0x5eed0024
v_mov_b32 v3, 0x5eed0028
v_mov_b32 v4, 0x5eed002c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:96
v_mov_b32 v1, 0x5eed0030
v_mov_b32 v2, 0x5eed0034
v_mov_b32 v3, 0x5eed0038
v_mov_b32 v4, 0x5eed003c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:112
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0
s_mov_b32 s6, 0x40
s_mov_b32 s7, 0x31016fac
s_mov_b32 s8, 0x16
s_sub_u32 s9, s8, 2
s_lshl_b32 s10, s9, 1
s_mov_b32 s16, 0x30
s_min_u32 s11, s10, s16
s_mov_b32 s12, 5
s_add_u32 s12, s12, 0x180
s_bfe_u32 s13, s12, 0x00080004
s_mov_b32 s14, 0x10
s_lshl_b32 s15, s14, 2
s_mov_b32 s23, 0x1234
s_mov_b32 s24, 0xabcd
s_pack_ll_b32_b16 s25, s23, s24
s_mov_b32 s26, 12
s_bfm_b32 s27, s26, 20
s_mov_b32 s28, 0xff
s_mov_b32 s34, 0xbf
s_andn2_b32 s29, s28, s34
s_mov_b32 s30, 0x1000
s_lshr_b32 s31, s30, 4
s_mul_i32 s30, s31, 3
s_add_u32 s31, s30, 7
s_mov_b32 s32, 0x80
s_sext_i32_i8 s33, s32
s_buffer_load_dword s20, s[4:7], s11 offset:0
s_buffer_load_dword s21, s[4:7], s13 offset:0
s_buffer_load_dword s22, s[4:7], s15 offset:0
s_waitcnt lgkmcnt(0)
v_mov_b32 v1, s20
v_mov_b32 v2, s21
v_mov_b32 v3, s22
v_mov_b32 v4, s25
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:0
v_mov_b32 v1, s27
v_mov_b32 v2, s29
v_mov_b32 v3, s31
v_mov_b32 v4, s33
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:16
s_endpgm

; 64-bit integer VALU compares with null sources (v_cmp and v_cmpx on u64 and i64).
; Proves that null reads as 64-bit 0 in either source (src0 and src1) across equality and ordering.
; Lane 0 of a wave32 group computes eight result words:
;   out[0] = unsigned eq/ne (null src0 & src1)            = 0xA5  (bits: 1, 0, 1, 0, 0, 1, 0, 1)
;   out[1] = unsigned lt/le/gt/ge (null src0)             = 0x4D  (bits: 1, 0, 1, 1, 0, 0, 1, 0)
;   out[2] = unsigned lt/le/gt/ge (null src1)             = 0xD4  (bits: 0, 0, 1, 0, 1, 0, 1, 1)
;   out[3] = signed i64 lt/le/gt/ge (null src0)           = 0xDA  (bits: 0, 1, 0, 1, 1, 0, 1, 1)
;   out[4] = signed i64 lt/le/gt/ge (null src1)           = 0xAD  (bits: 1, 0, 1, 1, 0, 1, 0, 1)
;   out[5] = signed i64 eq/ne (null src0 & src1)          = 0xA5  (bits: 1, 0, 1, 0, 0, 1, 0, 1)
;   out[6] = v_cmpx preserves VCC                         = 1
;   out[7] = v_cmpx with null narrows EXEC                = 0x7777
; group: 32,1,1
; Run: vk_compute_test cs_vcmp64_null.spv 1 dump -> out[0..7] = 165 77 212 218 173 165 1 30583
; mattr: +wavefrontsize32,-wavefrontsize64

; v[2:3] = 0, v[4:5] = 1, v[6:7] = -1
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 1
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, -1
v_mov_b32_e32 v7, -1

; word 0: unsigned equality and inequality
v_mov_b32_e32 v10, 0
v_cmp_eq_u64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v10, v20, 0, v10
v_cmp_eq_u64_e32 vcc_lo, null, v[4:5]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v10, v20, 1, v10
v_cmp_eq_u64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v10, v20, 2, v10
v_cmp_eq_u64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v10, v20, 3, v10
v_cmp_ne_u64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v10, v20, 4, v10
v_cmp_ne_u64_e32 vcc_lo, null, v[4:5]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v10, v20, 5, v10
v_cmp_ne_u64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v10, v20, 6, v10
v_cmp_ne_u64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v10, v20, 7, v10

; word 1: unsigned ordering, null as src0
v_mov_b32_e32 v11, 0
v_cmp_lt_u64_e64 s4, null, v[4:5]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v11, v20, 0, v11
v_cmp_lt_u64_e64 s4, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v11, v20, 1, v11
v_cmp_le_u64_e64 s4, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v11, v20, 2, v11
v_cmp_le_u64_e64 s4, null, v[4:5]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v11, v20, 3, v11
v_cmp_gt_u64_e64 s4, null, v[4:5]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v11, v20, 4, v11
v_cmp_gt_u64_e64 s4, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v11, v20, 5, v11
v_cmp_ge_u64_e64 s4, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v11, v20, 6, v11
v_cmp_ge_u64_e64 s4, null, v[4:5]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v11, v20, 7, v11

; word 2: unsigned ordering, null as src1
v_mov_b32_e32 v12, 0
v_cmp_lt_u64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v12, v20, 0, v12
v_cmp_lt_u64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v12, v20, 1, v12
v_cmp_le_u64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v12, v20, 2, v12
v_cmp_le_u64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v12, v20, 3, v12
v_cmp_gt_u64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v12, v20, 4, v12
v_cmp_gt_u64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v12, v20, 5, v12
v_cmp_ge_u64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v12, v20, 6, v12
v_cmp_ge_u64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v12, v20, 7, v12

; word 3: signed i64 ordering, null as src0
v_mov_b32_e32 v13, 0
v_cmp_lt_i64_e64 s4, null, v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v13, v20, 0, v13
v_cmp_lt_i64_e64 s4, null, v[4:5]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v13, v20, 1, v13
v_cmp_le_i64_e64 s4, null, v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v13, v20, 2, v13
v_cmp_le_i64_e64 s4, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v13, v20, 3, v13
v_cmp_gt_i64_e64 s4, null, v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v13, v20, 4, v13
v_cmp_gt_i64_e64 s4, null, v[4:5]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v13, v20, 5, v13
v_cmp_ge_i64_e64 s4, null, v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v13, v20, 6, v13
v_cmp_ge_i64_e64 s4, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v13, v20, 7, v13

; word 4: signed i64 ordering, null as src1
v_mov_b32_e32 v14, 0
v_cmp_lt_i64_e64 s4, v[6:7], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v14, v20, 0, v14
v_cmp_lt_i64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v14, v20, 1, v14
v_cmp_le_i64_e64 s4, v[6:7], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v14, v20, 2, v14
v_cmp_le_i64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v14, v20, 3, v14
v_cmp_gt_i64_e64 s4, v[6:7], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v14, v20, 4, v14
v_cmp_gt_i64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v14, v20, 5, v14
v_cmp_ge_i64_e64 s4, v[6:7], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v14, v20, 6, v14
v_cmp_ge_i64_e64 s4, v[4:5], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v14, v20, 7, v14

; word 5: signed i64 equality and inequality
v_mov_b32_e32 v15, 0
v_cmp_eq_i64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v15, v20, 0, v15
v_cmp_eq_i64_e32 vcc_lo, null, v[6:7]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v15, v20, 1, v15
v_cmp_eq_i64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v15, v20, 2, v15
v_cmp_eq_i64_e64 s4, v[6:7], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v15, v20, 3, v15
v_cmp_ne_i64_e32 vcc_lo, null, v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v15, v20, 4, v15
v_cmp_ne_i64_e32 vcc_lo, null, v[6:7]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v15, v20, 5, v15
v_cmp_ne_i64_e64 s4, v[2:3], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v15, v20, 6, v15
v_cmp_ne_i64_e64 s4, v[6:7], null
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v15, v20, 7, v15

; word 6: v_cmpx preserves VCC
s_mov_b32 s4, 0x12345678
s_mov_b32 vcc_lo, s4
v_cmpx_eq_u64_e32 null, v[2:3]
s_cmp_eq_u32 vcc_lo, s4
s_cselect_b32 s6, 1, 0
v_mov_b32_e32 v16, s6

; word 7: v_cmpx with null narrows EXEC
s_mov_b32 s5, exec_lo
v_mov_b32_e32 v25, 0x7777
v_cmpx_ne_u64_e32 null, v[2:3]
v_mov_b32_e32 v25, 0x9999
s_mov_b32 exec_lo, s5
v_mov_b32_e32 v17, v25

; Store out[0..7] from lane 0
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v14, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v16, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v17, v21, s[0:3], 0 offen
s_endpgm

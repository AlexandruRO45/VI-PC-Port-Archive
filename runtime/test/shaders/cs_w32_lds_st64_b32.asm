; GPU test for ds_write2st64_b32 and ds_read2st64_b32, which have no GPU shader on main (CPU-checked only). RDNA2 ISA
; pseudocode (DS two-address forms, st64):
;   ds_write2st64_b32  MEM[ADDR + OFFSET0 * 256] = DATA0[31:0]    MEM[ADDR + OFFSET1 * 256] = DATA1[31:0]
;   ds_read2st64_b32   RET[31:0] = MEM[ADDR + OFFSET0 * 256]      RET[63:32] = MEM[ADDR + OFFSET1 * 256]
; The offsets differ from each other and from the unit-4 and x32 strides a wrong decode would use.
; LDS is 4 KiB (RSRC2 0x40088). All 32 lanes fill it with 0xEEEEEEEE (lane L writes bytes 4L + 128k, k = 0..31) and
; s_barrier; lane 0 alone (EXEC = 1) then runs the ops below. At the end all 32 lanes read the whole LDS back and count
; the words that are not 0xEEEEEEEE: it must be exactly the 6 words the shader stored, so no op wrote a neighbouring
; word, one of the decoy addresses a wrong decode would hit (0x1A0, 0x3A0 for x32; 0x2C, 0x3C for unit 4) or any other word. out[7] is
; the second word of the last write if that count is right and the count otherwise.
;   out[0..1]  ds_write2st64_b32 v3 = 0x20, offset0:3 offset1:7 writes 0x99990009 at 0x20 + 3 * 256 = 0x320 and 0xAAAA000A
;              at 0x20 + 7 * 256 = 0x720; plain ds_read_b32 of both: 0x99990009 0xAAAA000A
;   out[2..3]  0xBBBB000B is seeded at 0x4C0 and 0xCCCC000C at 0x9C0; ds_read2st64_b32 v3 = 0xC0, offset0:4 offset1:9
;              reads 0xC0 + 4 * 256 = 0x4C0 and 0xC0 + 9 * 256 = 0x9C0: 0xBBBB000B 0xCCCC000C
;   out[4..5]  the same read with offset0:9 offset1:4 (offset0 larger) returns the words in the other order:
;              0xCCCC000C 0xBBBB000B
;   out[6..7]  ds_write2st64_b32 v3 = 0x30, offset0:11 offset1:2 writes DATA0 = 0xDDDD000D at 0x30 + 11 * 256 = 0xB30
;              and DATA1 = 0x1234ABCD at 0x30 + 2 * 256 = 0x230: plain reads of 0xB30 and 0x230 (out[7] when the
;              LDS count is 6): 0xDDDD000D 0x1234ABCD
; Run: vk_compute_test cs_w32_lds_st64_b32.spv 1 dump
;   -> out[0..7] = 0x99990009 0xAAAA000A 0xBBBB000B 0xCCCC000C 0xCCCC000C 0xBBBB000B 0xDDDD000D 0x1234ABCD
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_lshlrev_b32_e32 v1, 2, v0
v_mov_b32_e32 v2, 0xEEEEEEEE
ds_write_b32 v1, v2 offset:0
ds_write_b32 v1, v2 offset:128
ds_write_b32 v1, v2 offset:256
ds_write_b32 v1, v2 offset:384
ds_write_b32 v1, v2 offset:512
ds_write_b32 v1, v2 offset:640
ds_write_b32 v1, v2 offset:768
ds_write_b32 v1, v2 offset:896
ds_write_b32 v1, v2 offset:1024
ds_write_b32 v1, v2 offset:1152
ds_write_b32 v1, v2 offset:1280
ds_write_b32 v1, v2 offset:1408
ds_write_b32 v1, v2 offset:1536
ds_write_b32 v1, v2 offset:1664
ds_write_b32 v1, v2 offset:1792
ds_write_b32 v1, v2 offset:1920
ds_write_b32 v1, v2 offset:2048
ds_write_b32 v1, v2 offset:2176
ds_write_b32 v1, v2 offset:2304
ds_write_b32 v1, v2 offset:2432
ds_write_b32 v1, v2 offset:2560
ds_write_b32 v1, v2 offset:2688
ds_write_b32 v1, v2 offset:2816
ds_write_b32 v1, v2 offset:2944
ds_write_b32 v1, v2 offset:3072
ds_write_b32 v1, v2 offset:3200
ds_write_b32 v1, v2 offset:3328
ds_write_b32 v1, v2 offset:3456
ds_write_b32 v1, v2 offset:3584
ds_write_b32 v1, v2 offset:3712
ds_write_b32 v1, v2 offset:3840
ds_write_b32 v1, v2 offset:3968
s_waitcnt lgkmcnt(0)
s_barrier
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v22, 0xbbbb000b
ds_write_b32 v1, v22 offset:0x4c0
v_mov_b32_e32 v23, 0xcccc000c
ds_write_b32 v1, v23 offset:0x9c0
s_waitcnt lgkmcnt(0)
; ds_write2st64_b32
v_mov_b32_e32 v3, 0x20
v_mov_b32_e32 v18, 0x99990009
v_mov_b32_e32 v19, 0xAAAA000A
ds_write2st64_b32 v3, v18, v19 offset0:3 offset1:7
s_waitcnt lgkmcnt(0)
ds_read_b32 v10, v1 offset:0x320
ds_read_b32 v11, v1 offset:0x720
; ds_read2st64_b32, both offset orders
v_mov_b32_e32 v3, 0xC0
ds_read2st64_b32 v[12:13], v3 offset0:4 offset1:9
ds_read2st64_b32 v[14:15], v3 offset0:9 offset1:4
; ds_write2st64_b32, offset0 > offset1
v_mov_b32_e32 v3, 0x30
v_mov_b32_e32 v18, 0xDDDD000D
v_mov_b32_e32 v19, 0x1234ABCD
s_waitcnt lgkmcnt(0)
ds_write2st64_b32 v3, v18, v19 offset0:11 offset1:2
s_waitcnt lgkmcnt(0)
ds_read_b32 v16, v1 offset:0xb30
ds_read_b32 v17, v1 offset:0x230
s_waitcnt lgkmcnt(0)
; whole-LDS count of the words that are not the fill
s_mov_b32 exec_lo, -1
s_barrier
v_lshlrev_b32_e32 v1, 2, v0
ds_read_b32 v32, v1 offset:0
ds_read_b32 v33, v1 offset:128
ds_read_b32 v34, v1 offset:256
ds_read_b32 v35, v1 offset:384
ds_read_b32 v36, v1 offset:512
ds_read_b32 v37, v1 offset:640
ds_read_b32 v38, v1 offset:768
ds_read_b32 v39, v1 offset:896
ds_read_b32 v40, v1 offset:1024
ds_read_b32 v41, v1 offset:1152
ds_read_b32 v42, v1 offset:1280
ds_read_b32 v43, v1 offset:1408
ds_read_b32 v44, v1 offset:1536
ds_read_b32 v45, v1 offset:1664
ds_read_b32 v46, v1 offset:1792
ds_read_b32 v47, v1 offset:1920
ds_read_b32 v48, v1 offset:2048
ds_read_b32 v49, v1 offset:2176
ds_read_b32 v50, v1 offset:2304
ds_read_b32 v51, v1 offset:2432
ds_read_b32 v52, v1 offset:2560
ds_read_b32 v53, v1 offset:2688
ds_read_b32 v54, v1 offset:2816
ds_read_b32 v55, v1 offset:2944
ds_read_b32 v56, v1 offset:3072
ds_read_b32 v57, v1 offset:3200
ds_read_b32 v58, v1 offset:3328
ds_read_b32 v59, v1 offset:3456
ds_read_b32 v60, v1 offset:3584
ds_read_b32 v61, v1 offset:3712
ds_read_b32 v62, v1 offset:3840
ds_read_b32 v63, v1 offset:3968
s_waitcnt lgkmcnt(0)
s_mov_b32 s20, 0
v_cmp_ne_u32_e32 vcc_lo, v2, v32
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v33
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v34
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v35
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v36
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v37
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v38
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v39
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v40
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v41
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v42
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v43
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v44
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v45
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v46
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v47
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v48
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v49
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v50
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v51
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v52
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v53
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v54
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v55
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v56
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v57
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v58
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v59
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v60
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v61
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v62
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
v_cmp_ne_u32_e32 vcc_lo, v2, v63
s_bcnt1_i32_b32 s21, vcc_lo
s_add_i32 s20, s20, s21
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v26, s20
v_cmp_eq_u32_e32 vcc_lo, 6, v26
v_cndmask_b32_e32 v28, v26, v17, vcc_lo
v_mov_b32_e32 v30, 0
buffer_store_dword v10, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 4
buffer_store_dword v11, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 8
buffer_store_dword v12, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 12
buffer_store_dword v13, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 16
buffer_store_dword v14, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 20
buffer_store_dword v15, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 24
buffer_store_dword v16, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 28
buffer_store_dword v28, v30, s[0:3], 0 offen
s_endpgm

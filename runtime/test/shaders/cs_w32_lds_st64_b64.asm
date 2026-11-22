; GPU test for ds_write2st64_b64 and ds_read2st64_b64, which cs_lds_b64_cmpst.asm only checks through an XOR of their
; dwords and which docs/design.md otherwise lists as CPU-checked. RDNA2 ISA pseudocode (DS two-address forms, st64):
;   ds_write2st64_b64  MEM[ADDR + OFFSET0 * 512] = DATA0[63:0]    MEM[ADDR + OFFSET1 * 512] = DATA1[63:0]
;   ds_read2st64_b64   RET[63:0] = MEM[ADDR + OFFSET0 * 512]      RET[127:64] = MEM[ADDR + OFFSET1 * 512]
; A 64-bit element is two consecutive dwords, low first. The offsets differ from each other and from the unit-8 and x32
; strides a wrong decode would use.
; LDS is 4 KiB (RSRC2 0x40088). All 32 lanes fill it with 0xEEEEEEEE (lane L writes bytes 4L + 128k, k = 0..31) and
; s_barrier; lane 0 alone (EXEC = 1) then runs the ops below. At the end all 32 lanes read the whole LDS back and count
; the words that are not 0xEEEEEEEE: it must be exactly the 8 words the shader stored, so no op wrote a neighbouring
; word, one of the decoy addresses a wrong decode would hit (0x140, 0x340 for x32; 0x48, 0x58 for unit 8) or any other word. out[7] is
; the high dword of the last element if that count is right and the count otherwise.
;   out[0..3]  ds_write2st64_b64 v3 = 0x40, offset0:1 offset1:3 writes A = 0x22220002_11110001 at 0x40 + 512 = 0x240 and
;              B = 0x44440004_33330003 at 0x40 + 3 * 512 = 0x640; plain ds_read_b32 of 0x240, 0x244, 0x640, 0x644:
;              A.lo A.hi B.lo B.hi = 0x11110001 0x22220002 0x33330003 0x44440004
;   out[4..7]  C = 0x66660006_55550005 is seeded at 0x480 and D = 0x88880008_77770007 at 0xA80; ds_read2st64_b64
;              v3 = 0x80, offset0:5 offset1:2 (offset0 larger) reads D from 0x80 + 5 * 512 = 0xA80 first and C from
;              0x80 + 2 * 512 = 0x480 second: v[14:17] = D.lo D.hi C.lo C.hi
;              = 0x77770007 0x88880008 0x55550005 0x66660006 (out[7] is C.hi when the LDS count is 8)
; Run: vk_compute_test cs_w32_lds_st64_b64.spv 1 dump
;   -> out[0..7] = 0x11110001 0x22220002 0x33330003 0x44440004 0x77770007 0x88880008 0x55550005 0x66660006
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
v_mov_b32_e32 v8, 0x55550005
ds_write_b32 v1, v8 offset:0x480
v_mov_b32_e32 v9, 0x66660006
ds_write_b32 v1, v9 offset:0x484
v_mov_b32_e32 v18, 0x77770007
ds_write_b32 v1, v18 offset:0xa80
v_mov_b32_e32 v19, 0x88880008
ds_write_b32 v1, v19 offset:0xa84
s_waitcnt lgkmcnt(0)
; ds_write2st64_b64
v_mov_b32_e32 v3, 0x40
v_mov_b32_e32 v4, 0x11110001
v_mov_b32_e32 v5, 0x22220002
v_mov_b32_e32 v6, 0x33330003
v_mov_b32_e32 v7, 0x44440004
ds_write2st64_b64 v3, v[4:5], v[6:7] offset0:1 offset1:3
s_waitcnt lgkmcnt(0)
ds_read_b32 v10, v1 offset:0x240
ds_read_b32 v11, v1 offset:0x244
ds_read_b32 v12, v1 offset:0x640
ds_read_b32 v13, v1 offset:0x644
; ds_read2st64_b64, offset0 > offset1
v_mov_b32_e32 v3, 0x80
ds_read2st64_b64 v[14:17], v3 offset0:5 offset1:2
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
v_cmp_eq_u32_e32 vcc_lo, 8, v26
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

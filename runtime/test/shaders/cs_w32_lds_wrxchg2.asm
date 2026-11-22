; SHADER-311: ds_wrxchg2_rtn_b32 and ds_wrxchg2st64_rtn_b32, two 32-bit LDS exchanges per instruction.
; Both offsets default to 0, so the fixtures give offset1 explicitly. Only lane 0 runs the exchanges and the reads.
; Seed: mem bytes 0,4 = AAAAAAAA, BBBBBBBB; mem bytes 16,272 = CCCCCCCC, DDDDDDDD.
;   wrxchg2      vaddr byte 0,  offset0:0 offset1:1 (unit 4)   -> addrs 0 and 4
;                data 11111111,22222222 -> vdst = AAAAAAAA BBBBBBBB, mem[0]/[4] = 11111111/22222222
;   wrxchg2st64  vaddr byte 16, offset0:0 offset1:1 (unit 256) -> addrs 16 and 272
;                data 33333333,44444444 -> vdst = CCCCCCCC DDDDDDDD, mem[16]/[272] = 33333333/44444444
; out[0..7] = 0xAAAAAAAA 0xBBBBBBBB 0xCCCCCCCC 0xDDDDDDDD 0x11111111 0x22222222 0x33333333 0x44444444
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_wrxchg2.spv 1 dump
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v3, 0xaaaaaaaa
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v3, 0xbbbbbbbb
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 16
v_mov_b32_e32 v3, 0xcccccccc
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 272
v_mov_b32_e32 v3, 0xdddddddd
ds_write_b32 v1, v3
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0x11111111
v_mov_b32_e32 v3, 0x22222222
ds_wrxchg2_rtn_b32 v[10:11], v1, v2, v3 offset0:0 offset1:1
v_mov_b32_e32 v1, 16
v_mov_b32_e32 v4, 0x33333333
v_mov_b32_e32 v5, 0x44444444
ds_wrxchg2st64_rtn_b32 v[12:13], v1, v4, v5 offset0:0 offset1:1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0
ds_read_b32 v14, v1
v_mov_b32_e32 v1, 4
ds_read_b32 v15, v1
v_mov_b32_e32 v1, 16
ds_read_b32 v16, v1
v_mov_b32_e32 v1, 272
ds_read_b32 v17, v1
s_waitcnt lgkmcnt(0)
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

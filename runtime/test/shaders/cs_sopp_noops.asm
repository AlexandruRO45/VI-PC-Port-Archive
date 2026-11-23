; The GFX10 instructions translated as nothing (docs/design.md, "No-op SOPP/SMEM"): s_icache_inv, s_ttracedata,
; s_ttracedata_imm, s_code_end, s_version, ds_nop and v_clrexcp. Lane 0 of a wave32 group sets a scalar register,
; SCC, VCC and one LDS word, runs all seven between them and the reads, and stores what came back:
;   out[0] = s20 through v_mov_b32                     = 0x11111111  (a scalar register)
;   out[1] = s_cselect_b32 from the SCC set before it  = 0xAAAAAAAA  (SCC)
;   out[2] = v_cndmask_b32 against vcc_lo              = 0xDDDDDDDD  (VCC, lane 0's bit)
;   out[3] = the LDS word ds_read_b32 gets back        = 0x77        (memory; ds_nop wrote nothing)
; out[4..7] keep the 0xeeeeeeee fill: none of the seven wrote memory.
; The same state goes through tests/semantic/test_sopp_noops.py and its interpreter on the CPU (values, SCC and VCC
; survive); this is the same claim through a driver.
; group: 32,1,1
; Run: vk_compute_test cs_sopp_noops.spv 1 dump -> out[0..7] = 286331153 2863311530 3722304989 119
; 4008636142 4008636142 4008636142 4008636142
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v5, 0x20
v_mov_b32_e32 v6, 0x77
ds_write_b32 v5, v6
s_waitcnt lgkmcnt(0)
s_mov_b32 s20, 0x11111111
s_mov_b32 s21, 0xAAAAAAAA
s_mov_b32 s22, 0xBBBBBBBB
v_mov_b32_e32 v10, 1
v_mov_b32_e32 v11, 0xCCCCCCCC
v_mov_b32_e32 v12, 0xDDDDDDDD
v_cmp_eq_u32 vcc_lo, v10, 1
s_cmp_eq_u32 s20, 0x11111111
s_icache_inv
s_ttracedata
s_ttracedata_imm 0x1234
s_code_end
s_version 0x1
ds_nop
v_clrexcp
s_cselect_b32 s24, s21, s22
v_mov_b32_e32 v1, s20
v_cndmask_b32 v2, v11, v12, vcc_lo
ds_read_b32 v3, v5
s_waitcnt lgkmcnt(0)
buffer_store_dword v1, off, s[0:3], 0 offset:0
v_mov_b32_e32 v4, s24
buffer_store_dword v4, off, s[0:3], 0 offset:4
buffer_store_dword v2, off, s[0:3], 0 offset:8
buffer_store_dword v3, off, s[0:3], 0 offset:12
s_endpgm

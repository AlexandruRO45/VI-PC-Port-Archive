; global_load_dword_addtid / global_store_dword_addtid (GLOBAL 0x016 / 0x017) with positive & negative signed 12-bit
; offsets, saddr = SGPR pair and saddr = off, partial EXEC masking, and non-zero M0 (ignored).
; 1 wave32 group (32 lanes, 256-byte heap: out[0..7] at 0x00..0x1C, scratch[0..31] at 0x80..0xFC):
;   1. m0 = 0x40 (must NOT be added to global_*_addtid addresses).
;   2. All 32 lanes store (0xA000 + lane) to s[20:21](0x80) -> scratch[lane] = 0xA000 + lane.
;   3. Partial EXEC (even lanes only):
;      - store (0xB000 + lane) via s[22:23](0x70) offset:16 -> even scratch[lane] = 0xB000 + lane; odd untouched.
;      - pre-fill v3 = 0xC000 + lane, then load v3 via s[24:25](0x90) offset:-16 -> even v3 = 0xB000 + lane;
;        odd v3 stays 0xC000 + lane.
;   4. Full EXEC restored:
;      - load v4 via s[24:25](0x90) offset:-16 -> even 0xB000 + lane, odd 0xA000 + lane.
;      - load v5 via off offset:128 (0x80)     -> even 0xB000 + lane, odd 0xA000 + lane.
;   5. Selected lanes write 8 words to out[0..7]:
;      out[0] = lane  0 v3 = 0xB000 (even lane masked load with offset:-16)
;      out[1] = lane  1 v3 = 0xC001 (odd lane inactive, v3 preserved)
;      out[2] = lane  1 v4 = 0xA001 (odd lane memory untouched by masked store)
;      out[3] = lane  2 v4 = 0xB002 (even lane overwritten by masked offset:16 store, loaded with offset:-16)
;      out[4] = lane 30 v3 = 0xB01E (high even lane active load)
;      out[5] = lane 31 v3 = 0xC01F (high odd lane inactive preserve)
;      out[6] = lane 31 v4 = 0xA01F (high odd lane untouched memory)
;      out[7] = lane 29 v5 = 0xA01D (saddr=off offset:128 load)
; group: 32,1,1
; Run: vk_compute_test cs_w32_global_addtid.spv 1 dump -> out[0..7] = 0xB000 0xC001 0xA001 0xB002 0xB01E 0xC01F 0xA01F 0xA01D
; mcpu: gfx1030
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 m0, 0x40
s_mov_b32 s20, 0x80
s_mov_b32 s21, 0
s_mov_b32 s22, 0x70
s_mov_b32 s23, 0
s_mov_b32 s24, 0x90
s_mov_b32 s25, 0
v_add_nc_u32_e32 v1, 0xa000, v0
v_add_nc_u32_e32 v2, 0xb000, v0
v_add_nc_u32_e32 v3, 0xc000, v0
global_store_dword_addtid v1, s[20:21]
s_waitcnt_vscnt null, 0x0
s_mov_b32 s10, exec_lo
v_and_b32_e32 v6, 1, v0
v_cmpx_eq_u32_e32 0, v6
global_store_dword_addtid v2, s[22:23] offset:16
s_waitcnt_vscnt null, 0x0
global_load_dword_addtid v3, s[24:25] offset:-16
s_waitcnt vmcnt(0)
s_mov_b32 exec_lo, s10
global_load_dword_addtid v4, s[24:25] offset:-16
global_load_dword_addtid v5, off offset:128
s_waitcnt vmcnt(0)
; Write out[0..7] from specific lanes without races
s_mov_b32 s10, exec_lo
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0
buffer_store_dword v3, v10, s[0:3], 0 offen
s_mov_b32 exec_lo, s10
v_cmpx_eq_u32_e32 1, v0
v_mov_b32_e32 v10, 4
buffer_store_dword v3, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 8
buffer_store_dword v4, v10, s[0:3], 0 offen
s_mov_b32 exec_lo, s10
v_cmpx_eq_u32_e32 2, v0
v_mov_b32_e32 v10, 12
buffer_store_dword v4, v10, s[0:3], 0 offen
s_mov_b32 exec_lo, s10
v_cmpx_eq_u32_e32 30, v0
v_mov_b32_e32 v10, 16
buffer_store_dword v3, v10, s[0:3], 0 offen
s_mov_b32 exec_lo, s10
v_cmpx_eq_u32_e32 31, v0
v_mov_b32_e32 v10, 20
buffer_store_dword v3, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 24
buffer_store_dword v4, v10, s[0:3], 0 offen
s_mov_b32 exec_lo, s10
v_cmpx_eq_u32_e32 29, v0
v_mov_b32_e32 v10, 28
buffer_store_dword v5, v10, s[0:3], 0 offen
s_endpgm

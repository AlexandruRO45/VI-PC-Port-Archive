; TG_SIZE_EN: one SGPR after the workgroup id. Group 64x2x1 in wave32 = 4 waves; s4 = workgroup id x and
; s5 = 4 waves | wave id << 6 | wave id << 20 (Mesa: [5:0] count, [11:6] ordered id = wave id on GFX10.1,
; [24:20] wave id on GFX10.3). Lane 0 of each wave writes s5 at out[s4 * 4 + wave], with the wave taken from
; its own thread id (y * 2 + x / 32), through the user V# in s[0:3].
; rsrc2: 0xc88
; group: 64 2 1
; Run: vk_compute_test cs_tgsize.spv 2 dump -> out[0..7] = 4 1048644 2097284 3145924, the same for both groups
v_and_b32_e32 v2, 31, v0
v_cmpx_eq_u32_e32 0, v2
v_lshrrev_b32_e32 v3, 5, v0
v_lshl_add_u32 v4, v1, 1, v3
s_lshl_b32 s10, s4, 2
v_add_nc_u32_e32 v4, s10, v4
v_lshlrev_b32_e32 v5, 2, v4
v_mov_b32_e32 v6, s5
buffer_store_dword v6, v5, s[0:3], 0 offen
s_endpgm

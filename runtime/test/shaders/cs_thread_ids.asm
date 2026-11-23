; Compute input registers: v0, v1, v2 are the local thread id x, y, z of a 2x2x2 group.
; RSRC2 0x9088: 4 user SGPRs, TGID_X, TIDIG_COMP_CNT 2 (three thread-id components).
; Thread t = v0 + 2*v1 + 4*v2 stores v0 | v1 << 8 | v2 << 16 at heap word t.
; Run: vk_compute_test cs_thread_ids.spv 1 dump -> out[0..7] = 0 1 256 257 65536 65537 65792 65793
; group: 2,2,2
; rsrc2: 0x9088
v_lshl_or_b32 v3, v1, 8, v0
v_lshl_or_b32 v4, v2, 16, v3
v_lshl_add_u32 v5, v2, 1, v1
v_lshl_add_u32 v6, v5, 1, v0
v_lshlrev_b32_e32 v7, 2, v6
buffer_store_dword v4, v7, s[0:3], 0 offen
s_endpgm

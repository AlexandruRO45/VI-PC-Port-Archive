; group: 32,1,1
; A lane mask computed in the vector unit under a partial EXEC (#210, SHADER-402). vcc_lo = lanes 0-4 (0x1f) and
; s20 = lanes 30-31 (0xc0000000); v_or_b32 of the two runs while EXEC is lanes 0-3, then every lane stores. Hardware:
; lanes 0-3 get the wave's word 0xc000001f, lanes 4-31 keep 0xdeadbeef. Computing each lane's own bit in every lane
; and gathering at the store wrote 0xc000001f into lanes 4-31 as well.
; Expect: out[0..7] = 0xc000001f x4, 0xdeadbeef x4
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mov_b32_e32 v2, 0xdeadbeef
v_cmp_gt_u32_e64 vcc_lo, 5, v1
v_cmp_lt_u32_e64 s20, 29, v1
s_mov_b32 exec_lo, 0x0000000f
v_or_b32_e64 v2, vcc_lo, s20
s_mov_b32 exec_lo, -1
v_lshlrev_b32_e32 v3, 2, v1
buffer_store_dword v2, v3, s[0:3], 0 offen
s_endpgm

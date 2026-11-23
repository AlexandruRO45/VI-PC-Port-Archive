; s_movreld_b32 / s_movreld_b64 / s_movrelsd_2_b32: relative SGPR writes through M0 (RDNA2 ISA SOP1 0x30/0x31/0x49).
; Seeds s20..s27 = 0x11111111, 0x22222222, ..., 0x88888888.
; 1. m0 = 2: s_movreld_b32 s20, 0xcafebabe -> s22 = 0xcafebabe; s30 = s22 + 1 = 0xcafebabf (sgpr_imm invalidated).
; 2. m0 = 3: s_movreld_b32 s20, 42 -> s23 = 42 (0x2a).
; 3. m0 = 1: s_movreld_b64 s[20:21], s[20:21] -> overlapping read-before-write: s21 = 0x11111111, s22 = 0x22222222.
; 4. m0 = 4: s_movreld_b64 s[20:21], s[32:33] -> s24 = 0xdeadbeef, s25 = 0x12345678.
; 5. m0 = 0xfc06fc04 (dst 6, src 4, ignored bits set): s_movrelsd_2_b32 s20, s20 -> s26 = s24 = 0xdeadbeef; s27 kept.
; Lane 0 stores out[0..7] = s30, s21, s22, s23, s24, s25, s26, s27.
; group: 32,1,1
; Run: vk_compute_test cs_w32_s_movreld.spv 1 dump
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s20, 0x11111111
s_mov_b32 s21, 0x22222222
s_mov_b32 s22, 0x33333333
s_mov_b32 s23, 0x44444444
s_mov_b32 s24, 0x55555555
s_mov_b32 s25, 0x66666666
s_mov_b32 s26, 0x77777777
s_mov_b32 s27, 0x88888888
s_mov_b32 m0, 2
s_movreld_b32 s20, 0xcafebabe
s_add_u32 s30, s22, 1
s_mov_b32 m0, 3
s_movreld_b32 s20, 42
s_mov_b32 m0, 1
s_movreld_b64 s[20:21], s[20:21]
s_mov_b32 s32, 0xdeadbeef
s_mov_b32 s33, 0x12345678
s_mov_b32 m0, 4
s_movreld_b64 s[20:21], s[32:33]
s_mov_b32 m0, 0xfc06fc04
s_movrelsd_2_b32 s20, s20
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s30
v_mov_b32_e32 v21, 0
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s21
v_mov_b32_e32 v21, 4
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s22
v_mov_b32_e32 v21, 8
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s23
v_mov_b32_e32 v21, 12
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s24
v_mov_b32_e32 v21, 16
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s25
v_mov_b32_e32 v21, 20
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s26
v_mov_b32_e32 v21, 24
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s27
v_mov_b32_e32 v21, 28
buffer_store_dword v1, v21, s[0:3], 0 offen
s_endpgm

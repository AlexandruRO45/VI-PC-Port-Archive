; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_read_i8_d16 followed by ds_write_b16 of the same register, in wave64: the 16 bits written are the byte
; sign-extended to 16 bits (RDNA2 ISA: ds_read_i8_d16 D[15:0] = sext16(MEM[ADDR][7:0]), D[31:16] kept).
; Found by tools/lds_fuzz.py --seed 9950, program 95 at 64 lanes, on AMD's Windows driver (gfx1036, 32.0.21042.62):
; with the read lowered as OpBitFieldSExtract (variable offset, 8 bits) the written half came back zero-extended
; (0x0090 for the byte 0x90) although the register itself, stored to the buffer, was right. The same SPIR-V module
; gave the right half on an NVIDIA RTX 4080, so it is the driver; the read now sign-extends with xor/subtract.
; LDS dword 0 = 0x80CC90A5, so the bytes at byte address 0..3 are A5 90 CC 80 (all negative); dwords 4..7 start as 0.
; Each read goes to a register that starts as 0xBEEF1234; each ds_write_b16 writes one to the low half of a dword.
;   out[0] dword 4 after the write of the byte at 0 (0xA5) -> 0x0000FFA5   out[4] its register -> 0xBEEFFFA5
;   out[1] dword 5, byte at 1 (0x90)                       -> 0x0000FF90   out[5]              -> 0xBEEFFF90
;   out[2] dword 6, byte at 2 (0xCC)                       -> 0x0000FFCC   out[6]              -> 0xBEEFFFCC
;   out[3] dword 7, byte at 3 (0x80)                       -> 0x0000FF80   out[7]              -> 0xBEEFFF80
; Every lane reads and writes the same values, so the lanes' writes cannot disagree.
; group: 64,1,1
; Run: vk_compute_test cs_w64_lds_i8_d16_write.spv 1 dump
;   -> out[0..7] = 0x0000FFA5 0x0000FF90 0x0000FFCC 0x0000FF80 0xBEEFFFA5 0xBEEFFF90 0xBEEFFFCC 0xBEEFFF80
; mattr: +wavefrontsize64,-wavefrontsize32
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0x80cc90a5
ds_write_b32 v2, v3
v_mov_b32_e32 v3, 0
ds_write_b32 v2, v3 offset:16
ds_write_b32 v2, v3 offset:20
ds_write_b32 v2, v3 offset:24
ds_write_b32 v2, v3 offset:28
s_waitcnt lgkmcnt(0)
s_barrier
v_mov_b32_e32 v10, 0xbeef1234
v_mov_b32_e32 v11, 0xbeef1234
v_mov_b32_e32 v12, 0xbeef1234
v_mov_b32_e32 v13, 0xbeef1234
ds_read_i8_d16 v10, v2
ds_read_i8_d16 v11, v2 offset:1
ds_read_i8_d16 v12, v2 offset:2
ds_read_i8_d16 v13, v2 offset:3
s_waitcnt lgkmcnt(0)
ds_write_b16 v2, v10 offset:16
ds_write_b16 v2, v11 offset:20
ds_write_b16 v2, v12 offset:24
ds_write_b16 v2, v13 offset:28
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v14, v2 offset:16
ds_read_b32 v15, v2 offset:20
ds_read_b32 v16, v2 offset:24
ds_read_b32 v17, v2 offset:28
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v14, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v16, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v17, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v13, v21, s[0:3], 0 offen
s_endpgm

; Scalar stores (GFX10.1 SMEM s_store_dword* / s_buffer_store_dword*, gfx1010-gfx1013 only: llvm-mc rejects them for
; gfx1030/gfx1036, so the RDNA2 oracle can't run them). RDNA1 ISA: SDATA's N dwords are written to memory at the
; address an s_load / s_buffer_load of the same operands reads: s_store from a 64-bit pointer s[p:p+1], s_buffer_store
; from a V# (base = word0 | (word1 & 0xffff) << 32), plus the offset (an immediate, an SGPR, or both "sOFF offset:IMM").
; The translator writes them as PhysicalStorageBuffer stores at those addresses. Until this change they raised
; Unsupported. s[0:3] is the output buffer's V#; s[8:9] is its base as a plain 64-bit pointer.
;   out[0..3] s_buffer_store_dwordx4 s[20:23], s[0:3], 0x0           = 0x11111111 0x22222222 0x33333333 0x44444444
;   out[4..5] s_store_dwordx2 s[24:25], s[8:9], 0x10                 = 0x55555555 0x66666666
;   out[6]    s_buffer_store_dword s26, s[0:3], s5 (s5 = 0x18, a run-time value from v_readfirstlane) = 0x77777777
;   out[7]    s_store_dword s27, s[8:9], s6 offset:0x4 (s6 = 0x18: 0x18 + 4 = 0x1c)                = 0x88888888
; Every invocation writes the same words; nothing else writes the buffer.
; group: 32,1,1
; Run: vk_compute_test cs_w32_scalar_store.spv 1 dump -> out[0..7] = 286331153 572662306 858993459 1145324612 1431655765 1717986918 2004318071 2290649224
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s8, s0
s_and_b32 s9, s1, 0xffff
s_mov_b32 s20, 0x11111111
s_mov_b32 s21, 0x22222222
s_mov_b32 s22, 0x33333333
s_mov_b32 s23, 0x44444444
s_mov_b32 s24, 0x55555555
s_mov_b32 s25, 0x66666666
s_mov_b32 s26, 0x77777777
s_mov_b32 s27, 0x88888888
v_mov_b32_e32 v1, 0x18
v_readfirstlane_b32 s5, v1
s_mov_b32 s6, 0x18
s_buffer_store_dwordx4 s[20:23], s[0:3], 0x0
s_store_dwordx2 s[24:25], s[8:9], 0x10
s_buffer_store_dword s26, s[0:3], s5
s_store_dword s27, s[8:9], s6 offset:0x4
s_endpgm

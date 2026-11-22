; Wave32 code that uses the 64-bit scalar bit counters on plain data and v_mbcnt_hi_u32_b32. LLVM's wave32 output has them
; (clang -mcpu=gfx1036 -mno-wavefrontsize64 turns a 64-bit popcount, ctz and clz of uniform values into s_bcnt1_i32_b64,
; s_ff1_i32_b64 and s_flbit_i32_b64), but the translator read s_bcnt/s_ff _b64 and v_mbcnt_hi as wave64 markers: with a
; wave32 FSSM header it refused such a shader ("FSSM wave32 header contradicts wave64 code"), and without a header it
; translated it as wave64. The header comment below makes build.py write a wave32 FSSM header (magic and the 0xd4 bit 5
; flag), so the header decides. RDNA2 ISA pseudocode, S0 a 64-bit scalar:
;   s_bcnt1_i32_b64 D = number of 1 bits in S0    s_bcnt0_i32_b64 D = number of 0 bits
;   s_ff1_i32_b64   D = index of the lowest 1 bit (-1 if none)    s_ff0_i32_b64 D = index of the lowest 0 bit (-1 if none)
;   v_mbcnt_lo_u32_b32 D = popcount(S0 & ((1 << lane) - 1)) + S1 (lanes 0-31)
;   v_mbcnt_hi_u32_b32 D = popcount of the S0 bits of the lanes below this one among lanes 32-63, plus S1: none for lanes 0-31
; x1 = 0x00000100_80000000 (bits 31 and 40), x2 = 0xffffffff_ffffffff, x3 = 0xffff0000_ffffffff
;   out[0] s_bcnt1_i32_b64(x1) = 2          out[1] s_ff1_i32_b64(x1) = 31
;   out[2] s_bcnt0_i32_b64(x3) = 16         out[3] s_ff0_i32_b64(x3) = 32 (the low word is all ones)
;   out[4] s_bcnt1_i32_b64(x3) = 48         out[5] s_ff0_i32_b64(x2) = -1 (no zero bit)
;   out[6] lane 5 of v_mbcnt_hi(-1, v_mbcnt_lo(-1, 0)) = 5 (the lane id, as in a wave32 HIP kernel)
;   out[7] lane 20 of v_mbcnt_hi(0x0f0f0f0f, v_mbcnt_lo(0x0f0f0f0f, 0)) = 12 (bits 0-3, 8-11, 16-19 are below lane 20; the hi half adds none)
; fssm-wave: 32
; group: 32,1,1
; Run: vk_compute_test cs_w32_s64_data.spv 1 dump -> out[0..7] = 2 31 16 32 48 4294967295 5 12
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s10, 0x80000000
s_mov_b32 s11, 0x100
s_mov_b32 s12, -1
s_mov_b32 s13, -1
s_mov_b32 s14, -1
s_mov_b32 s15, 0xffff0000
s_bcnt1_i32_b64 s20, s[10:11]
s_ff1_i32_b64 s21, s[10:11]
s_bcnt0_i32_b64 s22, s[14:15]
s_ff0_i32_b64 s23, s[14:15]
s_bcnt1_i32_b64 s24, s[14:15]
s_ff0_i32_b64 s25, s[12:13]
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_readlane_b32 s26, v1, 5
v_mbcnt_lo_u32_b32 v2, 0x0f0f0f0f, 0
v_mbcnt_hi_u32_b32 v2, 0x0f0f0f0f, v2
v_readlane_b32 s27, v2, 20
v_mov_b32_e32 v20, s20
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s21
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s22
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s23
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s24
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s25
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s26
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s27
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm

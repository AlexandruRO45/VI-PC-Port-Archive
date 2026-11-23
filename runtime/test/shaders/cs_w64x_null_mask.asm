; Null mask source (docs/design.md contract table): read as a lane mask, the scalar register null supplies zero.
; A high-bit test, a zero-bit count, a first-set-bit search and a vector write under the EXEC that
; s_and_saveexec_b64 leaves, the four checks of the wave64 fixture tests/semantic/fixtures/null_mask_source.asm on
; the CPU interpreter, whose outputs are 1, 64, 0xffffffff, 7. The 32-bit forms and the complements follow.
; RDNA2 ISA (SOP1/SOP2): null reads as 0 for any operand; s_and_saveexec_b64: D = EXEC; EXEC = S0 & EXEC;
; SCC = (EXEC != 0).
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding
; Run: vk_compute_test cs_w64x_null_mask.spv 1 dump --require-wave64
;   (or translated with --wave64-emulate, on any subgroup size)
;   out[0] s_bitcmp0_b64 null, 42        = 1: bit 42 of the 64-bit zero
;   out[1] s_bcnt0_i32_b64 null          = 64 zero bits
;   out[2] s_ff1_i32_b64 null            = 0xffffffff: no set bit to find
;   out[3] the vector write under EXEC   = 7: v3 = 9 (after s_and_saveexec_b64 s[4:5], null) landed in no lane
;   out[4] the same write with EXEC back = 9: s_mov_b64 exec, -1 restored it, so out[3] is the null mask
;   out[5] s_bcnt1_i32_b64 null          = 0 set bits
;   out[6] s_bcnt0_i32_b32 null          = 32 zero bits of the low word
;   out[7] s_ff1_i32_b32 null            = 0xffffffff
; A null read as -1 instead of 0 gives out[0..7] = 1 32 0 9 9 32 32 4294967295: out[1] is the zero count of a
; one-word mask, out[2] the first set bit of it, and out[3] has the write landing because EXEC stays on.
; mattr: +wavefrontsize64,-wavefrontsize32
s_bitcmp0_b64 null, 42
s_cselect_b32 s10, 1, 0
s_bcnt0_i32_b64 s11, null
s_ff1_i32_b64 s12, null
v_mov_b32_e32 v3, 7
s_and_saveexec_b64 s[4:5], null
v_mov_b32_e32 v3, 9
s_mov_b64 exec, -1
v_mov_b32_e32 v4, v3
v_mov_b32_e32 v3, 9
s_bcnt1_i32_b64 s13, null
s_bcnt0_i32_b32 s14, null
s_ff1_i32_b32 s15, null
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v4, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v3, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s14
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm

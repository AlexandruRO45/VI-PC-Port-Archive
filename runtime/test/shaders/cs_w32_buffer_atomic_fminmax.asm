; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Atomic natural alignment, 64-bit all-or-nothing bounds, and buffer_atomic_{fmin,fmax} (#2710, docs/design.md).
; A 32-thread wave32 workgroup tests:
;   out[0] = buffer_atomic_fmin on dword 0x40 (seeded +0.0): fmin(+0.0, -0.0) stores -0.0, then non-glc
;            fmin(-0.0, qNaN) keeps the non-NaN -0.0; a final glc fmin with 1.0 returns old -0.0        = 0x80000000
;   out[1] = buffer_atomic_fmax on dword 0x44 (seeded qNaN): glc fmax(qNaN, -2.0) stores -2.0, returned
;            by the next glc fmax(-2.0, -0.0)                                                           = 0xC0000000
;   out[2] = dword 0x44 after non-glc buffer_atomic_fmax(-0.0, +0.0): NMax ranks +0.0 above -0.0        = 0x00000000
;   out[3] = dword 0x48 (seeded -1.0) after all 32 lanes contend with buffer_atomic_fmax of
;            0x3F800000 + (lane << 16): lane 31's maximum wins regardless of CAS order                  = 0x3F9F0000
;   out[4] = dword 0x50 (seeded 1.0) ORed with active lane 0's glc returns from unaligned
;            buffer_atomic_fmax at 0x52 and buffer_atomic_add at 0x51: both drop and return 0           = 0x3F800000
;   out[5] = inactive lane 1's v10 after the unaligned buffer_atomic_fmax glc: vdata is preserved       = 0x4000CAFE
;   out[6] = buffer_atomic_swap_x2 on V# base 0x60, num_records 12 (seeded 0x11112222, 0x33334444 at +0
;            and 0xAAAABBBB, 0xCCCCDDDD at +8): offset 4 is in bounds (4 + 8 <= 12) but fails 8-byte
;            alignment ((0x64 & 7) != 0), returning (0, 0) and leaving +4 intact for the aligned +0
;            swap_x2 (old high dword ORed with the dropped glc pair)                                    = 0x33334444
;   out[7] = dword 0x68 ORed with the glc pair of buffer_atomic_swap_x2 at offset 8 on the same V#
;            (8-byte aligned, low dword inside num_records 12, high dword past 12): all-or-nothing
;            8-byte bounds drops both dwords and returns (0, 0)                                         = 0xAAAABBBB
; group: 32,1,1
; Run: vk_compute_test cs_w32_buffer_atomic_fminmax.spv 1 dump -> out[0..7] = 2147483648 3221225472 0 1067384832 1065353216 1073793790 858997828 2863315899
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmp_eq_u32_e32 vcc_lo, 0, v0
s_and_saveexec_b32 s12, vcc_lo
v_mov_b32_e32 v20, 0x40
v_mov_b32_e32 v21, 0
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x44
v_mov_b32_e32 v21, 0x7fc00000
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x48
v_mov_b32_e32 v21, 0xbf800000
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x50
v_mov_b32_e32 v21, 1.0
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x60
v_mov_b32_e32 v21, 0x11112222
v_mov_b32_e32 v22, 0x33334444
buffer_store_dwordx2 v[21:22], v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x68
v_mov_b32_e32 v21, 0xaaaabbbb
v_mov_b32_e32 v22, 0xccccdddd
buffer_store_dwordx2 v[21:22], v20, s[0:3], 0 offen
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0
v_mov_b32_e32 v20, 0x40
v_mov_b32_e32 v2, 0x80000000
buffer_atomic_fmin v2, v20, s[0:3], 0 offen glc
v_mov_b32_e32 v3, 0x7fc00000
buffer_atomic_fmin v3, v20, s[0:3], 0 offen
v_mov_b32_e32 v4, 1.0
buffer_atomic_fmin v4, v20, s[0:3], 0 offen glc
v_mov_b32_e32 v20, 0x44
v_mov_b32_e32 v5, -2.0
buffer_atomic_fmax v5, v20, s[0:3], 0 offen glc
v_mov_b32_e32 v6, 0x80000000
buffer_atomic_fmax v6, v20, s[0:3], 0 offen glc
v_mov_b32_e32 v7, 0
buffer_atomic_fmax v7, v20, s[0:3], 0 offen
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0
buffer_load_dword v8, v20, s[0:3], 0 offen glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 0
buffer_store_dword v4, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
buffer_store_dword v6, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
buffer_store_dword v8, v20, s[0:3], 0 offen
s_waitcnt_vscnt null, 0
s_mov_b32 exec_lo, -1
s_barrier
v_lshlrev_b32_e32 v9, 16, v0
v_add_nc_u32_e32 v9, 0x3f800000, v9
v_mov_b32_e32 v20, 0x48
buffer_atomic_fmax v9, v20, s[0:3], 0 offen glc
v_mov_b32_e32 v10, 0x4000cafe
s_waitcnt vmcnt(0)
s_barrier
v_cmp_eq_u32_e32 vcc_lo, 0, v0
s_mov_b32 exec_lo, vcc_lo
v_mov_b32_e32 v20, 0x48
buffer_load_dword v9, v20, s[0:3], 0 offen glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 12
buffer_store_dword v9, v20, s[0:3], 0 offen
v_mov_b32_e32 v21, 0x52
buffer_atomic_fmax v10, v21, s[0:3], 0 offen glc
v_mov_b32_e32 v11, 0x12345678
v_mov_b32_e32 v22, 0x51
buffer_atomic_add v11, v22, s[0:3], 0 offen glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 0x50
buffer_load_dword v18, v20, s[0:3], 0 offen glc
s_waitcnt vmcnt(0)
v_or_b32_e32 v18, v18, v10
v_or_b32_e32 v18, v18, v11
v_mov_b32_e32 v20, 16
buffer_store_dword v18, v20, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmp_eq_u32_e32 vcc_lo, 1, v0
s_mov_b32 exec_lo, vcc_lo
v_mov_b32_e32 v20, 20
buffer_store_dword v10, v20, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmp_eq_u32_e32 vcc_lo, 0, v0
s_mov_b32 exec_lo, vcc_lo
s_mov_b32 s8, 0x60
s_mov_b32 s9, 0
s_mov_b32 s10, 12
s_mov_b32 s11, 0x31016fac
v_mov_b32_e32 v12, 0x55556666
v_mov_b32_e32 v13, 0x77778888
v_mov_b32_e32 v23, 4
buffer_atomic_swap_x2 v[12:13], v23, s[8:11], 0 offen glc
v_mov_b32_e32 v14, 0x99990000
v_mov_b32_e32 v15, 0xeeeeffff
v_mov_b32_e32 v24, 8
buffer_atomic_swap_x2 v[14:15], v24, s[8:11], 0 offen glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v16, 0x55556666
v_mov_b32_e32 v17, 0x77778888
v_mov_b32_e32 v25, 0
buffer_atomic_swap_x2 v[16:17], v25, s[8:11], 0 offen glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 0x68
buffer_load_dword v19, v20, s[0:3], 0 offen glc
s_waitcnt vmcnt(0)
v_or_b32_e32 v17, v17, v12
v_or_b32_e32 v17, v17, v13
v_or_b32_e32 v19, v19, v14
v_or_b32_e32 v19, v19, v15
v_mov_b32_e32 v20, 24
buffer_store_dword v17, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
buffer_store_dword v19, v20, s[0:3], 0 offen
s_endpgm

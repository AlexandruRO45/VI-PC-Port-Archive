; TEST-593: scalar register lists that end in vcc_lo,vcc_hi (LLVM's VCCTuple, as FidelityFX FSR4's FSPX programs
; print them) on the GPU (docs/design.md, "Scalar/buffer memory"; CPU-checked by tests/semantic/test_fspx_rules.py).
; Thread 0 seeds heap dwords 0x40..0x7f with 0xf5f00000 + i (i = 0..15), a V# at 0x80 (base 0xc0, stride 0,
; NUM_RECORDS 16, RAW) and the words 0xcafe1234 0xbeef5678 at 0xc0. The loads use a V# with base 0x40 and
; NUM_RECORDS 0x50. Words, in out[0..7]:
;   0..3 s_buffer_load_dwordx8 [s100..s105,vcc_lo,vcc_hi]: s100 = i 0, s105 = i 5, vcc_lo = i 6, vcc_hi = i 7
;   4, 5 s_buffer_load_dwordx16 [s92..s105,vcc_lo,vcc_hi]: s92 = i 0, vcc_hi = i 15
;   6    a V# loaded into [s104,s105,vcc_lo,vcc_hi] (from 0x80), used by s_buffer_load_dword at 0: 0xcafe1234
;   7    the same V# used by buffer_load_dword at 4: 0xbeef5678
; The five instructions on registers that run into vcc_lo,vcc_hi are written as LLVM 23 prints them for
; gfx1030-class targets (docs/fspx.md), with their machine words: llvm-mc 21 and gfx1013 neither assemble nor
; decode them, so run_gpu_tests.py passes a line with `; encoding: [...]` to the translator as written. Each
; is the encoding llvm-mc gives the same instruction on s[96:103] / s[88:103] / s[100:103] with the register
; field moved: SMEM SDATA (word 0 bits 12:6) to 100, 92 or 104, SBASE (bits 5:0) to 52, MUBUF SRSRC (word 1
; bits 20:16) to 26.
; One thread, no wave64 requirement.
; Run: vk_compute_test cs_fspx_slist.spv 1 dump -> out[0..7] = 0xf5f00000 0xf5f00005 0xf5f00006 0xf5f00007
; 0xf5f00000 0xf5f0000f 0xcafe1234 0xbeef5678.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0xf5f00000
v_mov_b32 v2, 0xf5f00001
v_mov_b32 v3, 0xf5f00002
v_mov_b32 v4, 0xf5f00003
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:64
v_mov_b32 v1, 0xf5f00004
v_mov_b32 v2, 0xf5f00005
v_mov_b32 v3, 0xf5f00006
v_mov_b32 v4, 0xf5f00007
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:80
v_mov_b32 v1, 0xf5f00008
v_mov_b32 v2, 0xf5f00009
v_mov_b32 v3, 0xf5f0000a
v_mov_b32 v4, 0xf5f0000b
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:96
v_mov_b32 v1, 0xf5f0000c
v_mov_b32 v2, 0xf5f0000d
v_mov_b32 v3, 0xf5f0000e
v_mov_b32 v4, 0xf5f0000f
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:112
v_mov_b32 v1, 0xc0
v_mov_b32 v2, 0
v_mov_b32 v3, 16
v_mov_b32 v4, 0x31016fac
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:128
v_mov_b32 v1, 0xcafe1234
v_mov_b32 v2, 0xbeef5678
buffer_store_dwordx2 v[1:2], off, s[0:3], 0 offset:192
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0
s_mov_b32 s6, 0x50
s_mov_b32 s7, 0x31016fac
s_buffer_load_dwordx8 [s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi], s[4:7], 0x0 ; encoding: [0x02,0x19,0x2c,0xf4,0x00,0x00,0x00,0xfa]
s_waitcnt lgkmcnt(0)
v_mov_b32 v10, s100
v_mov_b32 v11, s105
v_mov_b32 v12, vcc_lo
v_mov_b32 v13, vcc_hi
s_buffer_load_dwordx16 [s92,s93,s94,s95,s96,s97,s98,s99,s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi], s[4:7], 0x0 ; encoding: [0x02,0x17,0x30,0xf4,0x00,0x00,0x00,0xfa]
s_waitcnt lgkmcnt(0)
v_mov_b32 v14, s92
v_mov_b32 v15, vcc_hi
s_buffer_load_dwordx4 [s104,s105,vcc_lo,vcc_hi], s[4:7], 0x40 ; encoding: [0x02,0x1a,0x28,0xf4,0x40,0x00,0x00,0xfa]
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s10, [s104,s105,vcc_lo,vcc_hi], 0x0 ; encoding: [0xb4,0x02,0x20,0xf4,0x00,0x00,0x00,0xfa]
v_mov_b32 v1, 4
buffer_load_dword v17, v1, [s104,s105,vcc_lo,vcc_hi], 0 offen ; encoding: [0x00,0x10,0x30,0xe0,0x01,0x11,0x1a,0x80]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32 v16, s10
buffer_store_dwordx4 v[10:13], off, s[0:3], 0
buffer_store_dwordx4 v[14:17], off, s[0:3], 0 offset:16
s_endpgm

; group: 64,1,1
; rsrc2: 0x808c
; fssm-at: 0x150
; fssm-data: 0x35353531 0x45524f54
; Typed image_store to a B5G5R5A1_UNORM UAV texture (T# FORMAT 134, 1_5_5_5_UNORM): each thread stores four floats
; it reads from a raw buffer. Run: d3d12_b5g5r5a1_store_test (a compute PSO found by the SHA-256 of its synthetic
; FSSM header), which reads the texture back as packed 16-bit texels: where the GPU has no A1R5G5B5 storage the
; layer's storage view is R16_UINT, and the translator packs the texel for it.
; User data, as cs_m4_fill.asm: [0] four 32-bit constants (unused), [1] a descriptor table of two UAVs. RSRC2
; 0x808c = six user SGPRs and TGID_X, so the group index is s6. runtime/CMakeLists.txt builds this with
; LEONIDA_USER_DATA_SHIFT=4, so in the PSO every SGPR named here is 4 higher.
;   s[4:5] = console VA of the table's first slot (32 bytes a slot)
;   slot 0 (+0x00): T#, 16x16 B5G5R5A1_UNORM image (written)
;   slot 1 (+0x20): V#, raw buffer, 256 texels of four f32 (read)
; g = 64 * group + lane, 0..255:  image(x = g & 15, y = g >> 4) = input[4g .. 4g + 3]
s_load_dwordx8 s[8:15], s[4:5], 0x0
s_load_dwordx4 s[16:19], s[4:5], 0x20
s_lshl_b32 s7, s6, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s7, v0
v_lshlrev_b32_e32 v2, 4, v1
buffer_load_dwordx4 v[3:6], v2, s[16:19], 0 offen
s_waitcnt vmcnt(0)
v_and_b32_e32 v8, 15, v1
v_lshrrev_b32_e32 v9, 4, v1
image_store v[3:6], v[8:9], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
s_endpgm

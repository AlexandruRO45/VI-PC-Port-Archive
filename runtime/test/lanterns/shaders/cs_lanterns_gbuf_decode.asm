; Lanterns G-buffer decode (stage 22): reads five of the six targets back through T#s and decodes them as a title's
; lighting pass does: byte IDs by multiplying by 255 and truncating, the 10-bit codes by rounding, the motion as it is,
; and a lighting term from the decoded values. One wave32 group per 32 pixels of a row (TGID_X: the segment, TGID_Y:
; the row). lanterns_gbuf6.h computes the same 32 bytes a pixel:
;   w0 = the mrt0 IDs (byte 0 = red), w1 = the mrt1 codes (10 bits each, red lowest), w2 and w3 = the motion floats,
;   w4 = lighting (id0 * code0 + code1) / 4 + emissive red, w5 = the mrt5 IDs, w6 = w7 = 0.
; s[0:1]: the table: slots 0-5 the targets' T#s (slot 3, the sRGB albedo, is not read here), slot 6 the output's raw V#.
; stage: cs
; wave: 32
; group: 32,1,1
; rsrc2: 0x18a
;
; RSRC2 0x184 before D3D-448 (now 0x18a, see the moves below): two user SGPRs (the table), TGID_X and TGID_Y (s2, s3).
; D3D-448: a D3D12 compute PSO gets its root arguments in the console layout (rt::rootsig::pack_stage,
; docs/d3d12_vulkan.md section 3.5): from s4 in root-signature order, a descriptor table as ONE dword (the 32-bit
; VA of its first descriptor; the heaps live below 4 GiB), a constant one dword per value; s0..s3 are the driver's.
; RSRC2 0x18a therefore counts 5 user SGPRs (4 + 1), and the thread-group ids follow at s5. These moves rebuild the
; layout the code below was written for: table s4 -> s[0:1] (high half 0); thread-group ids s5..s6 -> s2..s3.
s_mov_b32 s0, s4
s_mov_b32 s1, 0
s_mov_b32 s2, s5
s_mov_b32 s3, s6
s_load_dwordx8 s[4:11], s[0:1], 0x0
s_load_dwordx8 s[12:19], s[0:1], 0x20
s_load_dwordx8 s[20:27], s[0:1], 0x40
s_load_dwordx8 s[28:35], s[0:1], 0x80
s_load_dwordx8 s[36:43], s[0:1], 0xa0
s_load_dwordx4 s[44:47], s[0:1], 0xc0
s_lshl_b32 s48, s2, 5
v_add_nc_u32_e32 v1, s48, v0
v_mov_b32_e32 v2, s3
v_mul_u32_u24_e32 v3, 0x180, v2
v_add_nc_u32_e32 v3, v3, v1
v_lshlrev_b32_e32 v3, 5, v3
s_waitcnt lgkmcnt(0)
image_load v[4:7], v[1:2], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D
image_load v[8:11], v[1:2], s[12:19] dmask:0xf dim:SQ_RSRC_IMG_2D
image_load v[12:13], v[1:2], s[20:27] dmask:0x3 dim:SQ_RSRC_IMG_2D
image_load v14, v[1:2], s[28:35] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_load v[16:19], v[1:2], s[36:43] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
; byte IDs: truncate(v * 255)
v_mul_f32_e32 v4, 0x437f0000, v4
v_mul_f32_e32 v5, 0x437f0000, v5
v_mul_f32_e32 v6, 0x437f0000, v6
v_mul_f32_e32 v7, 0x437f0000, v7
v_cvt_u32_f32_e32 v4, v4
v_cvt_u32_f32_e32 v5, v5
v_cvt_u32_f32_e32 v6, v6
v_cvt_u32_f32_e32 v7, v7
v_lshl_or_b32 v20, v5, 8, v4
v_lshl_or_b32 v20, v6, 16, v20
v_lshl_or_b32 v20, v7, 24, v20
; 10-bit codes: round(v * 1023)
v_mov_b32_e32 v30, 0x447fc000
v_fma_f32 v8, v8, v30, 0.5
v_fma_f32 v9, v9, v30, 0.5
v_fma_f32 v10, v10, v30, 0.5
v_cvt_u32_f32_e32 v8, v8
v_cvt_u32_f32_e32 v9, v9
v_cvt_u32_f32_e32 v10, v10
v_lshl_or_b32 v21, v9, 10, v8
v_lshl_or_b32 v21, v10, 20, v21
v_mov_b32_e32 v22, v12
v_mov_b32_e32 v23, v13
; lighting: (id0 * code0 + code1) / 4 + emissive red
v_cvt_f32_u32_e32 v24, v4
v_cvt_f32_u32_e32 v25, v8
v_cvt_f32_u32_e32 v26, v9
v_fma_f32 v24, v24, v25, v26
v_fma_f32 v24, v24, 0x3e800000, v14
; second IDs
v_mul_f32_e32 v16, 0x437f0000, v16
v_mul_f32_e32 v17, 0x437f0000, v17
v_mul_f32_e32 v18, 0x437f0000, v18
v_mul_f32_e32 v19, 0x437f0000, v19
v_cvt_u32_f32_e32 v16, v16
v_cvt_u32_f32_e32 v17, v17
v_cvt_u32_f32_e32 v18, v18
v_cvt_u32_f32_e32 v19, v19
v_lshl_or_b32 v25, v17, 8, v16
v_lshl_or_b32 v25, v18, 16, v25
v_lshl_or_b32 v25, v19, 24, v25
v_mov_b32_e32 v26, 0
v_mov_b32_e32 v27, 0
buffer_store_dwordx4 v[20:23], v3, s[44:47], 0 offen
buffer_store_dwordx4 v[24:27], v3, s[44:47], 0 offen offset:16
s_endpgm

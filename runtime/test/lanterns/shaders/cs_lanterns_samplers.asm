; Lanterns samplers (stage 20): three samplers held as literal S#s, as the console's compiler writes a root signature's
; static samplers (lanterns_samplers.h). One wave32 group of 32 lanes, one probe per lane.
; s[0:1]: the table (2 user SGPRs, the layout before D3D-448, which the moves below rebuild). Slot 0 (+ 0x0): the
; colour texture's T# (R32_FLOAT 4x4).
; Slot 1 (+ 0x20): the depth texture's T# (its R32_FLOAT view). Slot 2 (+ 0x40): a raw V# of the output, 16 bytes a lane.
; Literal S#s (all CLAMP_X/Y/Z 2 = clamp to the last texel, MIP_FILTER 1 = point, MIP_POINT_PRECLAMP set):
;   s[24:27] POINT        0x92 0 0x14000000 0
;   s[28:31] LINEAR       0x92 0 0x14500000 0   (XY_MAG_FILTER and XY_MIN_FILTER 1 = bilinear)
;   s[32:35] LESS_EQUAL   0x3092 0 0x14000000 0 (DEPTH_COMPARE_FUNC 3, point filtering)
; Lane i: x = (i & 7) * 0.5 + 0.25 and y = (i >> 3) + ((i >> 2) & 1) * 0.5 + 0.25 texels, u = x / 4, v = y / 4;
; comparison value 0.3 + 0.2 * (i & 3). It stores [POINT, LINEAR, LESS_EQUAL on depth, POINT on depth].
; stage: cs
; wave: 32
; group: 32,1,1
; rsrc2: 0xa
;
; RSRC2 0x4 before D3D-448 (now 0xa, see the moves below): two user SGPRs (s[0:1], the table), no thread-group IDs.
; D3D-448: a D3D12 compute PSO gets its root arguments in the console layout (rt::rootsig::pack_stage,
; docs/d3d12_vulkan.md section 3.5): from s4 in root-signature order, a descriptor table as ONE dword (the 32-bit
; VA of its first descriptor; the heaps live below 4 GiB), a constant one dword per value; s0..s3 are the driver's.
; RSRC2 0xa therefore counts 5 user SGPRs (4 + 1). These moves rebuild the
; layout the code below was written for: table s4 -> s[0:1] (high half 0).
s_mov_b32 s0, s4
s_mov_b32 s1, 0
s_load_dwordx8 s[4:11], s[0:1], 0x0
s_load_dwordx8 s[12:19], s[0:1], 0x20
s_load_dwordx4 s[20:23], s[0:1], 0x40
s_mov_b32 s24, 0x92
s_mov_b32 s25, 0
s_mov_b32 s26, 0x14000000
s_mov_b32 s27, 0
s_mov_b32 s28, 0x92
s_mov_b32 s29, 0
s_mov_b32 s30, 0x14500000
s_mov_b32 s31, 0
s_mov_b32 s32, 0x3092
s_mov_b32 s33, 0
s_mov_b32 s34, 0x14000000
s_mov_b32 s35, 0
; v[1:3] = u, v, LOD 0
v_and_b32_e32 v1, 7, v0
v_cvt_f32_u32_e32 v1, v1
v_fma_f32 v1, v1, 0.5, 0.25
v_bfe_u32 v2, v0, 2, 1
v_cvt_f32_u32_e32 v2, v2
v_lshrrev_b32_e32 v3, 3, v0
v_cvt_f32_u32_e32 v3, v3
v_fma_f32 v2, v2, 0.5, v3
v_add_f32_e32 v2, 0.25, v2
v_mul_f32_e32 v1, 0.25, v1
v_mul_f32_e32 v2, 0.25, v2
v_mov_b32_e32 v3, 0
; v[10:13] = comparison value, u, v, LOD 0
v_and_b32_e32 v4, 3, v0
v_cvt_f32_u32_e32 v4, v4
v_mov_b32_e32 v10, 0x3e99999a
v_fmac_f32_e32 v10, 0x3e4ccccd, v4
v_mov_b32_e32 v11, v1
v_mov_b32_e32 v12, v2
v_mov_b32_e32 v13, 0
v_lshlrev_b32_e32 v14, 4, v0
s_waitcnt lgkmcnt(0)
image_sample_l v6, v[1:3], s[4:11], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_l v7, v[1:3], s[4:11], s[28:31] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c_l v8, v[10:13], s[12:19], s[32:35] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_l v9, v[1:3], s[12:19], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[6:9], v14, s[20:23], 0 offen
s_endpgm

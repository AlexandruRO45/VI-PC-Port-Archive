; Lanterns composite pixel shader (stage 5): the scene texel under the pixel plus the bloom times 2^-19, into the
; final target. The pixel comes from SV_Position (x and y read: the ISG1 part fssm_pack.py writes for "position: xy",
; which puts them in v2 and v3); no interpolants.
; stage: ps
; wave: 32
; position: xy
;
; s[0:1] table: slot 0 the scene's T# (R16G16B16A16_FLOAT), slot 1 (+ 0x20) the bloom's V# (stride 16).
s_load_dwordx8 s[4:11], s[0:1], 0x0
s_load_dwordx4 s[12:15], s[0:1], 0x20
v_cvt_u32_f32_e32 v4, v2
v_cvt_u32_f32_e32 v5, v3
v_mul_u32_u24_e32 v10, 0x180, v5
v_add_nc_u32_e32 v10, v10, v4
s_waitcnt lgkmcnt(0)
image_load v[6:9], v[4:5], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D
buffer_load_dwordx3 v[11:13], v10, s[12:15], 0 idxen
s_waitcnt vmcnt(0)
v_fmac_f32_e32 v6, 0x36000000, v11
v_fmac_f32_e32 v7, 0x36000000, v12
v_fmac_f32_e32 v8, 0x36000000, v13
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm

; Lanterns G-buffer round trip (stage 22): a fullscreen pixel shader that writes six targets, every one through a packed
; half-float export (v_cvt_pkrtz_f16_f32 pairs, exp ... compr), as the title's deferred pixel shaders do. Everything comes
; from the pixel's integer position (x, y from SV_Position); lanterns_gbuf6.h computes the same.
;   mrt0 RGBA8 UNORM     byte IDs: x & 255, y & 255, (x + 3y) & 255, (x ^ y) & 255, each times 1/255
;   mrt1 RGB10A2 UNORM   codes: x & 511, 2y & 511, (x + y) & 511 times 1/1023; alpha (x & 3) times 1/3
;   mrt2 RG16 FLOAT      signed motion: (x - 192) / 8, (128 - y) / 16
;   mrt3 RGBA8 sRGB      albedo (linear): x & 255, y & 255, (x y) & 255 times 1/255; alpha 1
;   mrt4 RGBA16 FLOAT    emissive: ((x + y) & 511) / 4, -(x & 255) / 2, (y & 127) * 2, 1
;   mrt5 RGBA8 UNORM     second IDs: 7x & 255, 5y & 255, 11 (x + y) & 255, 255 - (x & 255), each times 1/255
; stage: ps
; wave: 32
; position: xy
v_cvt_u32_f32_e32 v4, v2
v_cvt_u32_f32_e32 v5, v3
; mrt0
v_and_b32_e32 v6, 0xff, v4
v_and_b32_e32 v7, 0xff, v5
v_mad_u32_u24 v8, v5, 3, v4
v_and_b32_e32 v8, 0xff, v8
v_xor_b32_e32 v9, v4, v5
v_and_b32_e32 v9, 0xff, v9
v_cvt_f32_u32_e32 v6, v6
v_cvt_f32_u32_e32 v7, v7
v_cvt_f32_u32_e32 v8, v8
v_cvt_f32_u32_e32 v9, v9
v_mul_f32_e32 v6, 0x3b808081, v6
v_mul_f32_e32 v7, 0x3b808081, v7
v_mul_f32_e32 v8, 0x3b808081, v8
v_mul_f32_e32 v9, 0x3b808081, v9
v_cvt_pkrtz_f16_f32 v40, v6, v7
v_cvt_pkrtz_f16_f32 v41, v8, v9
; mrt1
v_and_b32_e32 v10, 0x1ff, v4
v_lshlrev_b32_e32 v11, 1, v5
v_and_b32_e32 v11, 0x1ff, v11
v_add_nc_u32_e32 v12, v4, v5
v_and_b32_e32 v12, 0x1ff, v12
v_and_b32_e32 v13, 3, v4
v_cvt_f32_u32_e32 v10, v10
v_cvt_f32_u32_e32 v11, v11
v_cvt_f32_u32_e32 v12, v12
v_cvt_f32_u32_e32 v13, v13
v_mul_f32_e32 v10, 0x3a802008, v10
v_mul_f32_e32 v11, 0x3a802008, v11
v_mul_f32_e32 v12, 0x3a802008, v12
v_mul_f32_e32 v13, 0x3eaaaaab, v13
v_cvt_pkrtz_f16_f32 v42, v10, v11
v_cvt_pkrtz_f16_f32 v43, v12, v13
; mrt2
v_subrev_nc_u32_e32 v14, 0xc0, v4
v_cvt_f32_i32_e32 v14, v14
v_mul_f32_e32 v14, 0x3e000000, v14
v_sub_nc_u32_e32 v15, 0x80, v5
v_cvt_f32_i32_e32 v15, v15
v_mul_f32_e32 v15, 0x3d800000, v15
v_cvt_pkrtz_f16_f32 v44, v14, v15
v_mov_b32_e32 v45, 0
; mrt3 (v6 and v7 are x & 255 and y & 255 over 255 already)
v_mul_u32_u24_e32 v16, v4, v5
v_and_b32_e32 v16, 0xff, v16
v_cvt_f32_u32_e32 v16, v16
v_mul_f32_e32 v16, 0x3b808081, v16
v_mov_b32_e32 v17, 1.0
v_cvt_pkrtz_f16_f32 v46, v6, v7
v_cvt_pkrtz_f16_f32 v47, v16, v17
; mrt4
v_add_nc_u32_e32 v18, v4, v5
v_and_b32_e32 v18, 0x1ff, v18
v_cvt_f32_u32_e32 v18, v18
v_mul_f32_e32 v18, 0x3e800000, v18
v_and_b32_e32 v19, 0xff, v4
v_cvt_f32_u32_e32 v19, v19
v_mul_f32_e32 v19, -0.5, v19
v_and_b32_e32 v20, 0x7f, v5
v_cvt_f32_u32_e32 v20, v20
v_mul_f32_e32 v20, 2.0, v20
v_mov_b32_e32 v21, 1.0
v_cvt_pkrtz_f16_f32 v48, v18, v19
v_cvt_pkrtz_f16_f32 v49, v20, v21
; mrt5
v_mul_u32_u24_e32 v22, 7, v4
v_and_b32_e32 v22, 0xff, v22
v_mul_u32_u24_e32 v23, 5, v5
v_and_b32_e32 v23, 0xff, v23
v_add_nc_u32_e32 v24, v4, v5
v_mul_u32_u24_e32 v24, 11, v24
v_and_b32_e32 v24, 0xff, v24
v_and_b32_e32 v25, 0xff, v4
v_sub_nc_u32_e32 v25, 0xff, v25
v_cvt_f32_u32_e32 v22, v22
v_cvt_f32_u32_e32 v23, v23
v_cvt_f32_u32_e32 v24, v24
v_cvt_f32_u32_e32 v25, v25
v_mul_f32_e32 v22, 0x3b808081, v22
v_mul_f32_e32 v23, 0x3b808081, v23
v_mul_f32_e32 v24, 0x3b808081, v24
v_mul_f32_e32 v25, 0x3b808081, v25
v_cvt_pkrtz_f16_f32 v50, v22, v23
v_cvt_pkrtz_f16_f32 v51, v24, v25
exp mrt0 v40, v40, v41, v41 compr
exp mrt1 v42, v42, v43, v43 compr
exp mrt2 v44, v44, v45, v45 compr
exp mrt3 v46, v46, v47, v47 compr
exp mrt4 v48, v48, v49, v49 compr
exp mrt5 v50, v50, v51, v51 done compr vm
s_endpgm

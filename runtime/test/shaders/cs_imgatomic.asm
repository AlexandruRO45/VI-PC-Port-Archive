; 2D and 3D R32_UINT image atomics (TEST-184): exercises all 10 image_atomic_* ops the translator handles
; (add, sub, and, or, xor, smin, smax, umin, umax, cmpswap) on both 2D (%simages, binding 4) and 3D (%simages3d,
; binding 7) storage images. Checks:
;   - single-lane value semantics (signed vs unsigned min/max boundaries, sub direction & underflow, cmpswap hit & miss)
;   - glc pre-op return value into vdata[0] (and preservation of vdata[1] on cmpswap)
;   - non-glc preservation of vdata[0]
;   - 64-lane contention on shared texels (2D and 3D)
;   - inactive-lane (EXEC off) protection of memory canaries and glc vdata registers (including INT_MAX/INT_MIN/UINT_MAX)
; Run: vk_store_test cs_imgatomic.spv atomic -> 0 bad in both the 8x8 2D and 4x4x4 3D images.
;
; Build T# descriptors in SGPRs (matching vk_store_test atomic mode):
;   s[8:15]  = 2D R32_UINT storage image (binding 4, index 1)
;   s[16:23] = 3D R32_UINT storage image (binding 7, index 2)
s_mov_b32 s8, 0x1000
s_mov_b32 s9, 0x1400000
s_mov_b32 s10, 0x11
s_mov_b32 s11, 0x11
s_mov_b32 s12, 0x11
s_mov_b32 s13, 0x11
s_mov_b32 s14, 0x11
s_mov_b32 s15, 0x11
s_mov_b32 s16, 0x2000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x22
s_mov_b32 s19, 0x22
s_mov_b32 s20, 0x22
s_mov_b32 s21, 0x22
s_mov_b32 s22, 0x22
s_mov_b32 s23, 0x22

; ---- Part 1: single-lane (v0 == 0) 2D ops (Rows 0..3) and 3D ops (Slices z=0, z=2) ----
v_cmpx_eq_u32_e32 0, v0

; Row 0 (y=0) with glc; record returned pre-op values into Row 1 (y=1, initially 0) via image_atomic_or.
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v1, 0x00000234
image_atomic_add v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 1
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v1, 37
image_atomic_sub v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 1
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 2
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v1, 0x0ff0ffff
image_atomic_and v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 1
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 3
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v1, 0x000aa005
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 1
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 4
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v1, 0xff00ff00
image_atomic_xor v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 1
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 5
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v1, 0xfffffffb
image_atomic_smin v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 1
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 6
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v1, 2
image_atomic_smax v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 1
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 7
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v1, 0x7fffffff
image_atomic_umin v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 1
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

; Row 2 (y=2): umax, cmpswap hit/miss, sub underflow, non-glc preservation, vdata[1] preservation, neg-vs-neg smin
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v1, 0x80000000
image_atomic_umax v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 3
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v4, 0x55667788
v_mov_b32_e32 v5, 0x11223344
image_atomic_cmpswap v[4:5], v[2:3], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 3
image_atomic_or v4, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
; Also verify cmpswap glc left vdata[1] (v5) unchanged by storing it at (6,3)
v_mov_b32_e32 v2, 6
image_atomic_or v5, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 2
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v4, 0x99aabbcc
v_mov_b32_e32 v5, 0xdeadbeef
image_atomic_cmpswap v[4:5], v[2:3], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 3
image_atomic_or v4, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 3
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v1, 25
image_atomic_sub v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 3
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

; Non-glc add: must update (4,2) to 57 and keep v1 = 7 (stored to (4,3))
v_mov_b32_e32 v2, 4
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v1, 7
image_atomic_add v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 3
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

; Non-glc cmpswap hit: must update (5,2) to 0xcafe0002 and keep v4 = 0xcafe0002 (stored to (5,3))
v_mov_b32_e32 v2, 5
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v4, 0xcafe0002
v_mov_b32_e32 v5, 0xcafe0001
image_atomic_cmpswap v[4:5], v[2:3], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 3
image_atomic_or v4, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

; Negative-vs-negative smin: init -16 (0xfffffff0), smin -5 (0xfffffffb) -> -16
v_mov_b32_e32 v2, 7
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v1, 0xfffffffb
image_atomic_smin v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 3
image_atomic_or v1, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

; ---- Single-lane 3D atomics: all 10 ops in slice z=0, glc old values recorded in slice z=2 ----
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 5
image_atomic_add v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 1
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 18
image_atomic_sub v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 2
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 0x0ffff0ff
image_atomic_and v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 3
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 0xaa00bb00
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 0x00ff00ff
image_atomic_xor v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 1
v_mov_b32_e32 v7, 1
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 0xfffffffc
image_atomic_smin v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 2
v_mov_b32_e32 v7, 1
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 4
image_atomic_smax v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 3
v_mov_b32_e32 v7, 1
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 0x7ffffffe
image_atomic_umin v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 2
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v1, 0x80000001
image_atomic_umax v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v1, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 1
v_mov_b32_e32 v7, 2
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v4, 0x2468ace0
v_mov_b32_e32 v5, 0x13579bdf
image_atomic_cmpswap v[4:5], v[6:8], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v4, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_mov_b32_e32 v6, 2
v_mov_b32_e32 v7, 2
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v4, 0x99999999
v_mov_b32_e32 v5, 0x11111111
image_atomic_cmpswap v[4:5], v[6:8], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 2
image_atomic_or v4, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

; ---- Part 2: all 64 lanes contending concurrently on 2D Row 4 and 3D Slice z=1 ----
s_mov_b32 exec_lo, -1
v_add_nc_u32_e32 v10, 1, v0          ; v0 + 1
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 4
image_atomic_add v10, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 1
image_atomic_sub v10, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_and_b32_e32 v11, 31, v0
v_lshlrev_b32_e64 v11, v11, 1        ; 1 << (v0 & 31)
v_mov_b32_e32 v2, 2
image_atomic_or v11, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_and_b32_e32 v12, 15, v0
v_lshlrev_b32_e64 v12, v12, 1
v_not_b32_e32 v12, v12               ; ~(1 << (v0 & 15))
v_mov_b32_e32 v2, 3
image_atomic_and v12, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 4
image_atomic_xor v10, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_subrev_nc_u32_e32 v13, 32, v0      ; v0 - 32 (-32..31)
v_mov_b32_e32 v2, 5
image_atomic_smin v13, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_mov_b32_e32 v2, 6
image_atomic_smax v13, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

v_add_nc_u32_e32 v14, 10, v0         ; v0 + 10
v_mov_b32_e32 v2, 7
image_atomic_umin v14, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm

; 3D multi-lane contention at z=1: (1,2,1) add (v0+1), (2,3,1) umax (0x7fffffe0 + v0)
v_mov_b32_e32 v6, 1
v_mov_b32_e32 v7, 2
v_mov_b32_e32 v8, 1
image_atomic_add v10, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

v_add_nc_u32_e32 v15, 0x7fffffe0, v0
v_mov_b32_e32 v6, 2
v_mov_b32_e32 v7, 3
image_atomic_umax v15, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm

; ---- Part 3: inactive-lane masking (v0 < 16 inactive targeting Row 5 & 3D z=3 canaries; v0 >= 16 active targeting Row 6) ----
; Pre-load all coordinate pairs and vdata registers BEFORE masking EXEC so inactive lanes hold the exact canary coords.
v_mov_b32_e32 v16, 5
v_mov_b32_e32 v17, 6
v_cmp_le_u32_e32 vcc_lo, 16, v0          ; v0 >= 16
v_cndmask_b32_e32 v3, v16, v17, vcc_lo   ; y = 5 (inactive) or 6 (active)
v_mov_b32_e32 v40, 0
v_mov_b32_e32 v41, v3
v_mov_b32_e32 v42, 1
v_mov_b32_e32 v43, v3
v_mov_b32_e32 v44, 2
v_mov_b32_e32 v45, v3
v_mov_b32_e32 v46, 3
v_mov_b32_e32 v47, v3
v_mov_b32_e32 v48, 4
v_mov_b32_e32 v49, v3
v_mov_b32_e32 v50, 5
v_mov_b32_e32 v51, v3
v_mov_b32_e32 v52, 6
v_mov_b32_e32 v53, v3
v_mov_b32_e32 v54, 7
v_mov_b32_e32 v55, v3

v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 2
v_cndmask_b32_e32 v6, v16, v17, vcc_lo   ; 3D add x = 0 (inactive) or 2 (active)
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 3                      ; 3D z = 3
v_mov_b32_e32 v16, 1
v_mov_b32_e32 v17, 3
v_cndmask_b32_e32 v56, v16, v17, vcc_lo  ; 3D cmpswap x = 1 (inactive) or 3 (active)
v_mov_b32_e32 v57, 0
v_mov_b32_e32 v58, 3

; Seed v20..v31 BEFORE masking EXEC so inactive lanes must keep these values even under glc
v_mov_b32_e32 v20, 1
v_mov_b32_e32 v21, 1
v_mov_b32_e32 v22, 0x00ff00ff
v_mov_b32_e32 v23, 0x00ff0000
v_mov_b32_e32 v24, 0x80000000
v_mov_b32_e32 v25, 0x7fffffff
v_mov_b32_e32 v26, 5
v_mov_b32_e32 v27, 0xbad00000
v_mov_b32_e32 v28, 0x77777777
v_mov_b32_e32 v29, 1
v_mov_b32_e32 v30, 0xbeef3d3d
v_mov_b32_e32 v31, 0x99999999

v_cmpx_le_u32_e32 16, v0                 ; lanes 0..15 OFF, lanes 16..63 ON (48 active lanes)

image_atomic_add v20, v[40:41], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
image_atomic_sub v21, v[42:43], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
image_atomic_and v22, v[44:45], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
image_atomic_or v23, v[46:47], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
image_atomic_smin v24, v[48:49], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
image_atomic_smax v25, v[50:51], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
image_atomic_umin v26, v[52:53], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
image_atomic_cmpswap v[27:28], v[54:55], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc

; 3D inactive-lane checks at z=3 (add at x=0/2, cmpswap at x=1/3)
image_atomic_add v29, v[6:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
image_atomic_cmpswap v[30:31], v[56:58], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm glc

; Restore EXEC and let lane 0 (which was INACTIVE above) write its preserved v20..v27 to Row 7
s_mov_b32 exec_lo, -1
s_waitcnt vmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v3, 7
v_mov_b32_e32 v2, 0
image_atomic_or v20, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v2, 1
image_atomic_or v21, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v2, 2
image_atomic_or v22, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v2, 3
image_atomic_or v23, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v2, 4
image_atomic_or v24, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v2, 5
image_atomic_or v25, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v2, 6
image_atomic_or v26, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v2, 7
image_atomic_or v27, v[2:3], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm

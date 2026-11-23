; Lanterns XY (stage 24): a compute shader whose result depends on both local thread IDs, with a genuine 2-D local size
; (16x8, four wave32 waves) on a 2x3 grid of groups (lanterns_xy.h). v0 = local X, v1 = local Y (TIDIG_COMP_CNT 1),
; s2 = TGID_X, s3 = TGID_Y. Each lane writes gx + 1000 gy (gx = 16 TGID_X + X, gy = 8 TGID_Y + Y) to LDS entry Y * 16 + X,
; meets the group at s_barrier, reads the entry of its point mirror in the group (15 - X, 7 - Y), and stores both values
; at (gy * 32 + gx) * 8 bytes of the output.
; s[0:1]: the table (two user SGPRs, the layout before D3D-448, which the moves below rebuild). Slot 0: a raw V# of
; the output.
; stage: cs
; wave: 32
; group: 16,8,1
; rsrc2: 0x898a
;
; RSRC2 0x8984 before D3D-448 (now 0x898a, see the moves below): two user SGPRs, TGID_X_EN and TGID_Y_EN (s2, s3),
; TIDIG_COMP_CNT 1 (v0, v1), LDS_SIZE 1 x 512 bytes.
; D3D-448: a D3D12 compute PSO gets its root arguments in the console layout (rt::rootsig::pack_stage,
; docs/d3d12_vulkan.md section 3.5): from s4 in root-signature order, a descriptor table as ONE dword (the 32-bit
; VA of its first descriptor; the heaps live below 4 GiB), a constant one dword per value; s0..s3 are the driver's.
; RSRC2 0x898a therefore counts 5 user SGPRs (4 + 1), and the thread-group ids follow at s5. These moves rebuild the
; layout the code below was written for: table s4 -> s[0:1] (high half 0); thread-group ids s5..s6 -> s2..s3.
s_mov_b32 s0, s4
s_mov_b32 s1, 0
s_mov_b32 s2, s5
s_mov_b32 s3, s6
s_load_dwordx4 s[4:7], s[0:1], 0x0
s_lshl_b32 s8, s2, 4
s_lshl_b32 s9, s3, 3
v_add_nc_u32_e32 v2, s8, v0
v_add_nc_u32_e32 v3, s9, v1
v_mul_u32_u24_e32 v4, 0x3e8, v3
v_add_nc_u32_e32 v4, v4, v2
v_lshlrev_b32_e32 v5, 4, v1
v_add_nc_u32_e32 v5, v5, v0
v_lshlrev_b32_e32 v5, 2, v5
ds_write_b32 v5, v4
s_waitcnt lgkmcnt(0)
s_barrier
v_sub_nc_u32_e32 v6, 0x1fc, v5
ds_read_b32 v7, v6
v_lshlrev_b32_e32 v8, 5, v3
v_add_nc_u32_e32 v8, v8, v2
v_lshlrev_b32_e32 v8, 3, v8
v_mov_b32_e32 v9, v4
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v10, v7
buffer_store_dwordx2 v[9:10], v8, s[4:7], 0 offen
s_endpgm

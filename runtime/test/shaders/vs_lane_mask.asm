; Vertex shader that reads its subgroup lane (SHADER-302): v_cndmask takes its mask from user SGPR s2, which the
; translator can't know holds a lane mask, so each lane takes its own bit of it (Gen.mask_src through Gen.wave_lane).
; Under --no-vertex-subgroups (MoltenVK, where a vertex function has no SIMD-group lane id) the lane comes from a
; ballot and a scan instead of SubgroupLocalInvocationId. Same full-screen triangle and param0 as vs_ids; param1.x is
; 0x202 where the lane's bit of s2 is set, else 0x101, so ps_flat (the provoking vertex's attr1.x) shows the choice.
; Run: VKH_SGPRS=0,0,0 vk_harness vs_lane_mask.spv ps_flat.spv out.raw            -> mrt0 bits = 0x101, 0x102
;      VKH_SGPRS=0,0,0xffffffff vk_harness vs_lane_mask.spv ps_flat.spv out.raw   -> mrt0 bits = 0x202, 0x203
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, 2.0, -1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v3, v4, v5, v6 done
exp param0 v1, v2, v5, v6
v_mov_b32_e32 v7, 0x101
v_mov_b32_e32 v8, 0x202
v_cndmask_b32_e64 v9, v7, v8, s2
exp param1 v9, v9, v9, v9
s_endpgm

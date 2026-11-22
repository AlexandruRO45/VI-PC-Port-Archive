; SHADER-482, the control: with no `vm` export at all there is no valid mask, so EXEC at the END of the shader
; decides (ISA guide 11.1, and the same rule tests/semantic/test_export_valid_mask.py checks in
; `test_without_a_vm_export_exec_at_the_end_decides`).
;
; x = floor(u * 64) is the pixel's column (attr0.x of vs_ids, 0..1 over the 64x64 target), parity x & 1 as in
; ps_vm_mask_late.asm. The export has no VM, and EXEC is replaced by the even columns (`v_cmpx_eq 0, x & 1` keeps
; the lanes where the bit is 0) just before `s_endpgm`:
;   even x: on at the end  -> kept
;   odd x:  off at the end -> discarded
; This is the mirror image of ps_vm_mask_late.asm, and the pair is the point: the same "off at the export, on at
; the end" pattern keeps the pixel here and discards it there, because only the export there carried VM.
;
; Run: vk_harness vs_ids.spv ps_vm_mask_end.spv out.raw
;   -> mrt0 = (1.0, 1.0, 1.0, 1.0) on every even column, the clear value (-1.0) on every odd one
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
v_and_b32_e32 v3, 1, v2
v_mov_b32_e32 v4, 1.0
exp mrt0 v4, v4, v4, v4 done
v_cmpx_eq_u32_e32 0, v3
s_endpgm

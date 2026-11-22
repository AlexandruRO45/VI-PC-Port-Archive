; SHADER-482, second case: an export to NULL carries the valid mask too, and the LAST `vm` export is the one that
; counts, not the first. ISA guide 11.2.1: an export to NULL may carry only the mask, and the final mask sent wins.
;
; x = floor(u * 64) is the pixel's column (attr0.x of vs_ids, 0..1 over the 64x64 target), parity x & 1 as in
; ps_vm_mask_late.asm. Both halves are on at the `exp mrt0 ... vm`, which latches the mask as on; then EXEC is
; replaced by the even columns (`v_cmpx_eq 0, x & 1` keeps the lanes where the bit is 0) and a `exp null ... vm`
; follows:
;   even x: on at the first vm export, on at the NULL one  -> kept
;   odd x:  on at the first vm export, off at the NULL one -> discarded
; If the first vm export were the one that counted, or if a NULL export did not carry the mask at all, every pixel
; would come back as 1.0: both are the same claim (the last vm export is the mask, whatever its target), so this
; fixture separates it from "the first one wins" and not from each of them on its own.
;
; Run: vk_harness vs_ids.spv ps_vm_mask_null.spv out.raw
;   -> mrt0 = (1.0, 1.0, 1.0, 1.0) on every even column, the clear value (-1.0) on every odd one
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
v_and_b32_e32 v3, 1, v2
v_mov_b32_e32 v4, 1.0
exp mrt0 v4, v4, v4, v4 done vm
v_cmpx_eq_u32_e32 0, v3
exp null off, off, off, off done vm
s_mov_b64 exec, -1
s_endpgm

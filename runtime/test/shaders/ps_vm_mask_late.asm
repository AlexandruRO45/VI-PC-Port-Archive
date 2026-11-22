; SHADER-482: a pixel lane that was off in EXEC at the last `exp ... vm` it executed is discarded, even if it is on
; again afterwards. RDNA2 ISA reference guide (30-November-2020), 11.1 and 11.2.1: with VM set, EXEC is the valid
; mask, and the final one sent is used. docs/design.md's EXEC row marked that "not measured on hardware"; the CPU
; interpreter checks it in tests/semantic/test_export_valid_mask.py. This is the same rule on a real fragment stage.
;
; x = floor(u * 64) is the pixel's column (u = attr0.x of vs_ids, 0..1 over the 64x64 target), and its parity x & 1
; splits the frame into the two halves of the rule:
;   odd x:  on at the export, off at the end  -> kept      (the export decides, not EXEC at the end)
;   even x: off at the export, on at the end  -> discarded (the lane came back, and it does not matter)
; The export runs with EXEC = the odd pixels, and the last instruction before `s_endpgm` sets EXEC = the even ones.
; So the two outcomes are exact inverses of each other: a translator that used EXEC at the end instead of the
; latched valid mask would return the frame inverted (odd pixels cleared, even ones 1.0), and one that dropped the
; mask altogether would return all 4096 pixels as 1.0.
;
; Run: vk_harness vs_ids.spv ps_vm_mask_late.spv out.raw
;   -> mrt0 = (1.0, 1.0, 1.0, 1.0) on every odd column, the clear value (-1.0) on every even one
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
v_and_b32_e32 v3, 1, v2
v_mov_b32_e32 v4, 1.0
v_cmpx_ne_u32_e32 0, v3
exp mrt0 v4, v4, v4, v4 done vm
v_cmpx_eq_u32_e32 0, v3
s_endpgm

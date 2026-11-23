; SHADER-482, the control: an export with no `vm` still has a lane mask of its own, EXEC at that export, so a lane
; that sent no data is not written even when it is on again at the end (docs/design.md, EXEC: the color half,
; measured; tests/semantic/test_export_valid_mask.py checks the same rule in ColorExportMask).
;
; x = floor(u * 64) is the pixel's column (attr0.x of vs_ids, 0..1 over the 64x64 target), parity x & 1 as in
; ps_vm_mask_late.asm. The export has no VM: EXEC is replaced by the even columns (`v_cmpx_eq 0, x & 1` keeps the
; lanes where the bit is 0) *before* it, and every lane is on again before `s_endpgm`:
;   even x: on at the export, on at the end  -> kept
;   odd x:  off at the export, on at the end -> discarded
; This is the mirror image of ps_vm_mask_late.asm, and the pair is the point: the same "off at the export, on at
; the end" pattern discards the pixel here and keeps it there, because only the export there carried VM. Without
; `vm` the mask is still the export's EXEC, not the one at `s_endpgm`; a translator that used EXEC at the end
; would keep every pixel and return all 4096 as 1.0.
;
; The s_waitcnt matters on hardware (docs/design.md, EXEC): an export is asynchronous, and without it the
; hardware takes the EXEC of a later moment, so the very rule these fixtures measure would read as
; "EXEC at the end decides" (ps_exp_color_mask.asm learned the same thing on RDNA2).
;
; Run: vk_harness vs_ids.spv ps_vm_mask_end.spv out.raw
;   -> mrt0 = (1.0, 1.0, 1.0, 1.0) on every even column, the clear value (-1.0) on every odd one
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
v_and_b32_e32 v3, 1, v2
v_mov_b32_e32 v4, 1.0
v_cmpx_eq_u32_e32 0, v3
exp mrt0 v4, v4, v4, v4 done
s_waitcnt expcnt(0)
s_mov_b64 exec, -1
s_endpgm

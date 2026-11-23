; docs/design.md "Pixel-shader interpolants": a 32-bit v_interp read of a 16-bit-flagged table position is the
; stored value rounded f32 -> f16 -> f32 (one OpQuantizeToF16), while an unflagged position reads the value as it
; is. The synthetic FSSM header below is the pixel-input table (tools/fssm_header.py, 8-byte records from 0x138,
; position = the v_interp attr number): records 0 and 2 carry flag bit 4 (0x10, 16-bit), 1 and 3 do not, and the
; read-method bits say positions 0/1 are read by p1/p2 (bit 1) and 2/3 by the mov (bit 0), as the code does.
; Drawn with vs_interp_f16, which puts 1.1 (0x3f8ccccd) in x of param0..param3 at every vertex, so both
; interpolation paths see exactly 1.1 before the quantize.
;   x: p1/p2 read of flagged position 0   -> f16(1.1) = 0x3f8cc000
;   y: p1/p2 read of unflagged position 1 -> 1.1      = 0x3f8ccccd
;   z: mov read of flagged position 2     -> f16(1.1) = 0x3f8cc000
;   w: mov read of unflagged position 3   -> 1.1      = 0x3f8ccccd
; Run: vk_harness vs_interp_f16.spv ps_interp_f16_table.spv out.raw
; Expected mrt0 bits: 0x3f8cc000, 0x3f8ccccd, 0x3f8cc000, 0x3f8ccccd
; fssm-at: 0x138
; fssm-data: 0x000f0006 0x12000000 0x000f0106 0x02010000 0x000f0206 0x11020000 0x000f0306 0x01030000
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr1.x
v_interp_p2_f32_e32 v3, v1, attr1.x
v_interp_mov_f32_e32 v4, p0, attr2.x
v_interp_mov_f32_e32 v5, p0, attr3.x
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm

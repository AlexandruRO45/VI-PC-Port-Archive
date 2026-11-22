; Wave64 32-bit writes to one half of VCC while VCC holds a lane mask (VCC_LO = VCC[31:0], VCC_HI = VCC[63:32]):
; each 32-bit write updates only its own 32 lanes of the mask and keeps the other half, as exec_lo / exec_hi do.
; Until 2026-11-22 a vcc_lo write set all 64 lanes and a vcc_hi write never reached the mask that v_cndmask_b32,
; s_cbranch_vccz and s_bcnt1_i32_b64 read (docs/design.md, "Wave64 handling"; CPU-checked by
; tests/semantic/test_wave64_vcc_half_write.py).
;
; One 64-lane wave (v1 = lane 0..63). Each case builds a 64-lane VCC mask with v_cmp, writes vcc_lo or vcc_hi
; with a 32-bit SOP1/SOP2 (s_mov_b32, s_and_b32, s_or_b32), and checks the resulting 64-bit VCC through:
;   - per-lane v_cndmask_b32_e32 v, 0, v(=1), vcc in both halves (lanes 0..31 and 32..63), counted with v_cmp +
;     s_bcnt1_i32_b64 / s_ff1_i32_b64 as (bcnt << 8) | ff1; and
;   - wave-wide s_cbranch_vccz / s_cbranch_vccnz (which test whether all 64 bits of VCC are 0).
;
; Cases:
; 1) VCC = lanes 16..47; s_mov_b32 vcc_lo, 0 -> clears lanes 0..31, keeps lanes 32..47 (16 lanes, lowest 32).
;    out[0] = (16 << 8) | 32 = 4128 (0x1020). (Old translator cleared all 64 lanes -> 0.)
;    s_cbranch_vccz is NOT taken because lanes 32..47 are still set in vcc_hi -> s30 bit 0 = 1.
; 2) VCC = lanes 16..47; s_mov_b32 vcc_hi, 0 -> clears lanes 32..63, keeps lanes 16..31 (16 lanes, lowest 16).
;    out[1] = (16 << 8) | 16 = 4112 (0x1010). (Old translator left lanes 32..47 set -> (32 << 8) | 16 = 8208.)
; 3) VCC = lanes 40..47 (low half 0, high half non-zero); s_mov_b32 vcc_hi, 0 -> all 64 bits of VCC are now 0.
;    s_cbranch_vccz IS taken -> s30 bit 1 = 2; v_cndmask selects 0 on every lane -> out[2] = bcnt = 0.
;    (Old translator kept lanes 40..47 in %vcc -> s_cbranch_vccz not taken and out[2] = 8.)
; 4) VCC = lanes 40..47; s_mov_b32 vcc_lo, -1 -> sets lanes 0..31, keeps lanes 40..47 (32 + 8 = 40 lanes, lowest 0).
;    out[3] = (40 << 8) | 0 = 10240 (0x2800). (Old translator overwrote lanes 32..63 with -1 too -> 64 << 8 = 16384.)
; 5) VCC = lanes 8..15; s_mov_b32 vcc_hi, -1 -> keeps lanes 8..15, sets lanes 32..63 (8 + 32 = 40 lanes, lowest 8).
;    out[4] = (40 << 8) | 8 = 10248 (0x2808). (Old translator left lanes 32..63 clear -> (8 << 8) | 8 = 2056.)
; 6) VCC = lanes 4..59; s_and_b32 vcc_lo, vcc_lo, 0 -> clears lanes 0..31, keeps lanes 32..59 (28 lanes, lowest 32).
;    out[5] = (28 << 8) | 32 = 7200 (0x1c20).
; 7) VCC = lanes 12..19; s_or_b32 vcc_hi, vcc_hi, -1 -> keeps lanes 12..19, sets lanes 32..63 (8 + 32 = 40 lanes,
;    lowest 12). out[6] = (40 << 8) | 12 = 10252 (0x280c).
; 8) out[7] = branch-flag accumulator s30:
;      bit 0 (1): case 1 s_cbranch_vccz not taken (vcc_hi still non-zero after vcc_lo = 0)
;      bit 1 (2): case 3 s_cbranch_vccz taken (vcc becomes 0 after vcc_hi = 0)
;      bit 2 (4): after s_mov_b64 vcc, 0 then s_mov_b32 vcc_hi, -1, s_cbranch_vccnz taken (vcc_hi non-zero wakes VCC)
;    -> 1 | 2 | 4 = 7.
;
; Run: vk_compute_test cs_w64_vcc_half.spv 1 dump --require-wave64
;   (or translated with --wave64-emulate, on any subgroup size)
;   -> out[0..7] = 4128 4112 0 10240 10248 7200 10252 7
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mov_b32_e32 v2, 1
s_mov_b32 s30, 0

; 1) VCC = lanes 16..47; s_mov_b32 vcc_lo, 0 -> lanes 32..47 remain
v_cmp_le_u32_e32 vcc, 16, v1
v_cmp_gt_u32_e64 s[6:7], 48, v1
s_and_b64 vcc, vcc, s[6:7]
s_mov_b32 vcc_lo, 0
v_cndmask_b32_e32 v3, 0, v2, vcc
s_cbranch_vccz 1
s_or_b32 s30, s30, 1
v_cmp_ne_u32_e32 vcc, 0, v3
s_bcnt1_i32_b64 s8, vcc
s_ff1_i32_b64 s9, vcc
s_lshl_b32 s8, s8, 8
s_or_b32 s8, s8, s9

; 2) VCC = lanes 16..47; s_mov_b32 vcc_hi, 0 -> lanes 16..31 remain
v_cmp_le_u32_e32 vcc, 16, v1
s_and_b64 vcc, vcc, s[6:7]
s_mov_b32 vcc_hi, 0
v_cndmask_b32_e32 v3, 0, v2, vcc
v_cmp_ne_u32_e32 vcc, 0, v3
s_bcnt1_i32_b64 s10, vcc
s_ff1_i32_b64 s11, vcc
s_lshl_b32 s10, s10, 8
s_or_b32 s10, s10, s11

; 3) VCC = lanes 40..47; s_mov_b32 vcc_hi, 0 -> all 64 bits 0, s_cbranch_vccz taken
v_cmp_le_u32_e32 vcc, 40, v1
s_and_b64 vcc, vcc, s[6:7]
s_mov_b32 vcc_hi, 0
v_cndmask_b32_e32 v3, 0, v2, vcc
s_or_b32 s30, s30, 2
s_cbranch_vccz 1
s_mov_b32 s30, 0
v_cmp_ne_u32_e32 vcc, 0, v3
s_bcnt1_i32_b64 s12, vcc

; 4) VCC = lanes 40..47; s_mov_b32 vcc_lo, -1 -> lanes 0..31 and 40..47 (40 lanes, lowest 0)
v_cmp_le_u32_e32 vcc, 40, v1
s_and_b64 vcc, vcc, s[6:7]
s_mov_b32 vcc_lo, -1
v_cndmask_b32_e32 v3, 0, v2, vcc
v_cmp_ne_u32_e32 vcc, 0, v3
s_bcnt1_i32_b64 s13, vcc
s_ff1_i32_b64 s14, vcc
s_lshl_b32 s13, s13, 8
s_or_b32 s13, s13, s14

; 5) VCC = lanes 8..15; s_mov_b32 vcc_hi, -1 -> lanes 8..15 and 32..63 (40 lanes, lowest 8)
v_cmp_le_u32_e32 vcc, 8, v1
v_cmp_gt_u32_e64 s[16:17], 16, v1
s_and_b64 vcc, vcc, s[16:17]
s_mov_b32 vcc_hi, -1
v_cndmask_b32_e32 v3, 0, v2, vcc
v_cmp_ne_u32_e32 vcc, 0, v3
s_bcnt1_i32_b64 s15, vcc
s_ff1_i32_b64 s18, vcc
s_lshl_b32 s15, s15, 8
s_or_b32 s15, s15, s18

; 6) VCC = lanes 4..59; s_and_b32 vcc_lo, vcc_lo, 0 -> lanes 32..59 (28 lanes, lowest 32)
v_cmp_le_u32_e32 vcc, 4, v1
v_cmp_gt_u32_e64 s[16:17], 60, v1
s_and_b64 vcc, vcc, s[16:17]
s_and_b32 vcc_lo, vcc_lo, 0
v_cndmask_b32_e32 v3, 0, v2, vcc
v_cmp_ne_u32_e32 vcc, 0, v3
s_bcnt1_i32_b64 s19, vcc
s_ff1_i32_b64 s20, vcc
s_lshl_b32 s19, s19, 8
s_or_b32 s19, s19, s20

; 7) VCC = lanes 12..19; s_or_b32 vcc_hi, vcc_hi, -1 -> lanes 12..19 and 32..63 (40 lanes, lowest 12)
v_cmp_le_u32_e32 vcc, 12, v1
v_cmp_gt_u32_e64 s[16:17], 20, v1
s_and_b64 vcc, vcc, s[16:17]
s_or_b32 vcc_hi, vcc_hi, -1
v_cndmask_b32_e32 v3, 0, v2, vcc
v_cmp_ne_u32_e32 vcc, 0, v3
s_bcnt1_i32_b64 s21, vcc
s_ff1_i32_b64 s22, vcc
s_lshl_b32 s21, s21, 8
s_or_b32 s21, s21, s22

; 8) VCC = 0 (all lanes false); s_mov_b32 vcc_hi, -1 -> s_cbranch_vccnz taken
v_cmp_gt_u32_e32 vcc, 0, v1
s_mov_b32 vcc_hi, -1
s_or_b32 s30, s30, 4
s_cbranch_vccnz 1
s_mov_b32 s30, 0

v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s19
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s21
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s30
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm

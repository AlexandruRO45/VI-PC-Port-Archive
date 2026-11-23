; SHADER-917: GPU check of the "Buffer format store rounding and 2-component DST_SEL" contract row (SHADER-364).
; Every expected word follows that row's rules, which were measured natively on gfx1036 (tools/buffer_store_probe.py).
; One thread runs; the heap starts as 0xeeeeeeee.
; - out[0]: tbuffer_store_format_x 0.3 as 16_UNORM. The exact product 0.3f * 65535 = 19660.5008 rounds to 19661
;   (0x4ccd); an f32 OpFMul gives 19660.5 and RoundEven 19660. The upper two bytes keep 0xeeee.
; - out[1], out[2]: tbuffer_store_format_xyzw (0, 0, 0, +0.5) and (0, 0, 0, -0.5) as 2_10_10_10_SNORM. w is a
;   2-bit SNORM whose scale is 1, so +-0.5 rounds away from zero to +-1: 0x40000000 and 0xc0000000.
; - out[4..11]: buffer_store_format_xy (0x11, 0x22) through a 32_32_SINT V# (format 63) with DST_SEL
;   XY01 -> (x, y), YX01 -> (y, x), XYZW -> (x, w) = (0x11, 0), YXZW -> (w, x) = (0, 0x11).
; - out[12..13]: tbuffer_store_format_xy (0x11, 0x22) through the YX01 V#: MTBUF ignores DST_SEL, (x, y).
; Run: VKC_OUT_WORDS=64 VKC_DUMP=<file> vk_compute_test cs_fmt_store_round.spv 1 dump; check.py compares all 64 words
; (out[14..63] stay 0xeeeeeeee).
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 s4, s0
s_mov_b32 s5, s1
s_mov_b32 s6, s2

; UNORM16 rounding (out[0]).
v_mov_b32_e32 v1, 0x3e99999a
v_mov_b32_e32 v6, 0x0
tbuffer_store_format_x v1, v6, s[0:3], 0 format:[BUF_FMT_16_UNORM] offen

; 2-bit SNORM ties (out[1], out[2]).
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0.5
v_mov_b32_e32 v6, 0x4
tbuffer_store_format_xyzw v[1:4], v6, s[0:3], 0 format:[BUF_FMT_2_10_10_10_SNORM] offen
v_mov_b32_e32 v4, -0.5
v_mov_b32_e32 v6, 0x8
tbuffer_store_format_xyzw v[1:4], v6, s[0:3], 0 format:[BUF_FMT_2_10_10_10_SNORM] offen

; 2-component DST_SEL (out[4..11]): word3 = 0x3103f000 | DST_SEL (X 4, Y 5, Z 6, W 7, 0 0, 1 1; 3 bits each).
v_mov_b32_e32 v1, 0x11
v_mov_b32_e32 v2, 0x22
s_mov_b32 s7, 0x3103f22c
v_mov_b32_e32 v6, 0x10
buffer_store_format_xy v[1:2], v6, s[4:7], 0 offen
s_mov_b32 s7, 0x3103f225
v_mov_b32_e32 v6, 0x18
buffer_store_format_xy v[1:2], v6, s[4:7], 0 offen
s_mov_b32 s7, 0x3103ffac
v_mov_b32_e32 v6, 0x20
buffer_store_format_xy v[1:2], v6, s[4:7], 0 offen
s_mov_b32 s7, 0x3103ffa5
v_mov_b32_e32 v6, 0x28
buffer_store_format_xy v[1:2], v6, s[4:7], 0 offen

; MTBUF ignores DST_SEL (out[12..13]).
s_mov_b32 s7, 0x3103f225
v_mov_b32_e32 v6, 0x30
tbuffer_store_format_xy v[1:2], v6, s[4:7], 0 format:[BUF_FMT_32_32_SINT] offen
s_endpgm

; Scratch swizzle: SWIZZLE_ENABLE=1, idxen=0, ADD_TID_ENABLE=1 (docs/design.md, Swizzled buffer addressing).
; index = lane; buffer_offset = (index / IS * stride + (off & ~3)) * IS + (index % IS) * 4 + (off & 3), and a raw
; multi-dword access in this sub-mode steps 4 * IS bytes per dword, not 4.
; Thread 0 seeds the words these accesses read: X 0..0x0c at heap 0x40, X 0x40..0x4c at 0x80, X 0x80..0x8c at
; 0xc0 and X 0x90..0x9c at 0xd0, with 0x5eed0000 + X (X = byte - 0x40); everything else stays 0xee.
; V# S (s[4:7]): base 0x40, SWIZZLE, stride 8, NR 16, IS 16, OOB_SELECT 3, ADD_TID.
;   lanes 0..2, dwordx2 off 0: dword 0 = lane * 4, dword 1 = lane * 4 + 4 * 16.
; V# T (s[8:11]): base 0x40, SWIZZLE, stride 16, NR 16, IS 8, OOB_SELECT 3, ADD_TID.
;   lane 9, dword off 0: (9 / 8 * 16) * 8 + (9 % 8) * 4 = X 0x84.
; V# L (s[12:15]): base 0x40, linear, stride 16, NR 16, OOB_SELECT 1, ADD_TID.
;   lane 9, dword off 0: 9 * 16 = X 0x90 (the same access without the swizzle).
; V# O (s[24:27]): base 0, linear, stride 4, NR 64, OOB_SELECT 1, ADD_TID, for the per-lane output words.
; Words, in out[0..7]:
;   0 S lane 0 dword 0: X 0x00 -> 0x5eed0000
;   1 S lane 1 dword 0: X 0x04 -> 0x5eed0004
;   2 S lane 2 dword 0: X 0x08 -> 0x5eed0008
;   3 T lane 9:         X 0x84 -> 0x5eed0084
;   4 S lane 0 dword 1: X 0x40 -> 0x5eed0040
;   5 S lane 1 dword 1: X 0x44 -> 0x5eed0044
;   6 S lane 2 dword 1: X 0x48 -> 0x5eed0048
;   7 L lane 9:         X 0x90 -> 0x5eed0090
; A linear x2 or a +4 step would read 0x5eed0004, 0x5eed0008, 0x5eed000c in words 4..6 instead.
; Run: vk_compute_test cs_buf_swizzle_scratch.spv 1 dump -> out[0..7] = 0x5eed0000 0x5eed0004 0x5eed0008
; 0x5eed0084 0x5eed0040 0x5eed0044 0x5eed0048 0x5eed0090.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x5eed0000
v_mov_b32 v2, 0x5eed0004
v_mov_b32 v3, 0x5eed0008
v_mov_b32 v4, 0x5eed000c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:64
v_mov_b32 v1, 0x5eed0040
v_mov_b32 v2, 0x5eed0044
v_mov_b32 v3, 0x5eed0048
v_mov_b32 v4, 0x5eed004c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:128
v_mov_b32 v1, 0x5eed0080
v_mov_b32 v2, 0x5eed0084
v_mov_b32 v3, 0x5eed0088
v_mov_b32 v4, 0x5eed008c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:192
v_mov_b32 v1, 0x5eed0090
v_mov_b32 v2, 0x5eed0094
v_mov_b32 v3, 0x5eed0098
v_mov_b32 v4, 0x5eed009c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:208
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0x80080000
s_mov_b32 s6, 16
s_mov_b32 s7, 0x31a16fac
s_mov_b32 s8, 0x40
s_mov_b32 s9, 0x80100000
s_mov_b32 s10, 16
s_mov_b32 s11, 0x31816fac
s_mov_b32 s12, 0x40
s_mov_b32 s13, 0x00100000
s_mov_b32 s14, 16
s_mov_b32 s15, 0x11816fac
s_mov_b32 s24, 0
s_mov_b32 s25, 0x00040000
s_mov_b32 s26, 64
s_mov_b32 s27, 0x11816fac
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 3, v0
v_mov_b32 v1, 0
buffer_load_dwordx2 v[3:4], v1, s[4:7], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dword v3, off, s[24:27], 0
buffer_store_dword v4, off, s[24:27], 0 offset:16
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 9, v0
buffer_load_dword v5, v1, s[8:11], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dword v5, off, s[0:3], 0 offset:12
buffer_load_dword v6, v1, s[12:15], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dword v6, off, s[0:3], 0 offset:28
s_endpgm

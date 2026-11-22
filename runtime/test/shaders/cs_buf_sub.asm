; Sub-dword MUBUF loads and GFX10_FORMAT_8_UINT: buffer_load_ubyte / _sbyte / _ushort / _sshort read 1 or 2
; bytes with the V# OOB_SELECT end check on the element (docs/design.md, "Sub-dword MUBUF loads"); the format loads
; read a byte as an unsigned integer. CPU-checked by tests/semantic/test_buffer_load_sub.py; this is the GPU half.
; Thread 0 seeds heap bytes 8..15 (region offset X = byte 8 + X) with 0x7f, 0x80, 0xff, 0x01, 0xab, 0xcd, 0x00, 0xfe
; and loads through V#s it builds itself (base 8, OOB_SELECT 3, word3 0x30005fac):
;   buffer_load_ubyte   byte 0 -> 0x7f                                buffer_load_sbyte  byte 1 -> 0xffffff80
;   buffer_load_ushort  byte 3 -> 0xab01 (1 byte in each dword)       buffer_load_sshort byte 3 -> 0xffffab01
;   buffer_load_ushort  byte 6 -> 0xfe00                              buffer_load_ubyte  byte 7 -> 0xfe
; with NUM_RECORDS 4 the end check is 1 or 2 bytes, not the dword: ubyte byte 3 (3+1 <= 4) -> 0x01, byte 4 -> 0,
; ushort byte 2 (2+2 <= 4) -> 0x01ff, byte 3 (3+2 > 4) -> 0; with NUM_RECORDS 5 the crossing ushort at 3 -> 0xab01.
; OOB_SELECT 0 (stride 4, NUM_RECORDS 2): ushort idx 1 offset 2 (2+2 <= 4) -> 0xfe00, idx 0 offset 3 (5 > 4) -> 0,
; idx 1 with NUM_RECORDS 1 -> 0.
; tbuffer_load_format_x format:[BUF_FMT_8_UINT] with a non-zero V# format: bytes 0..3 -> 0x7f, 0x80, 0xff, 0x01,
; byte 4 with NUM_RECORDS 4 -> 0. buffer_load_format_xyzw with V# NUM_FORMAT 5 (GFX10_FORMAT_8_UINT) and
; DST_SEL (X, 0, 1, 1) at byte 2 -> 0xff, 0, 1, 1 (integers, not floats).
; Results go to out words 16..40; words 0..15, 22..23 and 41..63 stay 0xeeeeeeee.
; Run: VKC_OUT_WORDS=64 VKC_DUMP=<file> vk_compute_test cs_buf_sub.spv 1 dump; check.py's buf_sub_test compares all 64.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v2, 0x01ff807f
v_mov_b32 v3, 0xfe00cdab
buffer_store_dwordx2 v[2:3], off, s[0:3], 0 offset:8
s_waitcnt vmcnt(0)
s_mov_b32 s4, 8
s_mov_b32 s5, 0
s_mov_b32 s6, 8
s_mov_b32 s7, 0x30005fac
v_mov_b32 v1, 0
buffer_load_ubyte v4, v1, s[4:7], 0 offen
v_mov_b32 v1, 1
buffer_load_sbyte v5, v1, s[4:7], 0 offen
v_mov_b32 v1, 3
buffer_load_ushort v6, v1, s[4:7], 0 offen
v_mov_b32 v1, 3
buffer_load_sshort v7, v1, s[4:7], 0 offen
v_mov_b32 v1, 6
buffer_load_ushort v8, v1, s[4:7], 0 offen
v_mov_b32 v1, 7
buffer_load_ubyte v9, v1, s[4:7], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[4:7], off, s[0:3], 0 offset:64
buffer_store_dwordx2 v[8:9], off, s[0:3], 0 offset:80
s_mov_b32 s6, 4
v_mov_b32 v1, 3
buffer_load_ubyte v10, v1, s[4:7], 0 offen
v_mov_b32 v1, 4
buffer_load_ubyte v11, v1, s[4:7], 0 offen
v_mov_b32 v1, 2
buffer_load_ushort v12, v1, s[4:7], 0 offen
v_mov_b32 v1, 3
buffer_load_sshort v13, v1, s[4:7], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[10:13], off, s[0:3], 0 offset:96
s_mov_b32 s6, 5
v_mov_b32 v1, 3
buffer_load_ushort v14, v1, s[4:7], 0 offen
s_mov_b32 s5, 0x40000
s_mov_b32 s6, 2
s_mov_b32 s7, 0x5fac
v_mov_b32 v1, 1
v_mov_b32 v2, 2
buffer_load_ushort v15, v[1:2], s[4:7], 0 idxen offen
v_mov_b32 v1, 0
v_mov_b32 v2, 3
buffer_load_ushort v16, v[1:2], s[4:7], 0 idxen offen
s_mov_b32 s6, 1
v_mov_b32 v1, 1
v_mov_b32 v2, 0
buffer_load_ushort v17, v[1:2], s[4:7], 0 idxen offen
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[14:17], off, s[0:3], 0 offset:112
s_mov_b32 s5, 0
s_mov_b32 s6, 4
s_mov_b32 s7, 0x30014244
v_mov_b32 v1, 0
tbuffer_load_format_x v18, v1, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
v_mov_b32 v1, 1
tbuffer_load_format_x v19, v1, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
v_mov_b32 v1, 2
tbuffer_load_format_x v20, v1, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
v_mov_b32 v1, 3
tbuffer_load_format_x v21, v1, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[18:21], off, s[0:3], 0 offset:128
v_mov_b32 v1, 4
tbuffer_load_format_x v22, v1, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
s_mov_b32 s7, 0x30005244
v_mov_b32 v1, 2
buffer_load_format_xyzw v[23:26], v1, s[4:7], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dword v22, off, s[0:3], 0 offset:144
buffer_store_dwordx4 v[23:26], off, s[0:3], 0 offset:148
s_endpgm

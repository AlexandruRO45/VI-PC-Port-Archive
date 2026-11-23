; Wave64 compute, --wave64-emulate. Lanes with (5 * lane) & 15 > 12 enter a folded s_cbranch_execz region,
; where s_cmp + s_cbranch_scc0 is taken straight to the join, so the two adds must not run.
; Expected buffer dword per thread at 64 and 128 threads: 1.
; mattr: +wavefrontsize64,-wavefrontsize32
	v_mul_lo_u32 v3, v0, 5                  ; encoding: [0x03,0x00,0x69,0xd5,0x00,0x0b,0x01,0x00]
	v_and_b32_e32 v3, 15, v3                ; encoding: [0x8f,0x06,0x06,0x36]
	v_mov_b32_e32 v4, 1                     ; encoding: [0x81,0x02,0x08,0x7e]
	s_mov_b32 s11, 5                        ; encoding: [0x85,0x03,0x8b,0xbe]
	v_cmp_lt_u32_e32 vcc, 12, v3            ; encoding: [0x8c,0x06,0x82,0x7d]
	s_and_saveexec_b64 s[8:9], vcc          ; encoding: [0x6a,0x24,0x88,0xbe]
	s_cbranch_execz 5                       ; encoding: [0x05,0x00,0x88,0xbf]
	s_cmp_ge_u32 s11, 33                    ; encoding: [0x0b,0xa1,0x09,0xbf]
	s_cbranch_scc0 3                        ; encoding: [0x03,0x00,0x84,0xbf]
	v_add_nc_u32_e32 v4, 21, v4             ; encoding: [0x95,0x08,0x08,0x4a]
	v_add_nc_u32_e32 v4, 0x64, v4           ; encoding: [0xff,0x08,0x08,0x4a,0x64,0x00,0x00,0x00]
	s_or_b64 exec, exec, s[8:9]             ; encoding: [0x7e,0x08,0xfe,0x88]
	v_lshlrev_b32_e32 v9, 2, v0             ; encoding: [0x82,0x00,0x12,0x34]
	buffer_store_dword v4, v9, s[0:3], 0 offen ; encoding: [0x00,0x10,0x70,0xe0,0x09,0x04,0x00,0x80]
	s_endpgm                                ; encoding: [0x00,0x00,0x81,0xbf]

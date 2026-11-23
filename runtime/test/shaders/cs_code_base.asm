; group: 1,1,1
; Code-embedded data (docs/design.md): a constant table inside the code range, read through s_getpc_b64 +
; s_add_u32/s_addc_u32 + s_load. s_getpc_b64 gives the code base (push-constant dwords 34-35, where the runtime puts
; this shader's FSSM blob) plus the offset of the next instruction; vk_compute_test sets the base from
; VKC_CODE_BASE=0x800. The one thread stores four words at Xbox VA 0x880 through the output V# and reads them back
; through the code base, so a load lands on the table only if the translator adds the next instruction's offset as the
; assembler laid it out: the site at 0x38 adds 0x44 to 0x83c, and the one at 0x6c adds 0x10 to 0x870. Both must reach
; 0x880. The corpus loads constant tables four and eight dwords at a time, hence s_load_dwordx4; the second site reads
; one word at table offset 12. The add's carry is 0 at these addresses, so s_addc_u32 only exercises the instruction.
; Run: VKC_CODE_BASE=0x800 VKC_OUT_WORDS=552 vk_compute_test cs_code_base.spv 1 dump
; out[0..7] = CAFE0F00 1234ABCD 5EED0001 9E3779B9 9E3779B9 EEEEEEEE EEEEEEEE EEEEEEEE (hex)
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0xcafe0f00
v_mov_b32_e32 v3, 0x1234abcd
v_mov_b32_e32 v4, 0x5eed0001
v_mov_b32_e32 v5, 0x9e3779b9
v_mov_b32_e32 v6, 0x880
buffer_store_dwordx4 v[2:5], v6, s[0:3], 0 offen
s_waitcnt vmcnt(0)
s_getpc_b64 s[10:11]
s_add_u32 s10, s10, 0x44
s_addc_u32 s11, s11, 0
s_load_dwordx4 s[12:15], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, s12
v_mov_b32_e32 v3, s13
v_mov_b32_e32 v4, s14
v_mov_b32_e32 v5, s15
buffer_store_dwordx4 v[2:5], v1, s[0:3], 0 offen
s_getpc_b64 s[10:11]
s_add_u32 s10, s10, 0x10
s_addc_u32 s11, s11, 0
s_load_dword s12, s[10:11], 0xc
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, s12
v_mov_b32_e32 v3, 0x10
buffer_store_dword v2, v3, s[0:3], 0 offen
s_endpgm

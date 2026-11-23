; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; group: 1,1,1
; mattr: +wavefrontsize32,-wavefrontsize64,+dot1-insts
; The initial heap word (0xeeeeeeee) supplies runtime operands for the gfx1036 cases in
; tests/semantic/test_alu_dot4_i8.py (MEASURED_NEG). S0 = 0x04030201, S1 = 0x08070605.
; Their signed-byte dot is 70. Adding 100 gives 0xaa before modifiers.
; Words 0..3 flip bit 15 or 31 of one packed input, subtracting 128 times the partner byte
; (6, 8, 2, 4 respectively). Word 4 flips bit 31 of the positive addend.
; Words 5 and 6 clear bit 31 on opposite-sign addends: both must give 0xaa.
; Word 7 sets all six bits on the negative addend: clear its sign bit, then flip it.
; The result is 1*5 + (-126)*(-122) + 3*7 + (-124)*(-120) + 0x80000064.
; Run: vk_compute_test cs_dot4_i8_neg.spv 1 dump
; out[0..7] (hex): fffffdaa fffffcaa ffffffaa fffffeaa 800000aa 000000aa 000000aa 800076aa
buffer_load_dword v4, v0, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_xor_b32_e32 v1, 0xeaedecef, v4
v_xor_b32_e32 v2, 0xe6e9e8eb, v4
v_xor_b32_e32 v3, 0xeeeeee8a, v4
v_xor_b32_e32 v5, 0x80000000, v3
v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[1,0,0]
v_dot4_i32_i8 v9, v1, v2, v3 neg_hi:[1,0,0]
v_dot4_i32_i8 v10, v1, v2, v3 neg_lo:[0,1,0]
v_dot4_i32_i8 v11, v1, v2, v3 neg_hi:[0,1,0]
v_dot4_i32_i8 v12, v1, v2, v3 neg_lo:[0,0,1]
v_dot4_i32_i8 v13, v1, v2, v5 neg_hi:[0,0,1]
v_dot4_i32_i8 v14, v1, v2, v3 neg_hi:[0,0,1]
v_dot4_i32_i8 v15, v1, v2, v5 neg_lo:[1,1,1] neg_hi:[1,1,1]
buffer_store_dwordx4 v[8:11], v0, s[0:3], 0 offen
buffer_store_dwordx4 v[12:15], v0, s[0:3], 0 offen offset:16
s_endpgm

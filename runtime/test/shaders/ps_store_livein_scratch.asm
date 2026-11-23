; vs_persp supplies the triangle. Expect two copies of the perspective-correct
; vertex 1/2 weights at each covered pixel centre, and the clear value outside.
; The taken branch skips s_endpgm and an eight-byte scratch store. Its address is a
; source: classifying it as a destination suppresses the fragment prologue's live-in.
s_mov_b32 s4, 1
s_cmp_eq_u32 s4, 1
s_cbranch_scc1 3
s_endpgm
scratch_store_dword v0, v4, off
exp mrt0 v0, v1, v0, v1 done vm
s_endpgm

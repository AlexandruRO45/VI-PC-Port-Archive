; Contention test for global_atomic_swap_x2 with glc: every invocation of every workgroup swaps its own pair into one
; shared 64-bit element and stores the pair glc returned (TEST-561). Workgroup id in s4 (the default header: 4 user
; SGPRs, then TGID_X), thread id in v0, so invocation i = s4 * 32 + v0.
;   replacement pair of invocation i: low = 0x51000000 | i, high = ~low (the halves of a pair belong together)
;   the element: bytes 0x4000..0x4007 of the heap, which starts as the 0xEE fill, so the first pair anyone gets back
;                is (0xEEEEEEEE, 0xEEEEEEEE)
;   out[2 * i], out[2 * i + 1]: the pair invocation i got back
; With 64 groups (2048 invocations) the returned pairs fill words 0..4095 and the element is words 4096..4097.
; What must hold whatever the order: each dword is one 32-bit atomic exchange, so the 2048 returned low dwords plus the
; element's final low dword are exactly the fill plus the 2048 replacement lows, and going back from the final value
; each invocation's returned low names the invocation before it, through all 2048 to the fill. The same for the high
; dwords, in an order of their own.
; What may not hold: tools/rdna2spirv.py translates the swap as one atomic per dword, so two invocations swapping at
; once can tear a pair (A's low with B's high). swap_x2_contend.py counts those; hardware swaps the element whole.
; Run: VKC_OUT_WORDS=4104 VKC_DUMP=<file> vk_compute_test cs_w32_swap_x2_contend.spv 64 dump, then check the file
;      with swap_x2_contend.py (the first eight words depend on the order the invocations ran in).
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_lshl_add_u32 v1, s4, 5, v0
v_or_b32_e32 v30, 0x51000000, v1
v_not_b32_e32 v31, v30
v_mov_b32_e32 v10, 0x4000
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v36, 0
v_mov_b32_e32 v37, 0
global_atomic_swap_x2 v[36:37], v[10:11], v[30:31], off glc
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v5, 3, v1
buffer_store_dwordx2 v[36:37], v5, s[0:3], 0 offen
s_endpgm

; coverage: source cpu_aot_avx2_shifts
; Source-owned differential fixture for AVX2 broadcasts and variable per-lane shifts:
; VPBROADCASTB, VPBROADCASTW, VPBROADCASTD, VPBROADCASTQ,
; VPSLLVD, VPSRLVD, VPSRAVD, VPSLLVQ, VPSRLVQ.
.code

bcast_b256 PROC FRAME
    .endprolog
    vpbroadcastb ymm0, byte ptr [rcx]
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_b256 ENDP

bcast_w256 PROC FRAME
    .endprolog
    vpbroadcastw ymm0, word ptr [rcx]
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_w256 ENDP

bcast_d256 PROC FRAME
    .endprolog
    vpbroadcastd ymm0, dword ptr [rcx]
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_d256 ENDP

bcast_q256 PROC FRAME
    .endprolog
    vpbroadcastq ymm0, qword ptr [rcx]
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_q256 ENDP

; Register-source 256-bit broadcasts.
bcast_b256_reg PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpbroadcastb ymm0, xmm1
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_b256_reg ENDP

bcast_w256_reg PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpbroadcastw ymm0, xmm1
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_w256_reg ENDP

bcast_d256_reg PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpbroadcastd ymm0, xmm1
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_d256_reg ENDP

bcast_q256_reg PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpbroadcastq ymm0, xmm1
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_q256_reg ENDP

bcast_d256_alias PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpbroadcastd ymm0, xmm0
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_d256_alias ENDP

; Memory-source 128-bit broadcasts: pre-load YMM0 to verify upper 128 bits are zeroed.
bcast_b128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rdx]
    vpbroadcastb xmm0, byte ptr [rcx]
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_b128 ENDP

bcast_w128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rdx]
    vpbroadcastw xmm0, word ptr [rcx]
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_w128 ENDP

bcast_d128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rdx]
    vpbroadcastd xmm0, dword ptr [rcx]
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_d128 ENDP

bcast_q128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rdx]
    vpbroadcastq xmm0, qword ptr [rcx]
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_q128 ENDP

; Register-source 128-bit broadcasts: verify upper 128 bits of YMM0 are zeroed.
bcast_b128_reg PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rdx]
    vmovups ymm1, ymmword ptr [rcx]
    vpbroadcastb xmm0, xmm1
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_b128_reg ENDP

bcast_w128_reg PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rdx]
    vmovups ymm1, ymmword ptr [rcx]
    vpbroadcastw xmm0, xmm1
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_w128_reg ENDP

bcast_d128_reg PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rdx]
    vmovups ymm1, ymmword ptr [rcx]
    vpbroadcastd xmm0, xmm1
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_d128_reg ENDP

bcast_q128_reg PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rdx]
    vmovups ymm1, ymmword ptr [rcx]
    vpbroadcastq xmm0, xmm1
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_q128_reg ENDP

bcast_b128_alias PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpbroadcastb xmm0, xmm0
    vmovups ymmword ptr [rdx], ymm0
    vzeroupper
    ret
bcast_b128_alias ENDP

sllvd256 PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsllvd ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvd256 ENDP

srlvd256 PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsrlvd ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvd256 ENDP

sravd256 PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsravd ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sravd256 ENDP

sllvq256 PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsllvq ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvq256 ENDP

srlvq256 PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsrlvq ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvq256 ENDP

; Memory op2 256-bit variable shifts.
sllvd256_mem PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsllvd ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvd256_mem ENDP

srlvd256_mem PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsrlvd ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvd256_mem ENDP

sravd256_mem PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsravd ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sravd256_mem ENDP

sllvq256_mem PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsllvq ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvq256_mem ENDP

srlvq256_mem PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsrlvq ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvq256_mem ENDP

; Register op2 128-bit variable shifts: pre-load YMM0 to verify upper 128 bits are zeroed.
sllvd128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsllvd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvd128 ENDP

srlvd128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsrlvd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvd128 ENDP

sravd128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsravd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sravd128 ENDP

sllvq128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsllvq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvq128 ENDP

srlvq128 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsrlvq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvq128 ENDP

; Memory op2 128-bit variable shifts: pre-load YMM0 to verify upper 128 bits are zeroed.
sllvd128_mem PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vpsllvd xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvd128_mem ENDP

srlvd128_mem PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vpsrlvd xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvd128_mem ENDP

sravd128_mem PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vpsravd xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sravd128_mem ENDP

sllvq128_mem PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vpsllvq xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvq128_mem ENDP

srlvq128_mem PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vpsrlvq xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvq128_mem ENDP

; Register aliasing: dst == src2, dst == src1, and dst == src1 == src2.
sllvd256_alias_src2 PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vpsllvd ymm0, ymm1, ymm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvd256_alias_src2 ENDP

srlvd256_alias_src1 PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsrlvd ymm0, ymm0, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
srlvd256_alias_src1 ENDP

sravd256_alias_all PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpsravd ymm0, ymm0, ymm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sravd256_alias_all ENDP

sllvd256_alias_all PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpsllvd ymm0, ymm0, ymm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvd256_alias_all ENDP

sllvd128_alias_src2 PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vpsllvd xmm0, xmm1, xmm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvd128_alias_src2 ENDP

sllvq256_alias_src2 PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vpsllvq ymm0, ymm1, ymm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
sllvq256_alias_src2 ENDP

END

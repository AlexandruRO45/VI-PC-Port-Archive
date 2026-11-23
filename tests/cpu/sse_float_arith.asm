; coverage: source cpu_aot_sse_float_arith
; Source-owned differential fixture for 128-bit baseline SSE packed single-precision
; floating-point arithmetic instructions (addps, subps, mulps, divps).
.code

addps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    addps xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
addps_op ENDP

subps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    subps xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
subps_op ENDP

mulps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mulps xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
mulps_op ENDP

divps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    divps xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
divps_op ENDP

addps_self_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    addps xmm0, xmm0
    movdqu xmmword ptr [r8], xmm0
    ret
addps_self_op ENDP

subps_self_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    subps xmm0, xmm0
    movdqu xmmword ptr [r8], xmm0
    ret
subps_self_op ENDP

mulps_self_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    mulps xmm0, xmm0
    movdqu xmmword ptr [r8], xmm0
    ret
mulps_self_op ENDP

divps_self_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    divps xmm0, xmm0
    movdqu xmmword ptr [r8], xmm0
    ret
divps_self_op ENDP

addps_xmm1_dst_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    addps xmm1, xmm2
    movdqu xmmword ptr [r8], xmm1
    ret
addps_xmm1_dst_op ENDP

addps_xmm2_dst_op PROC FRAME
    .endprolog
    movdqu xmm2, xmmword ptr [rcx]
    movdqu xmm0, xmmword ptr [rdx]
    addps xmm2, xmm0
    movdqu xmmword ptr [r8], xmm2
    ret
addps_xmm2_dst_op ENDP

END

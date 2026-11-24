; coverage: source cpu_aot_float_shuffles
; Differential fixture for 51 vector shuffle, unpacking, and move instructions,
; testing full 256-bit vector state (legacy high-half preservation vs VEX.128 zeroing).
.code

movmskps_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    movmskps eax, xmm0
    mov dword ptr [r8], eax
    ret
movmskps_op ENDP

movmskpd_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    movmskpd eax, xmm0
    mov dword ptr [r8], eax
    ret
movmskpd_op ENDP

pmovmskb_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    pmovmskb eax, xmm0
    mov dword ptr [r8], eax
    ret
pmovmskb_op ENDP

vmovmskps_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovmskps eax, xmm0
    mov dword ptr [r8], eax
    ret
vmovmskps_op ENDP

vmovmskpd_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovmskpd eax, xmm0
    mov dword ptr [r8], eax
    ret
vmovmskpd_op ENDP

vpmovmskb_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vpmovmskb eax, xmm0
    mov dword ptr [r8], eax
    ret
vpmovmskb_op ENDP

shufps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    shufps xmm0, xmm1, 044h
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
shufps_op ENDP

unpckhps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    unpckhps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
unpckhps_op ENDP

unpcklps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    unpcklps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
unpcklps_op ENDP

movhlps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    movhlps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
movhlps_op ENDP

movlhps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    movlhps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
movlhps_op ENDP

movups_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    movups xmm0, xmmword ptr [rdx]
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
movups_op ENDP

rcpps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    rcpps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
rcpps_op ENDP

rcpss_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    rcpss xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
rcpss_op ENDP

rsqrtps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    rsqrtps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
rsqrtps_op ENDP

rsqrtss_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    rsqrtss xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
rsqrtss_op ENDP

shufpd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    shufpd xmm0, xmm1, 1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
shufpd_op ENDP

unpckhpd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    unpckhpd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
unpckhpd_op ENDP

unpcklpd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    unpcklpd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
unpcklpd_op ENDP

movupd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    movupd xmm0, xmmword ptr [rdx]
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
movupd_op ENDP

movapd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    movapd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
movapd_op ENDP

movdqa_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    movdqa xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
movdqa_op ENDP

pshufd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    pshufd xmm0, xmm1, 044h
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
pshufd_op ENDP

pshufhw_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    pshufhw xmm0, xmm1, 044h
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
pshufhw_op ENDP

pshuflw_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    pshuflw xmm0, xmm1, 044h
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
pshuflw_op ENDP

pslldq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    pslldq xmm0, 2
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
pslldq_op ENDP

psrldq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    psrldq xmm0, 2
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
psrldq_op ENDP

punpckhqdq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    punpckhqdq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
punpckhqdq_op ENDP

punpcklqdq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    punpcklqdq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
punpcklqdq_op ENDP

vshufps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vshufps xmm0, xmm0, xmm1, 044h
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vshufps_op ENDP

vunpckhps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vunpckhps xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vunpckhps_op ENDP

vunpcklps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vunpcklps xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vunpcklps_op ENDP

vshufpd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vshufpd xmm0, xmm0, xmm1, 1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vshufpd_op ENDP

vunpckhpd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vunpckhpd xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vunpckhpd_op ENDP

vunpcklpd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vunpcklpd xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vunpcklpd_op ENDP

vpshufd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpshufd xmm0, xmm1, 044h
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshufd_op ENDP

vpshufhw_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpshufhw xmm0, xmm1, 044h
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshufhw_op ENDP

vpshuflw_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpshuflw xmm0, xmm1, 044h
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshuflw_op ENDP

vmovhlps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vmovhlps xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovhlps_op ENDP

vmovlhps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vmovlhps xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovlhps_op ENDP

vmovaps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovaps xmm0, xmm0
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovaps_op ENDP

vmovapd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vmovapd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovapd_op ENDP

vmovdqa_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vmovdqa xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovdqa_op ENDP

vrcpps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vrcpps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vrcpps_op ENDP

vrcpss_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vrcpss xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vrcpss_op ENDP

vrsqrtps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vrsqrtps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vrsqrtps_op ENDP

vrsqrtss_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vrsqrtss xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vrsqrtss_op ENDP

vpslldq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpslldq xmm0, xmm0, 2
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpslldq_op ENDP

vpsrldq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpsrldq xmm0, xmm0, 2
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsrldq_op ENDP

vpunpckhqdq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpunpckhqdq xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpckhqdq_op ENDP

vpunpcklqdq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpunpcklqdq xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpcklqdq_op ENDP

END

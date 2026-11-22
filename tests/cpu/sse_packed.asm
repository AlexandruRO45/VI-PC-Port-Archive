; coverage: source cpu_aot_sse_packed
; Source-owned differential fixture for SSE packed integer instructions:
; PAVGB, PAVGW, PEXTRW, PINSRW, PMAXSW, PMAXUB, PMINSW, PMINUB, PMOVMSKB, PMULHUW, PSADBW.
.code

test_pavgb PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pavgb xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pavgb ENDP

test_pavgw PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pavgw xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pavgw ENDP

test_pmaxsw PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pmaxsw xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pmaxsw ENDP

test_pmaxub PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pmaxub xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pmaxub ENDP

test_pminsw PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pminsw xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pminsw ENDP

test_pminub PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pminub xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pminub ENDP

test_pmulhuw PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pmulhuw xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pmulhuw ENDP

test_psadbw PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    psadbw xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_psadbw ENDP

test_pmovmskb PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pmovmskb eax, xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pmovmskb ENDP

test_pextrw_0 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pextrw_0 ENDP

test_pextrw_1 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 1
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pextrw_1 ENDP

test_pextrw_2 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 2
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pextrw_2 ENDP

test_pextrw_3 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 3
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pextrw_3 ENDP

test_pextrw_4 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 4
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pextrw_4 ENDP

test_pextrw_5 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 5
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pextrw_5 ENDP

test_pextrw_6 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 6
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pextrw_6 ENDP

test_pextrw_7 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 7
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [rdx], r10
    ret
test_pextrw_7 ENDP

test_pinsrw_0 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pinsrw xmm0, edx, 0
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pinsrw_0 ENDP

test_pinsrw_1 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pinsrw xmm0, edx, 1
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pinsrw_1 ENDP

test_pinsrw_2 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pinsrw xmm0, edx, 2
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pinsrw_2 ENDP

test_pinsrw_3 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pinsrw xmm0, edx, 3
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pinsrw_3 ENDP

test_pinsrw_4 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pinsrw xmm0, edx, 4
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pinsrw_4 ENDP

test_pinsrw_5 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pinsrw xmm0, edx, 5
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pinsrw_5 ENDP

test_pinsrw_6 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pinsrw xmm0, edx, 6
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pinsrw_6 ENDP

test_pinsrw_7 PROC FRAME
    .endprolog
    mov r10d, 0
    cmp r10d, 1
    movdqu xmm0, xmmword ptr [rcx]
    pinsrw xmm0, edx, 7
    movdqu xmmword ptr [r8], xmm0
    mov r10d, 0
    jae short @F
    mov r10d, 2
@@:
    jne short @F
    or r10d, 1
@@:
    mov qword ptr [r9], r10
    mov rax, 0
    ret
test_pinsrw_7 ENDP

END

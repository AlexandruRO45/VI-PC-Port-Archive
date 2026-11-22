; Source-owned differential fixture for BMI1 bit-manipulation instructions
; (ANDN, BEXTR, BLSI, BLSMSK, BLSR, TZCNT) in 32-bit and 64-bit forms.
; Each export writes (CF << 1) | ZF to the output pointer and returns RAX. ANDN and BEXTR
; follow a compare that sets CF, and BLSMSK one that sets ZF, so the flags they clear are
; observed rather than inherited.
.code

andn32 PROC FRAME
    .endprolog
    mov rax, -1
    xor r9d, r9d
    cmp r9, 1                ; CF = 1, which the next instruction must clear
    andn eax, ecx, edx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [r8], r9
    ret
andn32 ENDP

andn64 PROC FRAME
    .endprolog
    xor r9d, r9d
    cmp r9, 1                ; CF = 1, which the next instruction must clear
    andn rax, rcx, rdx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [r8], r9
    ret
andn64 ENDP

bextr32 PROC FRAME
    .endprolog
    mov rax, -1
    xor r9d, r9d
    cmp r9, 1                ; CF = 1, which the next instruction must clear
    bextr eax, ecx, edx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [r8], r9
    ret
bextr32 ENDP

bextr64 PROC FRAME
    .endprolog
    xor r9d, r9d
    cmp r9, 1                ; CF = 1, which the next instruction must clear
    bextr rax, rcx, rdx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [r8], r9
    ret
bextr64 ENDP

blsi32 PROC FRAME
    .endprolog
    mov rax, -1
    blsi eax, ecx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [rdx], r9
    ret
blsi32 ENDP

blsi64 PROC FRAME
    .endprolog
    blsi rax, rcx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [rdx], r9
    ret
blsi64 ENDP

blsmsk32 PROC FRAME
    .endprolog
    mov rax, -1
    cmp ecx, ecx             ; ZF = 1, which BLSMSK must clear
    blsmsk eax, ecx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [rdx], r9
    ret
blsmsk32 ENDP

blsmsk64 PROC FRAME
    .endprolog
    cmp ecx, ecx             ; ZF = 1, which BLSMSK must clear
    blsmsk rax, rcx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [rdx], r9
    ret
blsmsk64 ENDP

blsr32 PROC FRAME
    .endprolog
    mov rax, -1
    blsr eax, ecx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [rdx], r9
    ret
blsr32 ENDP

blsr64 PROC FRAME
    .endprolog
    blsr rax, rcx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [rdx], r9
    ret
blsr64 ENDP

tzcnt32 PROC FRAME
    .endprolog
    mov rax, -1
    tzcnt eax, ecx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [rdx], r9
    ret
tzcnt32 ENDP

tzcnt64 PROC FRAME
    .endprolog
    tzcnt rax, rcx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [rdx], r9
    ret
tzcnt64 ENDP

END

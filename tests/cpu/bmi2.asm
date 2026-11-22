; Source-owned differential fixture for BMI2 bit-manipulation instructions
; (BZHI, MULX, PDEP, PEXT, RORX, SARX, SHLX, SHRX) in 32-bit and 64-bit forms.
.code

bzhi32 PROC FRAME
    .endprolog
    mov rax, -1
    xor r9d, r9d
    cmp r9, 1
    bzhi eax, ecx, edx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [r8], r9
    ret
bzhi32 ENDP

bzhi64 PROC FRAME
    .endprolog
    xor r9d, r9d
    cmp r9, 1
    bzhi rax, rcx, rdx
    mov r9d, 0
    jae short @F
    mov r9d, 2
@@:
    jne short @F
    or r9d, 1
@@:
    mov qword ptr [r8], r9
    ret
bzhi64 ENDP

mulx32 PROC FRAME
    .endprolog
    mov eax, edx
    mov edx, ecx
    mulx r9d, r10d, eax
    mov dword ptr [r8], r9d
    mov eax, r10d
    ret
mulx32 ENDP

mulx64 PROC FRAME
    .endprolog
    mov rax, rdx
    mov rdx, rcx
    mulx r9, r10, rax
    mov qword ptr [r8], r9
    mov rax, r10
    ret
mulx64 ENDP

mulx32_same PROC FRAME
    .endprolog
    mov eax, edx
    mov edx, ecx
    mulx eax, eax, eax
    ret
mulx32_same ENDP

mulx64_same PROC FRAME
    .endprolog
    mov rax, rdx
    mov rdx, rcx
    mulx rax, rax, rax
    ret
mulx64_same ENDP

pdep32 PROC FRAME
    .endprolog
    mov rax, -1
    pdep eax, ecx, edx
    ret
pdep32 ENDP

pdep64 PROC FRAME
    .endprolog
    pdep rax, rcx, rdx
    ret
pdep64 ENDP

pext32 PROC FRAME
    .endprolog
    mov rax, -1
    pext eax, ecx, edx
    ret
pext32 ENDP

pext64 PROC FRAME
    .endprolog
    pext rax, rcx, rdx
    ret
pext64 ENDP

rorx32_0 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 0
    ret
rorx32_0 ENDP

rorx32_1 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 1
    ret
rorx32_1 ENDP

rorx32_7 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 7
    ret
rorx32_7 ENDP

rorx32_15 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 15
    ret
rorx32_15 ENDP

rorx32_31 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 31
    ret
rorx32_31 ENDP

rorx32_32 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 32
    ret
rorx32_32 ENDP

rorx32_33 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 33
    ret
rorx32_33 ENDP

rorx32_255 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 255
    ret
rorx32_255 ENDP

rorx64_0 PROC FRAME
    .endprolog
    rorx rax, rcx, 0
    ret
rorx64_0 ENDP

rorx64_1 PROC FRAME
    .endprolog
    rorx rax, rcx, 1
    ret
rorx64_1 ENDP

rorx64_7 PROC FRAME
    .endprolog
    rorx rax, rcx, 7
    ret
rorx64_7 ENDP

rorx64_31 PROC FRAME
    .endprolog
    rorx rax, rcx, 31
    ret
rorx64_31 ENDP

rorx64_63 PROC FRAME
    .endprolog
    rorx rax, rcx, 63
    ret
rorx64_63 ENDP

rorx64_64 PROC FRAME
    .endprolog
    rorx rax, rcx, 64
    ret
rorx64_64 ENDP

rorx64_255 PROC FRAME
    .endprolog
    rorx rax, rcx, 255
    ret
rorx64_255 ENDP

sarx32 PROC FRAME
    .endprolog
    mov rax, -1
    sarx eax, ecx, edx
    ret
sarx32 ENDP

sarx64 PROC FRAME
    .endprolog
    sarx rax, rcx, rdx
    ret
sarx64 ENDP

shlx32 PROC FRAME
    .endprolog
    mov rax, -1
    shlx eax, ecx, edx
    ret
shlx32 ENDP

shlx64 PROC FRAME
    .endprolog
    shlx rax, rcx, rdx
    ret
shlx64 ENDP

shrx32 PROC FRAME
    .endprolog
    mov rax, -1
    shrx eax, ecx, edx
    ret
shrx32 ENDP

shrx64 PROC FRAME
    .endprolog
    shrx rax, rcx, rdx
    ret
shrx64 ENDP

END

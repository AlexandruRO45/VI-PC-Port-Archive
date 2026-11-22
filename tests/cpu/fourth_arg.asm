; A fourth integer argument in R9 for the jump-table translator.
; Every export takes RCX, RDX and R8 = address of three writable QWORDs. r9_mix, r9_loop and no_r9
; are translated with R9 as a fourth argument; r9_scratch takes three and uses R9 as scratch.
.code
; RAX = RCX + 2 * R9 - RDX, [R8] = R9.
r9_mix PROC FRAME
    .endprolog
    lea rax, [rcx+r9*2]
    sub rax, rdx
    mov QWORD PTR [r8], r9
    ret
r9_mix ENDP

; R9D counts additions of RDX to RAX = RCX down to zero when it is 1 to 64; otherwise RAX = RCX.
; [R8] = R9 as the function leaves it: unchanged when no addition runs, else 0, because the
; 32-bit DEC clears bits 32 to 63.
r9_loop PROC FRAME
    .endprolog
    mov rax, rcx
    cmp r9d, 64
    ja loop_done
    test r9d, r9d
    jz loop_done
loop_add:
    add rax, rdx
    dec r9d
    jnz loop_add
loop_done:
    mov QWORD PTR [r8], r9
    ret
r9_loop ENDP

; Does not touch R9. RAX = RCX - RDX, also stored to [R8].
no_r9 PROC FRAME
    .endprolog
    mov rax, rcx
    sub rax, rdx
    mov QWORD PTR [r8], rax
    ret
no_r9 ENDP

; Writes R9B before any read of R9 and reads only R9B, then writes all of R9: the incoming R9 is
; never read. RAX = (RCX & 0FFh) + RDX, [R8] = RDX - (RCX & 0FFh).
r9_scratch PROC FRAME
    .endprolog
    mov r9b, cl
    movzx eax, r9b
    mov r9, rdx
    sub r9, rax
    mov QWORD PTR [r8], r9
    add rax, rdx
    ret
r9_scratch ENDP
END

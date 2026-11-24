; call_mix(x, y, trace) with the same contract as direct.asm, but the call goes through an
; absolute code pointer in .data. The linker emits a DIR64 relocation for that pointer.
EXTERN __imp_mix_args:QWORD
.data
route DQ forward
.code
forward PROC
    jmp QWORD PTR [__imp_mix_args]
forward ENDP

call_mix PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rbx, rcx
    mov rsi, rdx
    add rcx, 7
    sub rdx, 3
    call QWORD PTR [route]
    add rax, rbx
    xor rax, rsi
    add rsp, 28h
    pop rsi
    pop rbx
    ret
call_mix ENDP
END

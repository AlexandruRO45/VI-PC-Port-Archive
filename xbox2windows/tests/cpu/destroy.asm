; Independent fixture for wrapper ownership and the two cleanup paths.
EXTERN __imp_free:QWORD
.code
release_context PROC FRAME
    push rsi
    .pushreg rsi
    push rdi
    .pushreg rdi
    push rbx
    .pushreg rbx
    sub rsp, 32
    .allocstack 32
    .endprolog
    test rcx, rcx
    jz empty_handle
    mov rsi, [rcx]
    test rsi, rsi
    jz empty_handle
    mov rdi, rdx
    lea rcx, [rsi+8]
    mov r10, [rsi]
    call qword ptr [r10+8]
    mov ebx, eax
    test rdi, rdi
    jz default_cleanup
    mov rcx, [rdi]
    mov rdx, rsi
    call qword ptr [rdi+16]
    jmp result
default_cleanup:
    mov rcx, rsi
    call qword ptr [__imp_free]
result:
    mov eax, ebx
    jmp finish
empty_handle:
    mov eax, 6
finish:
    add rsp, 32
    pop rbx
    pop rdi
    pop rsi
    ret
release_context ENDP
END

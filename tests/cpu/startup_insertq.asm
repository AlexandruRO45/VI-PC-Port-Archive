option casemap:none
EXTERN ExitProcess:PROC
.code
entry PROC
    sub rsp, 28h
    mov rax, 1122334455667788h
    movq xmm0, rax
    mov rax, 9900aabbccddeeffh
    movq xmm1, rax
    cmp rax, rax
    ; INSERTQ xmm0, xmm1, 16, 16
    db 0f2h, 00fh, 078h, 0c1h, 10h, 10h
    jne failed
    jc failed
    movq rdx, xmm0
    mov rcx, 11223344eeff7788h
    cmp rdx, rcx
    jne failed
    movq rdx, xmm1
    cmp rax, rdx
    jne failed
    xor ecx, ecx
    call ExitProcess
failed:
    mov ecx, 1
    call ExitProcess
entry ENDP
END

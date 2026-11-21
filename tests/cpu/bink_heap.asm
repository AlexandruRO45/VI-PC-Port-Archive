; Independently written control: round the END address down to a 32-byte boundary.
EXTERN XMemAlloc:PROC
EXTERN XMemFree:PROC

.code
heap_request PROC FRAME
    sub rsp, 40
    .allocstack 40
    .endprolog
    test rcx, rcx
    jz invalid
    mov edx, 0ffffffffh
    cmp rcx, rdx
    je invalid
    add rcx, 64
    mov edx, 05e50000h
    call XMemAlloc
    test rax, rax
    je done
    lea rdx, [rax+64]
    and rdx, -32
    mov rcx, rdx
    sub rcx, rax
    mov byte ptr [rdx-1], cl
    mov byte ptr [rdx-2], 0
    mov rax, rdx
    jmp done
invalid:
    xor eax, eax
done:
    add rsp, 40
    ret
heap_request ENDP

heap_release PROC
    test rcx, rcx
    je released
    movzx edx, byte ptr [rcx-1]
    sub rcx, rdx
    mov edx, 05e50000h
    jmp XMemFree
released:
    ret
heap_release ENDP
END

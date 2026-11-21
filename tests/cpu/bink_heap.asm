; Independently written control: round the END address down to a 32-byte boundary.
EXTERN XMemAlloc:PROC
EXTERN XMemFree:PROC

.data
alloc_hook QWORD 0
free_hook QWORD 0

.code
BinkSetMemory PROC
    mov free_hook, rdx
    mov alloc_hook, rcx
    ret
BinkSetMemory ENDP

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
    mov [rsp+32], rcx
    mov rax, alloc_hook
    test rax, rax
    jz fallback
    call rax
    cmp rax, -1
    je invalid
    test rax, rax
    jz fallback
    mov r8b, 3
    jmp align_result
fallback:
    mov rcx, [rsp+32]
    mov edx, 05e50000h
    call XMemAlloc
    test rax, rax
    je done
    xor r8d, r8d
align_result:
    lea rdx, [rax+64]
    and rdx, -32
    mov rcx, rdx
    sub rcx, rax
    mov byte ptr [rdx-1], cl
    mov byte ptr [rdx-2], r8b
    test r8b, r8b
    jz ready
    mov rax, free_hook
    mov [rdx-16], rax
ready:
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
    mov r10, rcx
    movzx edx, byte ptr [rcx-1]
    sub rcx, rdx
    cmp byte ptr [r10-2], 3
    jne release_default
    jmp qword ptr [r10-16]
release_default:
    mov edx, 05e50000h
    jmp XMemFree
released:
    ret
heap_release ENDP
END

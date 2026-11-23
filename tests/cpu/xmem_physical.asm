EXTERN __imp_XMemAllocatePhysicalPages:QWORD
EXTERN __imp_XMemFreePhysicalPages:QWORD

.data
page_count dq 2
pages dq 2 dup (0)

.code
native_entry PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov ecx, 20400000h
    lea rdx, page_count
    lea r8, pages
    call qword ptr [__imp_XMemAllocatePhysicalPages]
    test eax, eax
    jz fail_allocate
    cmp qword ptr [page_count], 2
    jne fail_count
    mov rax, qword ptr [pages]
    test rax, rax
    jz fail_pfn
    mov rdx, qword ptr [pages+8]
    test rdx, rdx
    jz fail_pfn
    cmp rax, rdx
    je fail_pfn

    mov ecx, 2
    lea rdx, pages
    call qword ptr [__imp_XMemFreePhysicalPages]
    test eax, eax
    jz fail_release

    mov qword ptr [page_count], 3
    mov ecx, 20000000h
    lea rdx, page_count
    lea r8, pages
    call qword ptr [__imp_XMemAllocatePhysicalPages]
    test eax, eax
    jnz fail_large
    cmp qword ptr [page_count], 0
    jne fail_large

    mov ecx, 1
    xor edx, edx
    call qword ptr [__imp_XMemFreePhysicalPages]
    test eax, eax
    jnz fail_null

    xor eax, eax
    jmp done

fail_allocate:
    mov eax, 1
    jmp done
fail_count:
    mov eax, 2
    jmp done
fail_pfn:
    mov eax, 3
    jmp done
fail_release:
    mov eax, 4
    jmp done
fail_large:
    mov eax, 5
    jmp done
fail_null:
    mov eax, 6
done:
    add rsp, 38h
    ret
native_entry ENDP
END
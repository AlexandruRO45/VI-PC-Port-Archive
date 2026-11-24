; Source-owned control for 32-bit effective addresses in the jump-table translator.
; With a 67h prefix the hardware wraps base + displacement at 32 bits, ignores the
; upper half of the base register and zero-extends the address.
.code
lea_wrap PROC FRAME
    .endprolog
    lea rax, [ecx+1]
    ret
lea_wrap ENDP

lea_back PROC FRAME
    .endprolog
    lea rax, [ecx-8]
    ret
lea_back ENDP

; RCX = address of word 2 in a page below 4 GiB, with arbitrary upper bits.
load_back PROC FRAME
    .endprolog
    mov eax, dword ptr [ecx-8]
    ret
load_back ENDP

; RCX as for load_back, EDX = value stored two words ahead and returned.
store_ahead PROC FRAME
    .endprolog
    mov dword ptr [ecx+8], edx
    mov eax, edx
    ret
store_ahead ENDP
END

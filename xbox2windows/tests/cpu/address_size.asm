; Source-owned control for 32-bit effective addresses (address-size prefix 67h).
; The hardware wraps base + index * scale + displacement at 32 bits and ignores
; the upper halves of RCX and RDX; the result is zero-extended to 64 bits.
.code
lea_sum32 PROC FRAME
    .endprolog
    lea rax, [ecx+edx]
    ret
lea_sum32 ENDP

lea_scaled32 PROC FRAME
    .endprolog
    lea rax, [ecx+edx*8-8]
    ret
lea_scaled32 ENDP

; RCX = low buffer address with arbitrary upper bits, RDX = 32-bit offset.
load32 PROC FRAME
    .endprolog
    mov eax, dword ptr [ecx+edx]
    ret
load32 ENDP

; R8D = value stored at the 32-bit address and returned.
store32 PROC FRAME
    .endprolog
    mov dword ptr [ecx+edx], r8d
    mov eax, r8d
    ret
store32 ENDP
END

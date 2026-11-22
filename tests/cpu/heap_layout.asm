; Source-owned arithmetic control for the device-init lowerer.
.code
heap_layout PROC FRAME
    .endprolog
    mov r8d,ecx
    shl r8d,9
    mov [rdx],r8d
    imul eax,ecx,19000h
    add eax,10000h
    mov [rdx+4],eax
    imul eax,ecx,0be000h
    or rax,1100h
    ret
heap_layout ENDP

; Compact public fixture: scan one flag byte per record and sum four sizes
; into the return value when the index output is written.
slot_probe PROC FRAME
    .endprolog
    xor eax,eax
    test ecx,ecx
    je exhausted
scan:
    movzx r10d,byte ptr [rdx+rax]
    test r10d,1
    je available
    inc eax
    cmp eax,ecx
    jb scan
exhausted:
    xor eax,eax
    ret
available:
    mov [r9],eax
    vmovdqu xmm2,xmmword ptr [r8]
    vpsrldq xmm1,xmm2,4
    vpaddd xmm2,xmm1,xmm2
    vpsrldq xmm0,xmm2,8
    vpaddd xmm0,xmm2,xmm0
    vmovd eax,xmm0
    ret
slot_probe ENDP

; Exercise full-width descriptor copies, zeroing and packed 16-bit fields.
descriptor_fields PROC FRAME
    .endprolog
    vmovups ymm1,ymmword ptr [rdx]
    vmovups ymmword ptr [r8],ymm1
    vzeroupper
    vmovups ymmword ptr [r8+32],ymm1
    vpxor xmm1,xmm1,xmm1
    vmovups ymmword ptr [r8+64],ymm1
    movsxd rax,ecx
    mov ax,17
    sub ecx,1
    cmovne eax,ecx
    ret
descriptor_fields ENDP

; Public control for conditional mip selection and descriptor stride math.
view_offset PROC FRAME
    .endprolog
    xor eax,eax
    dec r9d
    cmp r9d,1
    cmovbe eax,edx
    add eax,ecx
    cmp eax,511
    ja out_of_heap
    imul eax,r8d
    ret
out_of_heap:
    mov eax,0ffffffffh
    ret
view_offset ENDP

; Compare 32-bit registers with sign-extended 8-bit immediates.
index_guard PROC FRAME
    .endprolog
    mov eax,ecx
    cmp ecx,-1
    je no_index
    cmp ecx,-3
    jae reserved_index
    ret
no_index:
    xor eax,eax
    ret
reserved_index:
    mov eax,1
    ret
index_guard ENDP
END

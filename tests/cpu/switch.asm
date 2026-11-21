; Windows x64 jump-table dispatch in the two layouts emitted by MSVC and LLVM.
; select_*: ECX = selector, EDX = operand, R8 = address of one writable DWORD.
; run_ops:  RCX = byte program, EDX = nonzero length, R8 = address of one DWORD.
EXTERN __ImageBase:BYTE
.code
; MSVC layout: the table holds image-relative entries and the index is sign-extended.
select_rva PROC FRAME
    .endprolog
    mov eax, ecx
    cmp ecx, 6
    ja rva_default
    movsxd r9, ecx
    lea r10, __ImageBase
    mov r9d, DWORD PTR [(IMAGEREL rva_table) + r10 + r9*4]
    add r9, r10
    jmp r9
rva_case0:
    add edx, 11
    jmp rva_store
rva_case1:
    xor edx, 5A5Ah
    jmp rva_store
rva_case25:
    imul edx, edx, 3
    jmp rva_store
rva_case3:
    add edx, 7
rva_case4:
    sub edx, 2
    jmp rva_store
rva_case6:
    shl edx, 4
    jmp rva_store
rva_default:
    not edx
rva_store:
    mov DWORD PTR [r8], edx
    add eax, edx
    ret
    ALIGN 4
rva_table LABEL DWORD
    DD IMAGEREL rva_case0
IFDEF SWAP_TABLE
    DD IMAGEREL rva_case6
ELSE
    DD IMAGEREL rva_case1
ENDIF
    DD IMAGEREL rva_case25, IMAGEREL rva_case3, IMAGEREL rva_case4, IMAGEREL rva_case25
IFDEF SWAP_TABLE
    DD IMAGEREL rva_case1
ELSE
    DD IMAGEREL rva_case6
ENDIF
select_rva ENDP

; LLVM layout: entries are signed offsets from the table and the index is a
; zero-extending copy of the value that was compared.
select_rel PROC FRAME
    .endprolog
    mov eax, ecx
    cmp ecx, 6
    ja rel_default
    mov ecx, eax
    lea r9, rel_table
    movsxd rcx, DWORD PTR [r9+rcx*4]
    add rcx, r9
    jmp rcx
rel_case25:
    lea edx, [rdx+rdx*2]
    jmp rel_store
rel_default:
    not edx
    jmp rel_store
rel_case0:
    add edx, 11
    jmp rel_store
rel_case1:
    xor edx, 5A5Ah
    jmp rel_store
rel_case3:
    add edx, 7
rel_case4:
    add edx, -2
    jmp rel_store
rel_case6:
    shl edx, 4
rel_store:
    mov DWORD PTR [r8], edx
    add eax, edx
    ret
    ALIGN 4
rel_table:
    DD rel_case0 - rel_table
IFDEF SWAP_TABLE
    DD rel_case6 - rel_table
ELSE
    DD rel_case1 - rel_table
ENDIF
    DD rel_case25 - rel_table, rel_case3 - rel_table, rel_case4 - rel_table, rel_case25 - rel_table
IFDEF SWAP_TABLE
    DD rel_case1 - rel_table
ELSE
    DD rel_case6 - rel_table
ENDIF
select_rel ENDP

; A bytecode loop: each byte selects an operation on EAX through the table.
; An opcode above 4 stores the remaining length and returns all ones.
run_ops PROC FRAME
    .endprolog
    xor eax, eax
next_op:
    movzx r9d, BYTE PTR [rcx]
    add rcx, 1
    cmp r9d, 4
    ja bad_op
    lea r10, __ImageBase
    mov r9d, DWORD PTR [(IMAGEREL op_table) + r10 + r9*4]
    add r9, r10
    jmp r9
op_add:
    add eax, 3
    jmp step
op_mul:
    imul eax, eax, 5
    jmp step
op_xor:
    xor eax, 0A5h
    jmp step
op_shift:
    shr eax, 1
    jmp step
op_neg:
    neg eax
step:
    sub edx, 1
    jne next_op
    mov DWORD PTR [r8], eax
    ret
bad_op:
    mov DWORD PTR [r8], edx
    mov eax, -1
    ret
    ALIGN 4
op_table LABEL DWORD
IFDEF SWAP_TABLE
    DD IMAGEREL op_xor, IMAGEREL op_mul, IMAGEREL op_add
ELSE
    DD IMAGEREL op_add, IMAGEREL op_mul, IMAGEREL op_xor
ENDIF
    DD IMAGEREL op_shift, IMAGEREL op_neg
run_ops ENDP

; Narrow memory shifts: the DWORD at R8 first receives EDX, then ECX selects one of six byte or
; word shifts of it by CL = EDX bits 16..23. x86 masks the count to five bits for 8- and 16-bit
; destinations too, so counts of 8 to 31 still apply (a byte shifted left by 8 becomes zero).
shift_narrow PROC FRAME
    .endprolog
    mov DWORD PTR [r8], edx
    mov eax, ecx
    cmp ecx, 5
    ja narrow_done
    mov ecx, eax
    lea r9, narrow_table
    movsxd rcx, DWORD PTR [r9+rcx*4]
    add rcx, r9
    jmp rcx
narrow_shl8:
    mov ecx, edx
    shr ecx, 16
    shl BYTE PTR [r8], cl
    jmp narrow_done
narrow_shr8:
    mov ecx, edx
    shr ecx, 16
    shr BYTE PTR [r8], cl
    jmp narrow_done
narrow_sar8:
    mov ecx, edx
    shr ecx, 16
    sar BYTE PTR [r8], cl
    jmp narrow_done
narrow_shl16:
    mov ecx, edx
    shr ecx, 16
    shl WORD PTR [r8], cl
    jmp narrow_done
narrow_shr16:
    mov ecx, edx
    shr ecx, 16
    shr WORD PTR [r8], cl
    jmp narrow_done
narrow_sar16:
    mov ecx, edx
    shr ecx, 16
    sar WORD PTR [r8], cl
narrow_done:
    mov eax, DWORD PTR [r8]
    ret
    ALIGN 4
narrow_table:
    DD narrow_shl8 - narrow_table, narrow_shr8 - narrow_table, narrow_sar8 - narrow_table
    DD narrow_shl16 - narrow_table, narrow_shr16 - narrow_table, narrow_sar16 - narrow_table
shift_narrow ENDP
END

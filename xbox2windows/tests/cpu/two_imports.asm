; Two import descriptors for the pe_map import-walk comparison: mix_args from the named import
; provider and probe from the register probe. The test reads only the bound IAT slots; nothing
; here is called.
EXTERN __imp_mix_args:QWORD
EXTERN __imp_probe:QWORD
.code
call_mix PROC
    jmp QWORD PTR [__imp_mix_args]
call_mix ENDP

call_probe PROC
    jmp QWORD PTR [__imp_probe]
call_probe ENDP
END

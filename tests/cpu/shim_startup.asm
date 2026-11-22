; Run the companion after the shared EXE wrapper starts the return-based body.
EXTERN __imp_run_startup_shims:QWORD

.code
native_entry PROC
    jmp qword ptr [__imp_run_startup_shims]
native_entry ENDP
END

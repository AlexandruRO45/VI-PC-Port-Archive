# D3D12ConfigureVirtualMemoryX caller contract

On retail image SHA-256 e95020b2fa4a721db3df59fdecf75621730ac559d8835252f6c3b658d5a08993,
the function starting at RVA 0x1e1cf60 calls the import at 0x1e1cfe0
through thunk 0x5f3fd60 and IAT slot 0x6793c08.

The caller zeroes 32 stack bytes at RSP+0x30 and writes one little-endian
dword at offset 4. RCX points at the block. The stack frame guarantees
16-byte alignment here. The caller tests the sign of EAX and breaks on
failure. The following allocation has its own arguments.

| Cached predicate at RVA 0x75140 | Size used in caller arithmetic | Dword at +4 |
|---|---:|---:|
| Platform query equals 6 | 16 GiB | 3 |
| Otherwise on this path | 24 GiB | 4 |

The predicate is set by a byte-sized equality comparison. The caller scales
the selected size down by 4096, rounds up in groups of 0x400000 units, then
adds two. The helper at RVA 0x4da2350 returns its argument unchanged.
These shift counts describe caller arithmetic. The driver's page granularity,
required alignment, semantics of the zero bytes, and required VA state on
success remain unknown. The retail heap-corruption exception alone cannot
identify which component damaged the heap.

## Reproduce the ABI probe

On AMD64 Windows with MinGW-w64:

    gcc -std=c11 -municode tools/configure_vm_host.c -o configure_vm_host.exe
    configure_vm_host.exe
    configure_vm_host.exe C:\absolute\path\to\d3d12_xs.dll

With no argument the host compares every byte in both synthetic inputs and
checks both success and failure returns. With a shim DLL it calls the named
export once for each observed input shape and prints both HRESULTs. A shim
accepting both calls establishes ABI reachability only.

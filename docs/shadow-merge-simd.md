# CPU-side GPU write-back merge

The shadow copy-back path merges three byte streams: current title memory,
the host GPU copy, and the copy's pre-submission snapshot. Each byte
modified by the GPU takes precedence; otherwise preserve the CPU's
current byte. The host copy receives the resulting byte. The return value
is true only when the title bytes changed.

The AVX2 implementation processes 32 bytes at a time using equality masks,
byte selection and a reduction of changed bits. CPU feature detection
selects it; the scalar loop handles unsupported architectures and tails.
No GPU commands, barriers, copy-back range boundaries, page tracking,
or write precedence are changed. No new dependencies.

Tests cover conflicting writes, unchanged title data, every alignment
from 0 through 31, tails, and whole 64 KiB pieces. Run:

    cargo test -p victory-gpu shadow::merge --locked
    cargo test -p victory-gpu shadow::tests --locked
    cargo clippy -p victory-gpu --all-targets --locked -- -D warnings
    cargo xtask policy

Measure release-mode copy-back timings and whole frame times on identical
hardware and workloads before claiming a speedup. The compiler may
already vectorize some scalar code, and memory traffic can dominate the
merge. The explicit SIMD path does not reduce bytes read or written or
make claims about GPU driver behavior.

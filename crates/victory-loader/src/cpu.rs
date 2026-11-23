//! Instructions of the console's CPU that the host's CPU lacks.
//!
//! Title code is built for the console's AMD Zen 2 and uses instructions
//! only AMD implements. On a host without them (every Intel CPU) each one
//! raises an illegal-instruction exception; a vectored handler, ahead of the
//! crash reporter's, carries the instruction out on the faulting thread's
//! saved registers and resumes after it. Nothing is patched: the handler
//! acts only where the CPU itself refused an instruction, so it cannot
//! mistake data or the middle of another instruction for one, and a host
//! that has the instruction never enters it.
//!
//! | Instruction | Encoding | What the handler does |
//! |-------------|----------|-----------------------|
//! | `CLZERO` | `0F 01 FC` | zeroes the 64-byte line holding `[rax]` |
//! | `EXTRQ x, len, idx` | `66 0F 78 /0 ib ib` | bit field of `x`, to bit 0 |
//! | `EXTRQ x, y` | `66 0F 79 /r` | the same, `len`/`idx` from `y` |
//! | `INSERTQ x, y, len, idx` | `F2 0F 78 /r ib ib` | low bits of `y` into a field of `x` |
//! | `INSERTQ x, y` | `F2 0F 79 /r` | the same, `len`/`idx` from `y`'s high half |
//! | `MONITORX` | `0F 01 FA` | nothing |
//! | `MWAITX` | `0F 01 FB` | yields the thread and returns (a spurious wake-up) |
//! | `RDPRU` | `0F 01 FD` | the time-stamp counter for both registers |
//!
//! An exception costs microseconds, so the two loops compilers build
//! around `CLZERO` (see [`clzero`]) are carried out whole, in one exception
//! instead of one per 64 bytes.
//!
//! Not emulated: `MOVNTSS`/`MOVNTSD` (SSE4a, `F3`/`F2 0F 2B`), which Grand
//! Theft Auto VI does not contain; they would be reported as crashes.

use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::Foundation::EXCEPTION_ILLEGAL_INSTRUCTION;
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    EXCEPTION_POINTERS,
};

/// The instructions emulated, as the log names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Instruction {
    Clzero,
    Extrq,
    Insertq,
    Monitorx,
    Mwaitx,
    Rdpru,
}

const NAMES: [&str; 6] = ["CLZERO", "EXTRQ", "INSERTQ", "MONITORX", "MWAITX", "RDPRU"];

/// Exceptions taken per instruction (a whole `CLZERO` loop counts once).
static EMULATED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// How many instructions were emulated so far, by name; for a summary when
/// the title ends.
pub fn emulated_instructions() -> Vec<(&'static str, u64)> {
    NAMES
        .iter()
        .zip(&EMULATED)
        .map(|(name, n)| (*name, n.load(Ordering::Relaxed)))
        .filter(|(_, n)| *n != 0)
        .collect()
}

/// The emulated instructions the host CPU lacks, by name (CPUID).
pub fn missing_instructions() -> Vec<&'static str> {
    use std::arch::x86_64::__cpuid;
    let extended = __cpuid(0x8000_0000).eax;
    let leaf = |n: u32| (extended >= n).then(|| __cpuid(n));
    let (features, more) = (leaf(0x8000_0001), leaf(0x8000_0008));
    let has = |leaf: Option<u32>, bit: u32| leaf.is_some_and(|r| r >> bit & 1 != 0);
    let mut missing = Vec::new();
    if !has(more.map(|r| r.ebx), 0) {
        missing.push("CLZERO");
    }
    if !has(features.map(|r| r.ecx), 6) {
        missing.push("EXTRQ/INSERTQ (SSE4a)");
    }
    if !has(features.map(|r| r.ecx), 29) {
        missing.push("MONITORX/MWAITX");
    }
    if !has(more.map(|r| r.ebx), 4) {
        missing.push("RDPRU");
    }
    missing
}

/// Installs the handler, once; call before running title code. Says which
/// instructions the host lacks, if any.
pub fn emulate_missing_instructions() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let missing = missing_instructions();
        if !missing.is_empty() {
            tracing::info!(
                instructions = missing.join(", "),
                "the host CPU lacks instructions of the console's; they are emulated"
            );
        }
        // SAFETY: registers a handler that only reads the faulting code and
        // changes the faulting thread's saved registers. First in the
        // chain, so the crash reporter sees only what is not emulated.
        unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
    });
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes the exception being dispatched and the
    // faulting thread's context, which it resumes from.
    let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    if record.ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let at = context.Rip as usize;
    // SAFETY: the CPU fetched an instruction at `Rip`, on its saved state.
    let Some(instruction) = (unsafe { emulate(context) }) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    // Said at powers of two: the first use of each, and how hot it gets.
    let count = EMULATED[instruction as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if count.is_power_of_two() {
        tracing::info!(
            instruction = NAMES[instruction as usize],
            count,
            at = crate::crash::locate(at),
            "instruction emulated"
        );
    }
    EXCEPTION_CONTINUE_EXECUTION
}

/// The code bytes at `at`, up to 16 and never past the end of its page (the
/// next one may not be mapped).
///
/// # Safety
/// `at` is in a readable page.
unsafe fn code_at<'a>(at: usize) -> &'a [u8] {
    let len = (0x1000 - (at & 0xFFF)).min(16);
    // SAFETY: inside the page the caller named.
    unsafe { std::slice::from_raw_parts(at as *const u8, len) }
}

/// Carries out the instruction at `c.Rip` if it is one of ours and moves
/// `Rip` past it.
///
/// # Safety
/// `c` is the state of a thread that faulted with an illegal instruction
/// at `c.Rip`; for `CLZERO` the thread's `rax` is as valid as the title
/// made it (a bad address faults here as it would have there).
unsafe fn emulate(c: &mut CONTEXT) -> Option<Instruction> {
    let at = c.Rip as usize;
    // SAFETY: an instruction was fetched at `at`.
    let code = unsafe { code_at(at) };
    match *code {
        [0x0F, 0x01, 0xFC, ..] => {
            // The loop forms look 7 bytes back; not across a page start.
            let before = ((at & 0xFFF) >= 7).then(|| {
                // SAFETY: the same page as `at`.
                unsafe { &*((at - 7) as *const [u8; 7]) }
            });
            // SAFETY: forwarded from the caller.
            unsafe { clzero(c, before, &code[3..]) };
            Some(Instruction::Clzero)
        }
        [0x0F, 0x01, 0xFA, ..] => {
            // Arms a monitor on `[rax]`. `MWAITX` returns at once here, so
            // there is nothing to arm.
            c.Rip += 3;
            Some(Instruction::Monitorx)
        }
        [0x0F, 0x01, 0xFB, ..] => {
            // Waits for a write to the monitored line, an interrupt or the
            // timeout in `ebx`; it may return early for any reason, so its
            // callers test their condition again. Give the core away once.
            std::thread::yield_now();
            c.Rip += 3;
            Some(Instruction::Mwaitx)
        }
        [0x0F, 0x01, 0xFD, ..] => {
            // `ecx` 0 reads MPERF, 1 APERF (cycles at the nominal and the
            // actual frequency). The time-stamp counter serves for both: a
            // core that never changes frequency.
            let cycles = if c.Rcx as u32 <= 1 {
                // SAFETY: reads the time-stamp counter.
                unsafe { std::arch::x86_64::_rdtsc() }
            } else {
                0
            };
            c.Rax = cycles & 0xFFFF_FFFF;
            c.Rdx = cycles >> 32;
            c.Rip += 3;
            Some(Instruction::Rdpru)
        }
        _ => sse4a(c, code),
    }
}

/// `x`'s `len` bits from bit `idx`, at bit 0 (`EXTRQ`). A `len` of 0 is 64;
/// both count modulo 64, as the CPU reads six bits of each.
fn extract_field(x: u64, len: u8, idx: u8) -> u64 {
    (x >> (idx & 63)) & mask(len)
}

/// `x` with the `len` bits from bit `idx` replaced by `y`'s low `len` bits
/// (`INSERTQ`).
fn insert_field(x: u64, y: u64, len: u8, idx: u8) -> u64 {
    let (mask, idx) = (mask(len), idx & 63);
    x & !(mask << idx) | (y & mask) << idx
}

fn mask(len: u8) -> u64 {
    match len & 63 {
        0 => !0,
        len => (1 << len) - 1,
    }
}

/// `EXTRQ` and `INSERTQ`, register operands only (they have no others).
/// The destination's high half, undefined afterwards, is left as it was.
fn sse4a(c: &mut CONTEXT, code: &[u8]) -> Option<Instruction> {
    let (&prefix, rest) = code.split_first()?;
    let (rex, rest) = match rest.split_first()? {
        (&rex @ 0x40..=0x4F, rest) => (rex, rest),
        _ => (0, rest),
    };
    let (&[0x0F, opcode @ (0x78 | 0x79), modrm], immediates) = rest.split_first_chunk()? else {
        return None;
    };
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = usize::from((modrm >> 3) & 7 | (rex & 4) << 1);
    let rm = usize::from(modrm & 7 | (rex & 1) << 3);
    let used = code.len() - immediates.len();
    // SAFETY: an exception context carries the SSE registers in `FltSave`.
    let xmm = unsafe { &mut c.Anonymous.FltSave.XmmRegisters };
    let (instruction, len) = match (prefix, opcode) {
        (0x66, 0x78) if reg == 0 => {
            let &[len, idx] = immediates.first_chunk()?;
            xmm[rm].Low = extract_field(xmm[rm].Low, len, idx);
            (Instruction::Extrq, used + 2)
        }
        (0x66, 0x79) => {
            let [len, idx, ..] = xmm[rm].Low.to_le_bytes();
            xmm[reg].Low = extract_field(xmm[reg].Low, len, idx);
            (Instruction::Extrq, used)
        }
        (0xF2, 0x78) => {
            let &[len, idx] = immediates.first_chunk()?;
            xmm[reg].Low = insert_field(xmm[reg].Low, xmm[rm].Low, len, idx);
            (Instruction::Insertq, used + 2)
        }
        (0xF2, 0x79) => {
            let [len, idx, ..] = xmm[rm].High.to_le_bytes();
            xmm[reg].Low = insert_field(xmm[reg].Low, xmm[rm].Low, len, idx);
            (Instruction::Insertq, used)
        }
        _ => return None,
    };
    c.Rip += len as u64;
    Some(instruction)
}

/// The general register numbered `n` as instructions encode it.
fn register(c: &mut CONTEXT, n: u8) -> &mut u64 {
    match n & 15 {
        0 => &mut c.Rax,
        1 => &mut c.Rcx,
        2 => &mut c.Rdx,
        3 => &mut c.Rbx,
        4 => &mut c.Rsp,
        5 => &mut c.Rbp,
        6 => &mut c.Rsi,
        7 => &mut c.Rdi,
        8 => &mut c.R8,
        9 => &mut c.R9,
        10 => &mut c.R10,
        11 => &mut c.R11,
        12 => &mut c.R12,
        13 => &mut c.R13,
        14 => &mut c.R14,
        _ => &mut c.R15,
    }
}

/// The status flags `cmp a, b` leaves (CF, PF, AF, ZF, SF, OF), and the
/// mask of those bits in `EFLAGS`.
fn compare(a: u64, b: u64) -> u32 {
    let r = a.wrapping_sub(b);
    u32::from(a < b)
        | u32::from((r as u8).count_ones().is_multiple_of(2)) << 2
        | (((a ^ b ^ r) >> 4) as u32 & 1) << 4
        | u32::from(r == 0) << 6
        | ((r >> 63) as u32) << 7
        | ((((a ^ b) & (a ^ r)) >> 63) as u32) << 11
}
const STATUS_FLAGS: u32 = 0x8D5;

/// `cmp rm, reg` between 64-bit registers (`REX.W 39 /r`): (`rm`, `reg`).
fn compare_registers(code: &[u8]) -> Option<(u8, u8)> {
    let &[rex @ 0x48..=0x4F, 0x39, modrm] = code else {
        return None;
    };
    (modrm >> 6 == 3).then_some((
        modrm & 7 | (rex & 1) << 3,
        (modrm >> 3) & 7 | (rex & 4) << 1,
    ))
}

/// `CLZERO` at `c.Rip`: zeroes the 64-byte line holding `[rax]`. `before`
/// are the 7 code bytes ahead of it, `after` those that follow.
///
/// Two loops around it are carried out whole, registers, flags and all, as
/// if every pass had run (`a` and `b` any registers but `rax`):
///
/// ```text
/// top: clzero                 top: mov rax, a
///      add rax, 0x40               add a, 0x40
///      cmp rax, b                  clzero
///      jb  top                     cmp a, b
///                                  jb  top
/// ```
///
/// # Safety
/// As [`emulate`]: the lines the title's registers name are its to zero.
unsafe fn clzero(c: &mut CONTEXT, before: Option<&[u8; 7]>, after: &[u8]) {
    // SAFETY: (both uses) the line the title asked to zero.
    let zero = |rax: u64| unsafe { ((rax & !63) as *mut u8).write_bytes(0, 64) };
    if let [0x48, 0x83, 0xC0, 0x40, cmp @ .., 0x72, 0xF4] = &after[..after.len().min(9)]
        && let Some((0, limit)) = compare_registers(cmp)
    {
        let limit = *register(c, limit);
        loop {
            zero(c.Rax);
            c.Rax = c.Rax.wrapping_add(0x40);
            if c.Rax >= limit {
                break;
            }
        }
        c.EFlags = c.EFlags & !STATUS_FLAGS | compare(c.Rax, limit);
        c.Rip += 12;
        return;
    }
    if let Some(
        &[
            rex @ (0x48 | 0x4C),
            0x89,
            load,
            add_rex @ (0x48 | 0x49),
            0x83,
            add,
            0x40,
        ],
    ) = before
        && let [cmp @ .., 0x72, 0xF1] = &after[..after.len().min(5)]
        && let Some((a, b)) = compare_registers(cmp)
        && load & 0xC7 == 0xC0
        && a == (load >> 3) & 7 | (rex & 4) << 1
        && add & 0xF8 == 0xC0
        && a == add & 7 | (add_rex & 1) << 3
        && a != 0
        && b != 0
    {
        let limit = *register(c, b);
        loop {
            zero(c.Rax);
            let next = *register(c, a);
            if next >= limit {
                break;
            }
            c.Rax = next;
            *register(c, a) = next.wrapping_add(0x40);
        }
        c.EFlags = c.EFlags & !STATUS_FLAGS | compare(*register(c, a), limit);
        c.Rip += 8;
        return;
    }
    zero(c.Rax);
    c.Rip += 3;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::asm;
    use std::arch::x86_64::__m128i;

    /// Three cache lines of `0xAA`.
    #[repr(align(64))]
    struct Lines([u8; 192]);

    fn lines() -> Box<Lines> {
        Box::new(Lines([0xAA; 192]))
    }

    fn zeroed(lines: &Lines) -> [bool; 3] {
        let line = |i: usize| &lines.0[i * 64..][..64];
        for i in 0..3 {
            assert!(
                line(i).iter().all(|&b| b == 0) || line(i).iter().all(|&b| b == 0xAA),
                "line {i} partly zeroed"
            );
        }
        [line(0)[0] == 0, line(1)[0] == 0, line(2)[0] == 0]
    }

    // These run the real instructions: natively on a CPU that has them,
    // through the handler on one that has not.

    #[test]
    fn clzero_zeroes_the_line_holding_rax() {
        emulate_missing_instructions();
        let mut mem = lines();
        let inside = mem.0.as_mut_ptr() as usize + 64 + 5;
        let after: usize;
        // SAFETY: zeroes the middle line of `mem`.
        unsafe { asm!(".byte 0x0f, 0x01, 0xfc", inout("rax") inside => after) };
        assert_eq!(after, inside);
        assert_eq!(zeroed(&mem), [false, true, false]);
    }

    /// The flags a native `cmp a, b` leaves.
    fn native_compare(a: u64, b: u64) -> u32 {
        let flags: u64;
        // SAFETY: compares two registers and reads the flags.
        unsafe {
            asm!("cmp {a}, {b}", "pushfq", "pop {f}", a = in(reg) a, b = in(reg) b, f = out(reg) flags)
        };
        flags as u32 & STATUS_FLAGS
    }

    #[test]
    fn compare_sets_flags_like_the_cpu() {
        let values = [
            0,
            1,
            0x40,
            0x7F,
            0x80,
            0xFF,
            1 << 63,
            (1 << 63) - 1,
            !0,
            !0 - 0x40,
        ];
        for a in values {
            for b in values {
                assert_eq!(compare(a, b), native_compare(a, b), "cmp {a:#x}, {b:#x}");
            }
        }
    }

    #[test]
    fn clzero_loops_run_whole() {
        emulate_missing_instructions();
        // clzero; add rax, 0x40; cmp rax, r10; jb
        let mut mem = lines();
        let start = mem.0.as_mut_ptr() as u64;
        let (end, flags): (u64, u64);
        // SAFETY: zeroes the first two lines of `mem`.
        unsafe {
            asm!(
                ".byte 0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4",
                "pushfq",
                "pop {f}",
                inout("rax") start => end,
                in("r10") start + 100,
                f = out(reg) flags,
            )
        };
        assert_eq!(end, start + 128);
        assert_eq!(
            flags as u32 & STATUS_FLAGS,
            native_compare(end, start + 100)
        );
        assert_eq!(zeroed(&mem), [true, true, false]);

        // mov rax, rdx; add rdx, 0x40; clzero; cmp rdx, r8; jb
        let mut mem = lines();
        let start = mem.0.as_mut_ptr() as u64;
        let (last, next, flags): (u64, u64, u64);
        // SAFETY: zeroes the last two lines of `mem`.
        unsafe {
            asm!(
                ".byte 0x48, 0x89, 0xd0, 0x48, 0x83, 0xc2, 0x40, 0x0f, 0x01, 0xfc, 0x4c, 0x39, 0xc2, 0x72, 0xf1",
                "pushfq",
                "pop {f}",
                out("rax") last,
                inout("rdx") start + 64 => next,
                in("r8") start + 192,
                f = out(reg) flags,
            )
        };
        assert_eq!((last, next), (start + 128, start + 192));
        assert_eq!(
            flags as u32 & STATUS_FLAGS,
            native_compare(next, start + 192)
        );
        assert_eq!(zeroed(&mem), [false, true, true]);
    }

    fn xmm(high: u64, low: u64) -> __m128i {
        // SAFETY: two u64 of plain data.
        unsafe { std::mem::transmute([low, high]) }
    }

    fn low(x: __m128i) -> u64 {
        // SAFETY: two u64 of plain data.
        unsafe { std::mem::transmute::<__m128i, [u64; 2]>(x)[0] }
    }

    #[test]
    fn bit_fields_move_like_sse4a() {
        emulate_missing_instructions();
        let x = xmm(7, 0x1111_2222_3333_4444);
        let y = xmm(0, 0xAAAA_BBBB_CCCC_DDDD);
        let mut r: __m128i;
        // SAFETY: insertq xmm0, xmm1, 16, 16.
        unsafe {
            asm!(".byte 0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10", inout("xmm0") x => r, in("xmm1") y)
        };
        assert_eq!(low(r), 0x1111_2222_DDDD_4444);
        // SAFETY: insertq xmm9, xmm10, 8, 24.
        unsafe {
            asm!(".byte 0xf2, 0x45, 0x0f, 0x78, 0xca, 0x08, 0x18", inout("xmm9") x => r, in("xmm10") y)
        };
        assert_eq!(low(r), 0x1111_2222_DD33_4444);
        // SAFETY: extrq xmm0, 16, 8.
        unsafe { asm!(".byte 0x66, 0x0f, 0x78, 0xc0, 0x10, 0x08", inout("xmm0") x => r) };
        assert_eq!(low(r), 0x3344);
        // The register forms: length 16 and index 8, in bytes 0 and 1 of
        // the low half (extrq) or of the high half (insertq).
        let field = xmm(0x0810, 0x0810);
        // SAFETY: extrq xmm0, xmm1.
        unsafe { asm!(".byte 0x66, 0x0f, 0x79, 0xc1", inout("xmm0") x => r, in("xmm1") field) };
        assert_eq!(low(r), 0x3344);
        let y = xmm(0x0810, 0xAAAA_BBBB_CCCC_DDDD);
        // SAFETY: insertq xmm0, xmm1.
        unsafe { asm!(".byte 0xf2, 0x0f, 0x79, 0xc1", inout("xmm0") x => r, in("xmm1") y) };
        assert_eq!(low(r), 0x1111_2222_33DD_DD44);
    }

    #[test]
    fn fields_of_all_64_bits() {
        assert_eq!(extract_field(0x1234, 0, 0), 0x1234);
        assert_eq!(insert_field(0x1234, 0xFFFF, 0, 0), 0xFFFF);
        assert_eq!(extract_field(!0, 1, 63), 1);
        assert_eq!(insert_field(0, !0, 4, 60), 0xF << 60);
    }

    #[test]
    fn waits_and_counters_return() {
        emulate_missing_instructions();
        let mut word = 0u64;
        let (low, high): (u64, u64);
        // SAFETY: monitorx on `word`; mwaitx with a timeout of 1,000
        // cycles (ecx bit 1, ebx), which a CPU that has it honours; rdpru
        // of MPERF.
        unsafe {
            asm!(
                ".byte 0x0f, 0x01, 0xfa",
                inout("rax") &raw mut word => _,
                inout("rcx") 0u64 => _,
                inout("rdx") 0u64 => _,
            );
            asm!(
                "xchg rbx, {timeout}",
                ".byte 0x0f, 0x01, 0xfb",
                "mov rbx, {timeout}",
                timeout = inout(reg) 1000u64 => _,
                inout("rax") 0u64 => _,
                inout("rcx") 2u64 => _,
            );
            asm!(
                ".byte 0x0f, 0x01, 0xfd",
                out("rax") low,
                inout("rcx") 0u64 => _,
                out("rdx") high,
            );
        }
        assert_ne!(high << 32 | low, 0);
        assert!(high >> 32 == 0 && low >> 32 == 0);
    }
}

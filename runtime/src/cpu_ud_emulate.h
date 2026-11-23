// Emulation of x64 instructions the host CPU may lack, from a vectored exception handler.
//
// The game executable is built for the console's AMD CPU, so it can contain instructions a desktop CPU does not
// implement (AMD-only CLZERO, MONITORX/MWAITX, SSE4a, and CLWB on some Intel parts). On a CPU without them the
// instruction raises #UD, which Windows reports as STATUS_ILLEGAL_INSTRUCTION. The handler here looks the faulting
// instruction up in a table of emulators, performs its effect on the saved CONTEXT and resumes after it.
//
// Adding an instruction: write one emulator in cpu_ud_emulate.cpp, with the signature of `Emulator::fn`, and add one
// line to `kTable` in the same file. The dispatcher does the rest (reading the code bytes,
// advancing RIP, passing on what it does not recognise).
//
// Faults. An instruction that the real CPU would fault on (CLZERO on a read-only page, say) must not look like a
// #UD to the next handler. The emulator reports it by returning `write_fault(address)`: the dispatcher then rewrites
// the exception record in place into the page fault the CPU would have raised (STATUS_ACCESS_VIOLATION,
// ExceptionInformation = {1 (write), address}), leaves the CONTEXT as it was and returns EXCEPTION_CONTINUE_SEARCH.
#pragma once

#include <windows.h>

#include <cstddef>
#include <cstdint>

namespace rt::cpu_ud_emulate {

// The longest x64 instruction.
inline constexpr size_t kMaxInstructionLength = 15;

// One emulator. `code` points at the faulting instruction and holds the first `available` bytes of it
// (at most kMaxInstructionLength; fewer only when the code runs into unreadable memory). The emulator returns the
// instruction's length in bytes after performing its effect on `ctx` (everything except RIP, which the dispatcher
// advances), or 0 when the bytes are not its instruction or its effect cannot be performed (a bad address, say). A 0
// leaves `ctx` untouched, so the next emulator, then the next exception handler, sees the exception unchanged.
// The one other return value is kFault, from write_fault(), described above. A declined instruction (0) is passed on as
// the original #UD; a faulting one is not.
using EmulatorFn = size_t (*)(CONTEXT& ctx, const uint8_t* code, size_t available);

// Returned by an emulator whose instruction is valid but would fault: use write_fault() rather than naming it.
inline constexpr size_t kFault = SIZE_MAX;

// For an emulator to return when its instruction would raise a page fault writing to `address` (the first byte it
// cannot write). It records the address for the dispatcher, on this thread, and returns kFault. The emulator must not
// have modified `ctx`.
size_t write_fault(uintptr_t address);

struct Emulator {
    const char* name;
    EmulatorFn fn;
};

// Emulates the instruction that raised STATUS_ILLEGAL_INSTRUCTION in `info`. Returns EXCEPTION_CONTINUE_EXECUTION
// when it did (RIP then points after the instruction) and EXCEPTION_CONTINUE_SEARCH for any other exception or any
// instruction no emulator takes, with nothing modified. For an instruction that would fault it also returns
// EXCEPTION_CONTINUE_SEARCH, with only the exception record rewritten as the access violation (see "Faults"). It is the body of the vectored handler, exposed so tests can
// feed it a synthetic CONTEXT and EXCEPTION_RECORD on a CPU that has the real instructions.
LONG emulate(EXCEPTION_POINTERS* info);

// True when the host CPU implements CLZERO (CPUID 8000_0008h EBX bit 0).
bool host_has_clzero();

// True when the host CPU implements CLWB (CPUID 7.0 EBX bit 24).
bool host_has_clwb();

// Installs the vectored handler, first in line, once per process. Without `force` it does so only when the host CPU
// lacks an instruction the table emulates, since a CPU that has them executes them natively. Returns whether the
// handler is installed.
bool install(bool force = false);

// Removes the handler install() added (tests; the runtime keeps it for the life of the process).
void uninstall();

}  // namespace rt::cpu_ud_emulate

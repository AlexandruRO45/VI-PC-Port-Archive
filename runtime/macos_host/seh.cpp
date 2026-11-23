// Windows structured exception handling for guests built with the static CRT, whose C++ runtime
// (_CxxThrowException, __CxxFrameHandler4, __CxxCallCatchBlock) lives in the image itself: RaiseException captures a
// CONTEXT and walks the frames calling their language handlers (phase 1); the handler that catches calls RtlUnwindEx,
// which calls the unwind handlers up to its frame and, for STATUS_UNWIND_CONSOLIDATE, the catch callback, then
// restores the target frame's registers. Frames are walked with eh.cpp's .pdata unwinder.
// ponytail: the walks start from the raise context, not from RtlUnwindEx's caller, and the catch callback runs on the
// current stack under host frames, so an exception that leaves a catch block (rethrow, throw to an outer try) cannot
// be walked past it and fails; Windows hides the target context in a fake consolidation frame instead.
#include "host.h"

#include <cstring>

namespace host {
namespace {

constexpr size_t kContextSize = 1232;     // CONTEXT: P1Home..P4Home 0, ContextFlags 0x30, MxCsr 0x34, Rax 0x78,
constexpr size_t kGpr = 0x78, kXmm = 0x1A0;  // Rax..R15 in Windows register order then Rip at 0xF8, Xmm0 0x1A0
struct alignas(16) Context { uint8_t b[kContextSize]; };

uint64_t* gpr(Context& c) { return reinterpret_cast<uint64_t*>(c.b + kGpr); }  // gpr[16] is Rip
uint8_t* xmm(Context& c) { return c.b + kXmm; }
uint64_t& rip(Context& c) { return gpr(c)[16]; }

struct ExceptionRecord {
    uint32_t code, flags;
    uint64_t record, address;
    uint32_t n, pad;
    uint64_t info[15];
};
struct DispatcherContext {
    uint64_t control_pc, image_base;
    const void* function_entry;
    uint64_t establisher, target_ip;
    Context* context;
    uint64_t handler, handler_data;
    void* history;
    uint32_t scope_index, fill;
};
constexpr uint32_t kNoncontinuable = 1, kUnwinding = 2, kTargetUnwind = 0x20;
constexpr uint32_t kUnwindConsolidate = 0x80000029, kStatusUnwind = 0xC0000027;
using LanguageHandler = int32_t(MSABI*)(ExceptionRecord*, uint64_t, Context*, DispatcherContext*);

thread_local Context t_raise;  // the registers at the last RaiseException, where RtlUnwindEx starts

bool in_image(uint64_t pc) {
    uint64_t lo = reinterpret_cast<uint64_t>(g_image);
    uint32_t size;
    std::memcpy(&size, g_image + *reinterpret_cast<const uint32_t*>(g_image + 0x3C) + 24 + 56, 4);  // SizeOfImage
    return pc >= lo && pc < lo + size;
}

DispatcherContext dispatcher(uint64_t pc, const Unwound& u, Context* frame, uint64_t target_ip) {
    DispatcherContext dc{};
    dc.control_pc = pc;
    dc.image_base = reinterpret_cast<uint64_t>(g_image);
    dc.function_entry = lookup_function(pc);
    dc.establisher = u.establisher;
    dc.target_ip = target_ip;
    dc.context = frame;
    dc.handler = u.handler;
    dc.handler_data = u.handler_data;
    return dc;
}

}  // namespace
}  // namespace host

extern "C" [[noreturn]] MSABI void host_raise(host::Context* ctx);
extern "C" void host_raise_entry();                        // ms_abi: RaiseException(code, flags, n, args)
extern "C" void host_capture_context();                    // ms_abi: RtlCaptureContext(rcx)
extern "C" [[noreturn]] void host_restore_context(const host::Context*);  // SysV: rdi

__asm__(
    ".text\n.globl _host_raise_entry\n_host_raise_entry:\n"
    "  subq $0x4d8, %rsp\n"  // a CONTEXT on the stack, 16-aligned
    "  movq %rcx, 0x0(%rsp)\n  movq %rdx, 0x8(%rsp)\n  movq %r8, 0x10(%rsp)\n  movq %r9, 0x18(%rsp)\n"
    "  movq %rbx, 0x90(%rsp)\n  movq %rbp, 0xa0(%rsp)\n  movq %rsi, 0xa8(%rsp)\n  movq %rdi, 0xb0(%rsp)\n"
    "  movq %r12, 0xd8(%rsp)\n  movq %r13, 0xe0(%rsp)\n  movq %r14, 0xe8(%rsp)\n  movq %r15, 0xf0(%rsp)\n"
    "  movq 0x4d8(%rsp), %rax\n  movq %rax, 0xf8(%rsp)\n"  // Rip: the return address
    "  leaq 0x4e0(%rsp), %rax\n  movq %rax, 0x98(%rsp)\n"  // Rsp: as after the return
    "  movdqu %xmm6, 0x200(%rsp)\n  movdqu %xmm7, 0x210(%rsp)\n  movdqu %xmm8, 0x220(%rsp)\n"
    "  movdqu %xmm9, 0x230(%rsp)\n  movdqu %xmm10, 0x240(%rsp)\n  movdqu %xmm11, 0x250(%rsp)\n"
    "  movdqu %xmm12, 0x260(%rsp)\n  movdqu %xmm13, 0x270(%rsp)\n  movdqu %xmm14, 0x280(%rsp)\n"
    "  movdqu %xmm15, 0x290(%rsp)\n  stmxcsr 0x34(%rsp)\n"
    "  movq %rsp, %rcx\n  subq $0x20, %rsp\n  callq _host_raise\n  ud2\n"

    ".globl _host_capture_context\n_host_capture_context:\n"
    "  movq %rax, 0x78(%rcx)\n  movq %rcx, 0x80(%rcx)\n  movq %rdx, 0x88(%rcx)\n  movq %rbx, 0x90(%rcx)\n"
    "  movq %rbp, 0xa0(%rcx)\n  movq %rsi, 0xa8(%rcx)\n  movq %rdi, 0xb0(%rcx)\n  movq %r8, 0xb8(%rcx)\n"
    "  movq %r9, 0xc0(%rcx)\n  movq %r10, 0xc8(%rcx)\n  movq %r11, 0xd0(%rcx)\n  movq %r12, 0xd8(%rcx)\n"
    "  movq %r13, 0xe0(%rcx)\n  movq %r14, 0xe8(%rcx)\n  movq %r15, 0xf0(%rcx)\n"
    "  leaq 8(%rsp), %rax\n  movq %rax, 0x98(%rcx)\n  movq (%rsp), %rax\n  movq %rax, 0xf8(%rcx)\n"
    "  movdqu %xmm0, 0x1a0(%rcx)\n  movdqu %xmm1, 0x1b0(%rcx)\n  movdqu %xmm2, 0x1c0(%rcx)\n"
    "  movdqu %xmm3, 0x1d0(%rcx)\n  movdqu %xmm4, 0x1e0(%rcx)\n  movdqu %xmm5, 0x1f0(%rcx)\n"
    "  movdqu %xmm6, 0x200(%rcx)\n  movdqu %xmm7, 0x210(%rcx)\n  movdqu %xmm8, 0x220(%rcx)\n"
    "  movdqu %xmm9, 0x230(%rcx)\n  movdqu %xmm10, 0x240(%rcx)\n  movdqu %xmm11, 0x250(%rcx)\n"
    "  movdqu %xmm12, 0x260(%rcx)\n  movdqu %xmm13, 0x270(%rcx)\n  movdqu %xmm14, 0x280(%rcx)\n"
    "  movdqu %xmm15, 0x290(%rcx)\n  stmxcsr 0x34(%rcx)\n"
    "  pushfq\n  popq %rax\n  movl %eax, 0x44(%rcx)\n"
    "  movl $0x10000b, 0x30(%rcx)\n"  // CONTEXT_CONTROL | INTEGER | FLOATING_POINT
    "  movq 0x78(%rcx), %rax\n  retq\n"

    ".globl _host_restore_context\n_host_restore_context:\n"
    "  ldmxcsr 0x34(%rdi)\n"
    "  movdqu 0x1a0(%rdi), %xmm0\n  movdqu 0x1b0(%rdi), %xmm1\n  movdqu 0x1c0(%rdi), %xmm2\n"
    "  movdqu 0x1d0(%rdi), %xmm3\n  movdqu 0x1e0(%rdi), %xmm4\n  movdqu 0x1f0(%rdi), %xmm5\n"
    "  movdqu 0x200(%rdi), %xmm6\n  movdqu 0x210(%rdi), %xmm7\n  movdqu 0x220(%rdi), %xmm8\n"
    "  movdqu 0x230(%rdi), %xmm9\n  movdqu 0x240(%rdi), %xmm10\n  movdqu 0x250(%rdi), %xmm11\n"
    "  movdqu 0x260(%rdi), %xmm12\n  movdqu 0x270(%rdi), %xmm13\n  movdqu 0x280(%rdi), %xmm14\n"
    "  movdqu 0x290(%rdi), %xmm15\n"
    "  movq 0x78(%rdi), %rax\n  movq 0x80(%rdi), %rcx\n  movq 0x88(%rdi), %rdx\n  movq 0x90(%rdi), %rbx\n"
    "  movq 0xa0(%rdi), %rbp\n  movq 0xa8(%rdi), %rsi\n  movq 0xb8(%rdi), %r8\n  movq 0xc0(%rdi), %r9\n"
    "  movq 0xc8(%rdi), %r10\n  movq 0xd0(%rdi), %r11\n  movq 0xd8(%rdi), %r12\n  movq 0xe0(%rdi), %r13\n"
    "  movq 0xe8(%rdi), %r14\n  movq 0xf0(%rdi), %r15\n"
    "  movq 0x98(%rdi), %rsp\n  pushq 0xf8(%rdi)\n  movq 0xb0(%rdi), %rdi\n  retq\n");

namespace host {
namespace {

[[noreturn]] void resume(const Context& c) {
    _thread_set_tsd_base(reinterpret_cast<uint64_t>(t_teb));
    host_restore_context(&c);
}

// Phase 1: find the frame whose handler takes the exception. A handler that catches never returns: it calls
// RtlUnwindEx, which resumes in the catch's continuation.
[[noreturn]] void raise(Context& raised) {
    ExceptionRecord rec{};
    std::memcpy(&rec.code, raised.b + 0x0, 4);
    rec.flags = *reinterpret_cast<uint32_t*>(raised.b + 0x8) & kNoncontinuable;
    rec.address = rip(raised);
    uint64_t n = *reinterpret_cast<uint64_t*>(raised.b + 0x10);
    auto* args = *reinterpret_cast<const uint64_t**>(raised.b + 0x18);
    rec.n = static_cast<uint32_t>(n > 15 ? 15 : n);
    if (args) std::memcpy(rec.info, args, rec.n * 8);
    *reinterpret_cast<uint32_t*>(raised.b + 0x30) = 0x10000b;
    t_raise = raised;

    Context frame = raised;
    for (;;) {
        uint64_t pc = rip(frame);
        if (!in_image(pc)) {
            char what[64];
            std::snprintf(what, sizeof what, "unhandled exception 0x%08x", rec.code);
            if (rec.code == 0xE06D7363 && rec.n >= 3 && rec.info[1]) {  // MSVC C++: std::exception::what at +8
                const char* msg = *reinterpret_cast<const char* const*>(rec.info[1] + 8);
                fail(std::string(what) + ": " + (msg ? msg : "?"));
            }
            fail(what);
        }
        Unwound u = unwind_frame(gpr(frame), xmm(frame));
        if (!(u.flags & 1)) continue;  // UNW_FLAG_EHANDLER
        DispatcherContext dc = dispatcher(pc, u, &frame, 0);
        int32_t r = call_guest(reinterpret_cast<LanguageHandler>(u.handler), &rec, u.establisher, &raised, &dc);
        if (r == 0) {  // ExceptionContinueExecution
            if (rec.flags & kNoncontinuable) fail("a handler continued a noncontinuable exception");
            resume(raised);
        }
        if (r != 1) fail("exception handler returned " + std::to_string(r));
    }
}

// Phase 2: call every unwind handler from the raise up to target_frame, then resume there.
[[noreturn]] void RtlUnwindEx(uint64_t target_frame, uint64_t target_ip, ExceptionRecord* rec, uint64_t retval,
                              Context*, void*) {
    if (!target_frame) fail("unsupported: RtlUnwindEx without a target frame");
    ExceptionRecord local{};
    if (!rec) {
        local.code = kStatusUnwind;
        rec = &local;
    }
    rec->flags |= kUnwinding;
    Context frame = t_raise;
    for (;;) {
        uint64_t pc = rip(frame);
        if (!in_image(pc)) fail("RtlUnwindEx: target frame not found");
        Context caller = frame;
        Unwound u = unwind_frame(gpr(caller), xmm(caller));
        bool target = u.establisher == target_frame;
        if (target) rec->flags |= kTargetUnwind;
        if (u.flags & 2) {  // UNW_FLAG_UHANDLER
            DispatcherContext dc = dispatcher(pc, u, &frame, target_ip);
            int32_t r = call_guest(reinterpret_cast<LanguageHandler>(u.handler), rec, u.establisher, &frame, &dc);
            if (r != 1) fail("unwind handler returned " + std::to_string(r));
        }
        if (target) break;
        frame = caller;
    }
    gpr(frame)[0] = retval;
    rip(frame) = target_ip;
    if (rec->code == kUnwindConsolidate && rec->n >= 1)  // the catch block runs here and names the continuation
        rip(frame) = call_guest(reinterpret_cast<uint64_t(MSABI*)(ExceptionRecord*)>(rec->info[0]), rec);
    resume(frame);
}

const void* RtlLookupFunctionEntry(uint64_t pc, uint64_t* image_base, void*) {
    if (!in_image(pc)) return nullptr;
    *image_base = reinterpret_cast<uint64_t>(g_image);
    return lookup_function(pc);
}

uint64_t RtlVirtualUnwind(uint32_t type, uint64_t, uint64_t pc, const void*, Context* ctx, uint64_t* handler_data,
                          uint64_t* establisher, void*) {
    rip(*ctx) = pc;
    Unwound u = unwind_frame(gpr(*ctx), xmm(*ctx));
    *establisher = u.establisher;
    if (!(u.flags & type)) return 0;
    *handler_data = u.handler_data;
    return u.handler;
}

}  // namespace

#define K(name) EXPORT("kernel32.dll", name)
Exports seh_exports() {
    return {
        {"kernel32.dll!RaiseException", reinterpret_cast<void*>(&host_raise_entry)},
        {"kernel32.dll!RtlCaptureContext", reinterpret_cast<void*>(&host_capture_context)},
        K(RtlUnwindEx), K(RtlLookupFunctionEntry), K(RtlVirtualUnwind),
    };
}

}  // namespace host

extern "C" MSABI void host_raise(host::Context* ctx) {
    host::HostGs g;  // never destroyed: the exception resumes somewhere in the guest
    host::raise(*ctx);
}

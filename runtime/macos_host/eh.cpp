// C++ exceptions thrown and caught inside the game image: _CxxThrowException captures the thrower's registers, the
// host walks the guest frames with the image's .pdata unwind codes, reads the __CxxFrameHandler3 tables (FuncInfo)
// of each frame, runs destructor funclets, calls the matching catch funclet, and resumes at its continuation.
// ponytail: no rethrow, no nested exceptions, no __finally blocks in __C_specific_handler frames, no
// std::current_exception, no epilog detection (frames are only ever unwound from call sites).
#include "host.h"

#include <cstring>

namespace host {

struct Ctx {               // the layout the assembly below relies on
    uint64_t gpr[16];      // Windows register numbers: rax rcx rdx rbx rsp rbp rsi rdi r8..r15
    uint64_t rip;          // 128
    uint64_t pad;
    uint8_t xmm[16][16];   // 144
};

}  // namespace host

extern "C" [[noreturn]] MSABI void host_cxx_throw(void* object, const void* throw_info, host::Ctx* ctx);
extern "C" void host_cxx_throw_entry();                   // ms_abi entry: rcx object, rdx ThrowInfo
extern "C" [[noreturn]] void host_resume(const host::Ctx*);  // SysV: rdi

// _CxxThrowException: save the non-volatile registers as they are in the thrower, rsp/rip as after its call returns.
__asm__(
    ".text\n.globl _host_cxx_throw_entry\n_host_cxx_throw_entry:\n"
    "  subq $0x198, %rsp\n"  // 400-byte Ctx, rsp 16-aligned
    "  movq %rbx, 24(%rsp)\n  movq %rbp, 40(%rsp)\n  movq %rsi, 48(%rsp)\n  movq %rdi, 56(%rsp)\n"
    "  movq %r12, 96(%rsp)\n  movq %r13, 104(%rsp)\n  movq %r14, 112(%rsp)\n  movq %r15, 120(%rsp)\n"
    "  movq 0x198(%rsp), %rax\n  movq %rax, 128(%rsp)\n"
    "  leaq 0x1a0(%rsp), %rax\n  movq %rax, 32(%rsp)\n"
    "  movdqu %xmm6, 240(%rsp)\n  movdqu %xmm7, 256(%rsp)\n  movdqu %xmm8, 272(%rsp)\n  movdqu %xmm9, 288(%rsp)\n"
    "  movdqu %xmm10, 304(%rsp)\n  movdqu %xmm11, 320(%rsp)\n  movdqu %xmm12, 336(%rsp)\n"
    "  movdqu %xmm13, 352(%rsp)\n  movdqu %xmm14, 368(%rsp)\n  movdqu %xmm15, 384(%rsp)\n"
    "  movq %rsp, %r8\n  subq $0x20, %rsp\n  callq _host_cxx_throw\n  ud2\n"
    ".globl _host_resume\n_host_resume:\n"
    "  movdqu 240(%rdi), %xmm6\n  movdqu 256(%rdi), %xmm7\n  movdqu 272(%rdi), %xmm8\n  movdqu 288(%rdi), %xmm9\n"
    "  movdqu 304(%rdi), %xmm10\n  movdqu 320(%rdi), %xmm11\n  movdqu 336(%rdi), %xmm12\n"
    "  movdqu 352(%rdi), %xmm13\n  movdqu 368(%rdi), %xmm14\n  movdqu 384(%rdi), %xmm15\n"
    "  movq 24(%rdi), %rbx\n  movq 40(%rdi), %rbp\n  movq 48(%rdi), %rsi\n"
    "  movq 96(%rdi), %r12\n  movq 104(%rdi), %r13\n  movq 112(%rdi), %r14\n  movq 120(%rdi), %r15\n"
    "  movq 128(%rdi), %rax\n  movq 32(%rdi), %rsp\n  movq 56(%rdi), %rdi\n  jmpq *%rax\n");

namespace host {
namespace {

template <typename T> T rd(uint64_t addr) {
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof v);
    return v;
}

struct RuntimeFunction { uint32_t begin, end, unwind; };

const RuntimeFunction* lookup_function(uint64_t rip) {
    const uint8_t* nt = g_image + rd<uint32_t>(reinterpret_cast<uint64_t>(g_image) + 0x3C);
    uint32_t rva = rd<uint32_t>(reinterpret_cast<uint64_t>(nt) + 24 + 136);  // PE32+ data directory 3
    uint32_t size = rd<uint32_t>(reinterpret_cast<uint64_t>(nt) + 24 + 140);
    auto* table = reinterpret_cast<const RuntimeFunction*>(g_image + rva);
    uint32_t off = static_cast<uint32_t>(rip - reinterpret_cast<uint64_t>(g_image));
    size_t lo = 0, hi = size / sizeof(RuntimeFunction);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (off < table[mid].begin) hi = mid;
        else if (off >= table[mid].end) lo = mid + 1;
        else return &table[mid];
    }
    return nullptr;
}

struct Unwound {
    uint64_t establisher = 0;
    uint64_t handler = 0;       // language handler address, 0 if none
    uint64_t handler_data = 0;  // what follows the handler RVA in UNWIND_INFO
};

// RtlVirtualUnwind for a frame stopped at a call site: ctx becomes the caller's state.
void apply_codes(Ctx& ctx, const RuntimeFunction* rf, uint64_t rip_offset, uint64_t& frame, Unwound* out) {
    uint64_t base = reinterpret_cast<uint64_t>(g_image);
    uint64_t ui = base + rf->unwind;
    uint8_t flags = rd<uint8_t>(ui) >> 3, count = rd<uint8_t>(ui + 2), fr = rd<uint8_t>(ui + 3);
    uint8_t frame_reg = fr & 0xF, frame_off = fr >> 4;
    if (out) {
        frame = frame_reg ? ctx.gpr[frame_reg] - 16ull * frame_off : ctx.gpr[4];
        out->establisher = frame;
    }
    auto code = [&](int i) { return rd<uint16_t>(ui + 4 + 2 * i); };
    for (int i = 0; i < count;) {
        uint16_t c = code(i);
        uint8_t at = c & 0xFF, op = (c >> 8) & 0xF, info = c >> 12;
        int slots = op == 1 ? (info ? 3 : 2) : op == 4 || op == 8 ? 2 : op == 5 || op == 9 ? 3 : 1;
        if (at > rip_offset) { i += slots; continue; }  // in the prolog, not executed yet
        uint64_t& rsp = ctx.gpr[4];
        switch (op) {
            case 0: ctx.gpr[info] = rd<uint64_t>(rsp); rsp += 8; break;                      // PUSH_NONVOL
            case 1: rsp += info ? (code(i + 1) | uint32_t(code(i + 2)) << 16) : code(i + 1) * 8u; break;
            case 2: rsp += info * 8u + 8; break;                                             // ALLOC_SMALL
            case 3: rsp = ctx.gpr[frame_reg] - 16ull * frame_off; break;                     // SET_FPREG
            case 4: ctx.gpr[info] = rd<uint64_t>(frame + code(i + 1) * 8u); break;           // SAVE_NONVOL
            case 5: ctx.gpr[info] = rd<uint64_t>(frame + (code(i + 1) | uint32_t(code(i + 2)) << 16)); break;
            case 8: std::memcpy(ctx.xmm[info], reinterpret_cast<void*>(frame + code(i + 1) * 16u), 16); break;
            case 9: std::memcpy(ctx.xmm[info], reinterpret_cast<void*>(frame + (code(i + 1) | uint32_t(code(i + 2)) << 16)), 16); break;
            default: fail("unwind: unsupported opcode " + std::to_string(op));
        }
        i += slots;
    }
    uint64_t tail = ui + 4 + 2 * ((count + 1) & ~1);
    if (flags & 4) {  // CHAININFO: the parent's codes, all executed
        apply_codes(ctx, reinterpret_cast<const RuntimeFunction*>(tail), ~0ull, frame, nullptr);
    } else if (out && (flags & 3)) {
        out->handler = base + rd<uint32_t>(tail);
        out->handler_data = tail + 4;
    }
}

Unwound virtual_unwind(Ctx& ctx) {
    Unwound u;
    const RuntimeFunction* rf = lookup_function(ctx.rip);
    if (rf) {
        uint64_t frame = 0;
        apply_codes(ctx, rf, ctx.rip - reinterpret_cast<uint64_t>(g_image) - rf->begin, frame, &u);
    } else {
        u.establisher = ctx.gpr[4];  // leaf function
    }
    ctx.rip = rd<uint64_t>(ctx.gpr[4]);
    ctx.gpr[4] += 8;
    return u;
}

// __CxxFrameHandler3 and the MSVC throw metadata; every pointer is an RVA from the image base.
struct FuncInfo { uint32_t magic; int32_t max_state; uint32_t unwind_map, n_try, try_map, n_ip, ip_map; int32_t unwind_help; uint32_t es, flags; };
struct UnwindEntry { int32_t to_state; uint32_t action; };
struct TryBlock { int32_t low, high, catch_high, n_catches; uint32_t handlers; };
struct HandlerType { uint32_t adjectives, type; int32_t catch_obj; uint32_t handler, frame; };
struct IpState { uint32_t ip; int32_t state; };
struct ThrowInfo { uint32_t attributes; int32_t unwind, forward_compat, catchable_types; };
struct CatchableType { uint32_t properties; int32_t type, mdisp, pdisp, vdisp, size, copy; };

template <typename T> const T* at_rva(uint32_t rva) { return reinterpret_cast<const T*>(g_image + rva); }

int state_at(const FuncInfo* fi, uint64_t rip) {
    uint32_t off = static_cast<uint32_t>(rip - reinterpret_cast<uint64_t>(g_image));
    int state = -1;
    const IpState* map = at_rva<IpState>(fi->ip_map);
    for (uint32_t i = 0; i < fi->n_ip && map[i].ip <= off; ++i) state = map[i].state;
    return state;
}

void local_unwind(const FuncInfo* fi, uint64_t frame, int state, int target) {
    const UnwindEntry* map = at_rva<UnwindEntry>(fi->unwind_map);
    while (state > target) {
        if (state >= fi->max_state) fail("unwind: bad EH state");
        if (map[state].action)
            call_guest(reinterpret_cast<uint64_t(MSABI*)(uint64_t, uint64_t)>(g_image + map[state].action), 0, frame);
        state = map[state].to_state;
    }
}

const char* type_name(uint32_t type_rva) { return reinterpret_cast<const char*>(g_image + type_rva + 16); }

const CatchableType* match(const HandlerType& h, const ThrowInfo* ti) {
    auto* array = at_rva<int32_t>(ti->catchable_types);
    for (int i = 0; i < array[0]; ++i) {
        const CatchableType* ct = at_rva<CatchableType>(array[i + 1]);
        if (!h.type) return ct;  // catch (...)
        if ((ct->properties & 2) && !(h.adjectives & 8)) continue;  // by reference only
        if (std::strcmp(type_name(h.type), type_name(ct->type)) == 0) return ct;
    }
    return nullptr;
}

void copy_catch_object(const HandlerType& h, const CatchableType* ct, void* object, uint64_t frame) {
    if (!h.type || !h.catch_obj) return;
    auto* obj = static_cast<uint8_t*>(object) + ct->mdisp;
    if (ct->pdisp >= 0) obj += ct->pdisp + rd<int32_t>(rd<uint64_t>(reinterpret_cast<uint64_t>(object) + ct->pdisp) + ct->vdisp);
    void* dst = reinterpret_cast<void*>(frame + h.catch_obj);
    if (h.adjectives & 8) std::memcpy(dst, &obj, 8);
    else if ((ct->properties & 1) || !ct->copy) std::memcpy(dst, obj, ct->size);
    else if (ct->properties & 4)
        call_guest(reinterpret_cast<void(MSABI*)(void*, void*, int)>(g_image + ct->copy), dst, obj, 1);
    else call_guest(reinterpret_cast<void(MSABI*)(void*, void*)>(g_image + ct->copy), dst, obj);
}

[[noreturn]] void unexpected_handler() { fail("unsupported: a language handler was called directly"); }
using HandlerFn = void();
void* const kFrameHandler3 = reinterpret_cast<void*>(&Thunk<HandlerFn, &unexpected_handler>::call);
void unexpected_handler2() { fail("unsupported: __C_specific_handler was called directly"); }
void* const kSpecificHandler = reinterpret_cast<void*>(&Thunk<HandlerFn, &unexpected_handler2>::call);

uint64_t resolve_stub(uint64_t p) {  // jmp [rip+disp32], as lld-link emits for imported handlers
    if (rd<uint8_t>(p) == 0xFF && rd<uint8_t>(p + 1) == 0x25) return rd<uint64_t>(p + 6 + rd<int32_t>(p + 2));
    return p;
}

struct Pending { const FuncInfo* fi; uint64_t frame; int state; };

[[noreturn]] void dispatch(void* object, const ThrowInfo* ti, Ctx ctx) {
    uint64_t lo = reinterpret_cast<uint64_t>(g_image);
    uint64_t hi = lo + rd<uint32_t>(lo + rd<uint32_t>(lo + 0x3C) + 24 + 56);  // SizeOfImage
    std::vector<Pending> passed;
    for (;;) {
        if (ctx.rip < lo || ctx.rip >= hi)
            fail("unhandled C++ exception (no catch in the game's frames)");
        Ctx here = ctx;
        Unwound u = virtual_unwind(ctx);
        if (!u.handler) continue;
        uint64_t target = resolve_stub(u.handler);
        if (target == reinterpret_cast<uint64_t>(kSpecificHandler)) continue;  // SEH __try: does not catch C++
        if (target != reinterpret_cast<uint64_t>(kFrameHandler3)) fail("unwind: unknown language handler");
        const FuncInfo* fi = at_rva<FuncInfo>(rd<uint32_t>(u.handler_data));
        int state = state_at(fi, here.rip);
        const TryBlock* tries = at_rva<TryBlock>(fi->try_map);
        for (uint32_t t = 0; t < fi->n_try; ++t) {
            if (state < tries[t].low || state > tries[t].high) continue;
            const HandlerType* hs = at_rva<HandlerType>(tries[t].handlers);
            for (int k = 0; k < tries[t].n_catches; ++k) {
                const CatchableType* ct = match(hs[k], ti);
                if (!ct) continue;
                copy_catch_object(hs[k], ct, object, u.establisher);
                for (const Pending& p : passed) local_unwind(p.fi, p.frame, p.state, -1);
                local_unwind(fi, u.establisher, state, tries[t].low);
                auto funclet = reinterpret_cast<uint64_t(MSABI*)(uint64_t, uint64_t)>(g_image + hs[k].handler);
                uint64_t resume = call_guest(funclet, 0, u.establisher);
                if (ti->unwind) call_guest(reinterpret_cast<void(MSABI*)(void*)>(g_image + ti->unwind), object);
                here.rip = resume;
                _thread_set_tsd_base(reinterpret_cast<uint64_t>(t_teb));
                host_resume(&here);
            }
        }
        passed.push_back({fi, u.establisher, state});
    }
}

}  // namespace

Exports eh_exports() {
    return {
        {"vcruntime140.dll!_CxxThrowException", reinterpret_cast<void*>(&host_cxx_throw_entry)},
        {"vcruntime140.dll!__CxxFrameHandler3", kFrameHandler3},
        {"vcruntime140.dll!__C_specific_handler", kSpecificHandler},
    };
}

}  // namespace host

extern "C" MSABI void host_cxx_throw(void* object, const void* throw_info, host::Ctx* ctx) {
    host::HostGs g;
    host::dispatch(object, static_cast<const host::ThrowInfo*>(throw_info), *ctx);
}

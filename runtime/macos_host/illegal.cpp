// x86 instructions the Xbox CPU runs and Rosetta 2 does not (isa_probe.c): ADX (adcx/adox), SSE4a insertq, clzero,
// clwb. Rosetta raises SIGILL on them. The first trap at a site proves the bytes are code, so the site is patched
// there and then: a jmp to a stub that does the same thing with instructions Rosetta has, or NOPs for clwb (a cache
// write-back hint). Later passes run the stub at full speed and never trap again. clzero is 3 bytes, too short for
// a jmp, so it is emulated on every trap.
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ucontext.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace {
struct Insn {
    const uint8_t* p;
    int len = 0;
    uint8_t pfx = 0, rex = 0, modrm = 0;
    int modrm_at = 0;  // offset of the ModRM byte
};

// Length of ModRM + SIB + displacement starting at m.
int modrm_len(const uint8_t* m) {
    int mod = m[0] >> 6, rm = m[0] & 7, n = 1;
    if (mod == 3) return 1;
    if (rm == 4) {
        ++n;
        if (mod == 0 && (m[1] & 7) == 5) n += 4;
    } else if (mod == 0 && rm == 5) {
        n += 4;  // rip-relative
    }
    return n + (mod == 1 ? 1 : mod == 2 ? 4 : 0);
}

bool rip_relative(uint8_t modrm) { return (modrm >> 6) == 0 && (modrm & 7) == 5; }

// A memory operand based on rsp: its displacement must grow by what the stub pushed before it.
bool rsp_based(const Insn& in) {
    const uint8_t* m = in.p + in.modrm_at;
    return (m[0] >> 6) != 3 && (m[0] & 7) == 4 && (m[1] & 7) == 4 && !(in.rex & 1);
}

// Stubs live within ±2 GB of their site, so a rel32 jmp reaches them both ways.
struct StubPage {
    uint8_t* base;
    size_t used;
};
std::vector<StubPage> g_pages;
constexpr size_t kStubPage = 1 << 16;
constexpr int64_t kReach = int64_t(1) << 30;

uint8_t* stub_space(const uint8_t* site, size_t n) {
    for (auto& pg : g_pages)
        if (pg.used + n <= kStubPage && pg.base - site < kReach && site - pg.base < kReach) {
            uint8_t* p = pg.base + pg.used;
            pg.used += (n + 15) & ~size_t(15);
            return p;
        }
    for (int64_t off : {-(int64_t(256) << 20), int64_t(256) << 20, -(int64_t(768) << 20), int64_t(768) << 20}) {
        auto hint = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(site) + off) & ~uintptr_t(0xFFFF));
        auto* p = static_cast<uint8_t*>(
            mmap(hint, kStubPage, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0));
        if (p == MAP_FAILED) continue;
        if (p - site < kReach && site - p < kReach) {
            g_pages.push_back({p, 0});
            return stub_space(site, n);
        }
        munmap(p, kStubPage);
    }
    return nullptr;
}

struct Emit {
    std::vector<uint8_t> b;
    Emit& operator()(std::initializer_list<uint8_t> v) {
        b.insert(b.end(), v);
        return *this;
    }
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i))); }
};

// adc with the original ModRM operand, rip-relative and rsp-relative displacements fixed for where it now runs.
// `at` is the stub address the adc starts at, `pushed` what the stub put on the stack before it.
void emit_adc(Emit& e, const Insn& in, const uint8_t* at, int pushed) {
    size_t start = e.b.size();
    if (in.rex) e({in.rex});
    e({0x13});
    const uint8_t* m = in.p + in.modrm_at;
    int ml = modrm_len(m);
    if (rip_relative(m[0])) {
        int32_t disp;
        std::memcpy(&disp, m + 1, 4);
        const uint8_t* target = in.p + in.len + disp;
        e({m[0]});
        const uint8_t* end = at + (e.b.size() - start) + 4;
        e.u32(uint32_t(target - end));
    } else if (rsp_based(in)) {
        int mod = m[0] >> 6;
        int32_t disp = mod == 1 ? int8_t(m[2]) : mod == 2 ? [&] { int32_t d; std::memcpy(&d, m + 2, 4); return d; }() : 0;
        e({uint8_t(0x80 | (m[0] & 0x3F)), m[1]});  // mod=2: disp32
        e.u32(uint32_t(disp + pushed));
    } else {
        e.b.insert(e.b.end(), m, m + ml);
    }
}

// adcx: dst += src + CF, only CF changes. adox: the same through OF. adc does the sum; the stub keeps the flags
// from before and copies in only the carry bit.
std::vector<uint8_t> adx_stub(const Insn& in, const uint8_t* at, bool ox) {
    Emit e;
    e({0x9C});                                                // pushfq
    if (ox) e({0x48, 0x0F, 0xBA, 0x24, 0x24, 0x0B});          // bt qword [rsp], 11 : CF = OF
    emit_adc(e, in, at + e.b.size(), 8);
    e({0x9C, 0x48, 0x0F, 0xBA, 0x24, 0x24, 0x00});            // pushfq; bt qword [rsp], 0
    if (ox) {
        e({0x72, 0x0B, 0x48, 0x81, 0x64, 0x24, 0x08, 0xFF, 0xF7, 0xFF, 0xFF,  // jc 1f; and [rsp+8], ~0x800
           0xEB, 0x09, 0x48, 0x81, 0x4C, 0x24, 0x08, 0x00, 0x08, 0x00, 0x00}); // jmp 2f; 1: or [rsp+8], 0x800
    } else {
        e({0x72, 0x08, 0x48, 0x83, 0x64, 0x24, 0x08, 0xFE,     // jc 1f; and [rsp+8], ~1
           0xEB, 0x06, 0x48, 0x83, 0x4C, 0x24, 0x08, 0x01});   // jmp 2f; 1: or [rsp+8], 1
    }
    e({0x48, 0x8D, 0x64, 0x24, 0x08, 0x9D});                  // 2: lea rsp, [rsp+8]; popfq
    return e.b;
}

// insertq dst, src, len, idx (immediate form): bits [idx, idx+len) of dst's low quadword take src's low len bits.
std::vector<uint8_t> insertq_stub(const Insn& in) {
    int dst = ((in.modrm >> 3) & 7) | (in.rex & 4 ? 8 : 0), src = (in.modrm & 7) | (in.rex & 1 ? 8 : 0);
    int len = in.p[in.len - 2] & 63, idx = in.p[in.len - 1] & 63;
    uint64_t low = len == 0 ? ~uint64_t(0) : (uint64_t(1) << len) - 1, keep = ~(low << idx);
    auto rex = [](int x) { return uint8_t(0x48 | (x & 8 ? 4 : 0)); };
    Emit e;
    e({0x9C, 0x50, 0x51});                                                    // pushfq; push rax; push rcx
    e({0x66, rex(dst), 0x0F, 0x7E, uint8_t(0xC0 | (dst & 7) << 3)});          // movq rax, xmm_dst
    e({0x66, rex(src), 0x0F, 0x7E, uint8_t(0xC1 | (src & 7) << 3)});          // movq rcx, xmm_src
    size_t low_at = e.b.size() + 3;
    e({0x48, 0x23, 0x0D}).u32(0);                                             // and rcx, [rip+low]
    if (idx) e({0x48, 0xC1, 0xE1, uint8_t(idx)});                             // shl rcx, idx
    size_t keep_at = e.b.size() + 3;
    e({0x48, 0x23, 0x05}).u32(0);                                             // and rax, [rip+keep]
    e({0x48, 0x09, 0xC8});                                                    // or rax, rcx
    e({0x66, rex(dst), 0x0F, 0x3A, 0x22, uint8_t(0xC0 | (dst & 7) << 3), 0}); // pinsrq xmm_dst, rax, 0
    e({0x59, 0x58, 0x9D});                                                    // pop rcx; pop rax; popfq
    // the caller appends the jmp back (5 bytes), then the two constants follow
    size_t consts = e.b.size() + 5;
    uint32_t d1 = uint32_t(consts - (low_at + 4)), d2 = uint32_t(consts + 8 - (keep_at + 4));
    std::memcpy(&e.b[low_at], &d1, 4);
    std::memcpy(&e.b[keep_at], &d2, 4);
    e.b.resize(consts);  // room for the jmp, written by the caller
    for (uint64_t v : {low, keep}) for (int i = 0; i < 8; ++i) e.b.push_back(uint8_t(v >> (8 * i)));
    return e.b;
}

enum class Kind { None, Adcx, Adox, Insertq, Clzero, Clwb };

Kind decode(const uint8_t* p, Insn& in) {
    in.p = p;
    int i = 0;
    while (p[i] == 0x66 || p[i] == 0xF2 || p[i] == 0xF3) in.pfx = p[i++];  // the last mandatory prefix counts
    if ((p[i] & 0xF0) == 0x40) in.rex = p[i++];
    if (p[i] != 0x0F) return Kind::None;
    if (p[i + 1] == 0x01 && p[i + 2] == 0xFC && !in.rex && !in.pfx) return in.len = i + 3, Kind::Clzero;
    if (p[i + 1] == 0x38 && p[i + 2] == 0xF6 && (in.pfx == 0x66 || in.pfx == 0xF3)) {
        in.modrm_at = i + 3, in.modrm = p[i + 3];
        in.len = i + 3 + modrm_len(p + i + 3);
        return in.pfx == 0x66 ? Kind::Adcx : Kind::Adox;
    }
    if (p[i + 1] == 0x78 && in.pfx == 0xF2 && (p[i + 2] >> 6) == 3) {
        in.modrm_at = i + 2, in.modrm = p[i + 2];
        in.len = i + 5;
        return Kind::Insertq;
    }
    if (p[i + 1] == 0xAE && in.pfx == 0x66 && ((p[i + 2] >> 3) & 7) == 6 && (p[i + 2] >> 6) != 3) {
        in.len = i + 2 + modrm_len(p + i + 2);
        return Kind::Clwb;
    }
    return Kind::None;
}

void write_code(uint8_t* site, const uint8_t* bytes, int n) {
    auto page = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(site) & ~uintptr_t(getpagesize() - 1));
    size_t span = site + n - page;
    mach_vm_address_t addr = reinterpret_cast<mach_vm_address_t>(site);
    mach_vm_size_t size;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object;
    int prot = PROT_READ | PROT_EXEC;
    if (mach_vm_region(mach_task_self(), &addr, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info),
                       &count, &object) == KERN_SUCCESS)
        prot = info.protection;  // VM_PROT_* bits match PROT_*
    mprotect(page, span, prot | PROT_WRITE);  // ponytail: a site spanning two pages takes the first one's protection
    // Another thread may reach the site while it changes: a 2-byte jmp-to-self holds it until the patch is whole.
    const uint8_t spin[2] = {0xEB, 0xFE};
    if (n > 2) {
        __atomic_store_n(reinterpret_cast<uint16_t*>(site), *reinterpret_cast<const uint16_t*>(spin), __ATOMIC_SEQ_CST);
        std::memcpy(site + 2, bytes + 2, n - 2);
    }
    uint16_t head;
    std::memcpy(&head, bytes, 2);
    __atomic_store_n(reinterpret_cast<uint16_t*>(site), head, __ATOMIC_SEQ_CST);
    mprotect(page, span, prot);
}

std::mutex g_lock;
}  // namespace

// Called from the SIGILL handler with the host's gs in place. True: the instruction is handled, resume at the
// context's rip (the patched site, or past an emulated clzero).
bool fix_illegal(ucontext_t* uc) {
    auto& ss = uc->uc_mcontext->__ss;
    auto* site = reinterpret_cast<uint8_t*>(ss.__rip);
    std::lock_guard<std::mutex> lock(g_lock);
    if (site[0] == 0xE9 || site[0] == 0x90 || (site[0] == 0xEB && site[1] == 0xFE)) return true;  // patched meanwhile
    Insn in;
    Kind k = decode(site, in);
    if (k == Kind::None) return false;
    if (k == Kind::Clzero) {  // zeroes the 64-byte line rax points into
        std::memset(reinterpret_cast<void*>(ss.__rax & ~uint64_t(63)), 0, 64);
        ss.__rip += in.len;
        return true;
    }
    std::vector<uint8_t> patch(in.len, 0x90);
    if (k != Kind::Clwb) {
        if (in.len < 5) return false;
        uint8_t* stub = stub_space(site, 128);
        if (!stub) return false;
        std::vector<uint8_t> code = k == Kind::Insertq ? insertq_stub(in) : adx_stub(in, stub, k == Kind::Adox);
        size_t jmp_at = k == Kind::Insertq ? code.size() - 21 : code.size();
        if (k != Kind::Insertq) code.resize(code.size() + 5);
        int32_t back = int32_t(site + in.len - (stub + jmp_at + 5));
        code[jmp_at] = 0xE9;
        std::memcpy(&code[jmp_at + 1], &back, 4);
        std::memcpy(stub, code.data(), code.size());
        int32_t to = int32_t(stub - (site + 5));
        patch[0] = 0xE9;
        std::memcpy(&patch[1], &to, 4);
    }
    write_code(site, patch.data(), in.len);
    return true;
}

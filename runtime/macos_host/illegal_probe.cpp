// Checks illegal.cpp: each instruction Rosetta 2 rejects runs twice (the trap that patches it, then the stub), and
// the result must equal what the instruction does on a real x86 CPU.
// Build: clang++ -arch x86_64 -std=c++17 illegal_probe.cpp illegal.cpp -o illegal_probe
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

bool fix_illegal(ucontext_t* uc);
static int traps;
static void on_ill(int, siginfo_t*, void* uc) {
    ++traps;
    if (!fix_illegal(static_cast<ucontext_t*>(uc))) { std::puts("unhandled SIGILL"); _exit(1); }
}

static uint8_t* g_code;
using Fn = unsigned __int128 (*)();  // rax = result, rdx = flags after
// Runs: push flags; popfq; <body>; pushfq; pop rdx; ret, with rax = a, rcx = b.
static unsigned __int128 run(std::vector<uint8_t> body, uint64_t a, uint64_t b, uint32_t flags) {
    std::vector<uint8_t> c = {0x48, 0xB8};
    for (int i = 0; i < 8; ++i) c.push_back(uint8_t(a >> 8 * i));
    c.insert(c.end(), {0x48, 0xB9});
    for (int i = 0; i < 8; ++i) c.push_back(uint8_t(b >> 8 * i));
    c.push_back(0x68);
    for (int i = 0; i < 4; ++i) c.push_back(uint8_t(flags >> 8 * i));
    c.push_back(0x9D);
    c.insert(c.end(), body.begin(), body.end());
    c.insert(c.end(), {0x9C, 0x5A, 0xC3});
    static size_t at = 0;
    uint8_t* p = g_code + at;
    at += 256;
    mprotect(g_code, 1 << 16, PROT_READ | PROT_WRITE);
    std::memcpy(p, c.data(), c.size());
    std::memcpy(p + 200, &b, 8);  // for the rip-relative case
    mprotect(g_code, 1 << 16, PROT_READ | PROT_EXEC);
    auto f = reinterpret_cast<Fn>(p);
    unsigned __int128 r1 = f(), r2 = f();
    if (r1 != r2) std::puts("  first and second run differ");
    return r1;
}

static int fails;
static void check(const char* what, uint64_t got, uint64_t want) {
    std::printf("%-40s %s  got %016llx want %016llx\n", what, got == want ? "ok  " : "FAIL", (unsigned long long)got,
                (unsigned long long)want);
    fails += got != want;
}

int main() {
    struct sigaction sa = {};
    sa.sa_sigaction = on_ill;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &sa, nullptr);
    g_code = static_cast<uint8_t*>(mmap(nullptr, 1 << 16, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0));

    const uint32_t kMask = 0x8D5;  // CF PF AF ZF SF OF
    for (uint32_t fin : {0x2u, 0x8D7u}) {
        uint64_t a = 0xFFFFFFFFFFFFFFF0, b = 0x20;
        int cf = fin & 1, of = (fin >> 11) & 1;
        char name[64];
        // adcx rax, rcx
        unsigned __int128 r = run({0x66, 0x48, 0x0F, 0x38, 0xF6, 0xC1}, a, b, fin);
        unsigned __int128 sum = (unsigned __int128)a + b + cf;
        std::snprintf(name, sizeof name, "adcx rax, rcx      flags %03x", fin);
        check(name, uint64_t(r), uint64_t(sum));
        check("  flags", uint64_t(r >> 64) & kMask, ((fin & ~1u) | uint32_t(sum >> 64)) & kMask);
        // adox rax, rcx
        r = run({0xF3, 0x48, 0x0F, 0x38, 0xF6, 0xC1}, a, b, fin);
        sum = (unsigned __int128)a + b + of;
        std::snprintf(name, sizeof name, "adox rax, rcx      flags %03x", fin);
        check(name, uint64_t(r), uint64_t(sum));
        check("  flags", uint64_t(r >> 64) & kMask, ((fin & ~0x800u) | uint32_t(sum >> 64) << 11) & kMask);
        // push rcx; adcx rax, [rsp]; pop rcx
        r = run({0x51, 0x66, 0x48, 0x0F, 0x38, 0xF6, 0x04, 0x24, 0x59}, a, b, fin);
        sum = (unsigned __int128)a + b + cf;
        std::snprintf(name, sizeof name, "adcx rax, [rsp]    flags %03x", fin);
        check(name, uint64_t(r), uint64_t(sum));
        // adox rax, [rip+164]: b stored at +200, the instruction ends at +36
        r = run({0xF3, 0x48, 0x0F, 0x38, 0xF6, 0x05, 164, 0, 0, 0}, a, b, fin);
        sum = (unsigned __int128)a + b + of;
        std::snprintf(name, sizeof name, "adox rax, [rip]    flags %03x", fin);
        check(name, uint64_t(r), uint64_t(sum));
    }
    // insertq: dst xmm0 <- rax, src xmm1 <- rcx; and xmm9/xmm8 for the REX form
    struct { int len, idx; } iq[] = {{16, 16}, {8, 0}, {8, 24}, {16, 48}, {0, 0}};
    for (auto [len, idx] : iq) {
        uint64_t a = 0x1122334455667788, b = 0xA1B2C3D4E5F60718;
        uint64_t low = len == 0 ? ~0ull : (1ull << len) - 1, want = (a & ~(low << idx)) | ((b & low) << idx);
        char name[64];
        unsigned __int128 r = run({0x66, 0x48, 0x0F, 0x6E, 0xC0, 0x66, 0x48, 0x0F, 0x6E, 0xC9,  // movq xmm0,rax; xmm1,rcx
                                   0xF2, 0x0F, 0x78, 0xC1, uint8_t(len), uint8_t(idx),           // insertq xmm0, xmm1
                                   0x66, 0x48, 0x0F, 0x7E, 0xC0},                                 // movq rax, xmm0
                                  a, b, 2);
        std::snprintf(name, sizeof name, "insertq xmm0, xmm1, %d, %d", len, idx);
        check(name, uint64_t(r), want);
        r = run({0x66, 0x4C, 0x0F, 0x6E, 0xC8, 0x66, 0x4C, 0x0F, 0x6E, 0xC1,  // movq xmm9,rax; xmm8,rcx
                 0xF2, 0x45, 0x0F, 0x78, 0xC8, uint8_t(len), uint8_t(idx),     // insertq xmm9, xmm8
                 0x66, 0x4C, 0x0F, 0x7E, 0xC8},                                 // movq rax, xmm9
                a, b, 0x8D7);
        std::snprintf(name, sizeof name, "insertq xmm9, xmm8, %d, %d", len, idx);
        check(name, uint64_t(r), want);
        check("  flags kept", uint64_t(r >> 64) & kMask, 0x8D5);
    }
    // clzero zeroes the 64-byte line rax points into; clwb [rcx] does nothing visible
    alignas(64) static uint8_t buf[192];
    std::memset(buf, 0xAB, sizeof buf);
    run({0x0F, 0x01, 0xFC}, uint64_t(buf + 64 + 5), 0, 2);
    int zero = 0, kept = 0;
    for (int i = 0; i < 192; ++i) (i >= 64 && i < 128 ? zero : kept) += i >= 64 && i < 128 ? buf[i] == 0 : buf[i] == 0xAB;
    check("clzero", uint64_t(zero) << 32 | kept, uint64_t(64) << 32 | 128);
    run({0x66, 0x0F, 0xAE, 0x31}, 0, uint64_t(buf), 2);
    check("clwb [rcx]", buf[0], 0xAB);
    std::printf("traps %d, %s\n", traps, fails ? "FAILED" : "all ok");
    return fails != 0;
}

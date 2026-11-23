// Which x86-64 extensions of the Xbox Series CPU (Zen 2) Rosetta 2 runs, and which CPUID reports.
// Build: clang -arch x86_64 -O1 isa_probe.c -o isa_probe   Run: ./isa_probe  (and ROSETTA_ADVERTISE_AVX=1 ./isa_probe)
#include <cpuid.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static sigjmp_buf jb;
static void on_sig(int s) { siglongjmp(jb, s); }

// Each test runs the instruction and returns 1 if the result is the expected one.
static int t_popcnt(void) { uint64_t r, x = 0xF0F0; __asm__ volatile("popcnt %1,%0" : "=r"(r) : "r"(x)); return r == 8; }
static int t_lzcnt(void)  { uint64_t r, x = 1; __asm__ volatile("lzcnt %1,%0" : "=r"(r) : "r"(x)); return r == 63; }
static int t_movbe(void)  { uint32_t r, x = 0x11223344; __asm__ volatile("movbe %1,%0" : "=r"(r) : "m"(x)); return r == 0x44332211; }
static int t_bmi1(void)   { uint64_t r, x = 0xF0; __asm__ volatile("tzcnt %1,%0" : "=r"(r) : "r"(x)); uint64_t a; __asm__ volatile("andn %2,%1,%0" : "=r"(a) : "r"(0x0Full), "r"(0xFFull)); return r == 4 && a == 0xF0; }
static int t_bmi2(void)   { uint64_t r; __asm__ volatile("pdep %2,%1,%0" : "=r"(r) : "r"(0b101ull), "r"(0b11010ull)); return r == 0b10010; }
static int t_adx(void)    { uint64_t a = ~0ull, b = 1; __asm__ volatile("clc; adcx %1,%0; adcx %1,%0" : "+r"(a) : "r"(b) : "cc"); return a == 1; }
static int t_avx(void)    { float o[8], i[8] = {1,2,3,4,5,6,7,8}; __asm__ volatile("vmovups %1,%%ymm0; vaddps %%ymm0,%%ymm0,%%ymm0; vmovups %%ymm0,%0; vzeroupper" : "=m"(o) : "m"(i) : "xmm0"); return o[7] == 16; }
static int t_avx2(void)   { int o[8], i[8] = {1,2,3,4,5,6,7,8}; __asm__ volatile("vmovdqu %1,%%ymm0; vpmulld %%ymm0,%%ymm0,%%ymm0; vpermq $0x1b,%%ymm0,%%ymm0; vmovdqu %%ymm0,%0; vzeroupper" : "=m"(o) : "m"(i) : "xmm0"); return o[0] == 49 && o[7] == 4; }
static int t_fma(void)    { float o[8], a[8] = {2,2,2,2,2,2,2,2}, c[8] = {1,1,1,1,1,1,1,1}; __asm__ volatile("vmovups %1,%%ymm0; vmovups %2,%%ymm1; vfmadd213ps %%ymm1,%%ymm0,%%ymm0; vmovups %%ymm0,%0; vzeroupper" : "=m"(o) : "m"(a), "m"(c) : "xmm0", "xmm1"); return o[3] == 5; }
static int t_f16c(void)   { uint16_t h[4] = {0x3C00, 0x4000, 0xC000, 0}; float f[4]; __asm__ volatile("vcvtph2ps %1,%%xmm0; vmovups %%xmm0,%0" : "=m"(f) : "m"(h) : "xmm0"); return f[0] == 1 && f[1] == 2 && f[2] == -2; }
static int t_aes(void)    { __asm__ volatile("pxor %%xmm0,%%xmm0; aesenc %%xmm0,%%xmm0" ::: "xmm0"); return 1; }
static int t_pclmul(void) { __asm__ volatile("pxor %%xmm0,%%xmm0; pclmulqdq $0,%%xmm0,%%xmm0" ::: "xmm0"); return 1; }
static int t_sha(void)    { __asm__ volatile("pxor %%xmm0,%%xmm0; sha256msg1 %%xmm0,%%xmm0" ::: "xmm0"); return 1; }
static int t_rdrand(void) { uint64_t r; unsigned char ok; __asm__ volatile("rdrand %0; setc %1" : "=r"(r), "=qm"(ok)); return ok; }
static int t_rdseed(void) { uint64_t r; unsigned char ok; __asm__ volatile("rdseed %0; setc %1" : "=r"(r), "=qm"(ok)); return ok; }
static int t_sse4a(void)  { __asm__ volatile("pxor %%xmm0,%%xmm0; extrq $8,$0,%%xmm0" ::: "xmm0"); return 1; }
static int t_xsave(void)  { uint32_t lo, hi; __asm__ volatile("xor %%ecx,%%ecx; xgetbv" : "=a"(lo), "=d"(hi) :: "ecx"); return (lo & 6) == 6; }
static int t_clwb(void)   { static char line[64]; __asm__ volatile("clwb %0" :: "m"(line)); return 1; }
static int t_clzero(void) { static char line[64] __attribute__((aligned(64))); __asm__ volatile("clzero" :: "a"(line) : "memory"); return 1; }
static int t_avx512(void) { __asm__ volatile("vpxorq %%zmm0,%%zmm0,%%zmm0" ::: "xmm0"); return 1; }

struct T { const char* name; int (*fn)(void); int cpuid_leaf, reg, bit; };  // reg: 0 ebx 1 ecx 2 edx
static const struct T tests[] = {
    {"POPCNT", t_popcnt, 1, 1, 23},   {"LZCNT", t_lzcnt, 0x80000001, 1, 5}, {"MOVBE", t_movbe, 1, 1, 22},
    {"BMI1", t_bmi1, 7, 0, 3},        {"BMI2", t_bmi2, 7, 0, 8},            {"ADX", t_adx, 7, 0, 19},
    {"AVX", t_avx, 1, 1, 28},         {"AVX2", t_avx2, 7, 0, 5},            {"FMA3", t_fma, 1, 1, 12},
    {"F16C", t_f16c, 1, 1, 29},       {"AES", t_aes, 1, 1, 25},             {"PCLMULQDQ", t_pclmul, 1, 1, 1},
    {"SHA", t_sha, 7, 0, 29},         {"RDRAND", t_rdrand, 1, 1, 30},       {"RDSEED", t_rdseed, 7, 0, 18},
    {"XSAVE/XGETBV", t_xsave, 1, 1, 27}, {"CLWB", t_clwb, 7, 0, 24},       {"SSE4a (AMD)", t_sse4a, 0x80000001, 1, 6},
    {"CLZERO (AMD)", t_clzero, 0x80000008, 0, 0}, {"AVX-512 (not on Zen 2)", t_avx512, 7, 0, 16},
};

static int cpuid_bit(const struct T* t) {
    unsigned r[4] = {0};
    if (!__get_cpuid_count(t->cpuid_leaf, 0, &r[0], &r[1], &r[2], &r[3])) return 0;
    unsigned v = t->reg == 0 ? r[1] : t->reg == 1 ? r[2] : r[3];
    return (v >> t->bit) & 1;
}

int main(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sig;
    sigaction(SIGILL, &sa, 0);
    sigaction(SIGSEGV, &sa, 0);
    sigaction(SIGBUS, &sa, 0);
    printf("%-24s %-8s %s\n", "extension", "CPUID", "runs");
    for (size_t i = 0; i < sizeof tests / sizeof *tests; i++) {
        const struct T* t = &tests[i];
        const char* res;
        int s = sigsetjmp(jb, 1);
        if (s) res = s == SIGILL ? "SIGILL" : "crash";
        else res = t->fn() ? "ok" : "WRONG RESULT";
        printf("%-24s %-8s %s\n", t->name, cpuid_bit(t) ? "yes" : "no", res);
    }
    return 0;
}

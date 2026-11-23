// Does Rosetta 2 give x86 results, not ARM ones, where the two differ? Float corner cases, memory ordering
// between threads. Build: clang -arch x86_64 -O1 -pthread isa_semantics.c
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fails;
static void check(const char* what, int ok, const char* detail) {
    printf("%-44s %s  %s\n", what, ok ? "x86" : "DIFFERS", detail);
    fails += !ok;
}
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float fl(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t get_mxcsr(void) { uint32_t m; __asm__ volatile("stmxcsr %0" : "=m"(m)); return m; }
static void set_mxcsr(uint32_t m) { __asm__ volatile("ldmxcsr %0" :: "m"(m)); }

static float add(float a, float b) { __asm__ volatile("addss %1,%0" : "+x"(a) : "x"(b)); return a; }
static float mul(float a, float b) { __asm__ volatile("mulss %1,%0" : "+x"(a) : "x"(b)); return a; }
static float divs(float a, float b) { __asm__ volatile("divss %1,%0" : "+x"(a) : "x"(b)); return a; }
static float mins(float a, float b) { __asm__ volatile("minss %1,%0" : "+x"(a) : "x"(b)); return a; }
static float fma1(float a, float b, float c) { __asm__ volatile("vfmadd213ss %2,%1,%0" : "+x"(a) : "x"(b), "x"(c)); return a; }
static int32_t cvtt(float a) { int32_t r; __asm__ volatile("cvttss2si %1,%0" : "=r"(r) : "x"(a)); return r; }

static void floats(void) {
    char d[96];
    uint32_t m0 = get_mxcsr();

    float a = fl(0x3F800001);  // 1 + 2^-23
    float r = fma1(a, a, -fl(0x3F800002));  // fused keeps the 2^-46 term that mul then add rounds away
    snprintf(d, sizeof d, "got %a, fused 0x1p-46", r);
    check("FMA is fused (one rounding)", r == 0x1p-46f, d);

    uint32_t n = bits(mul(0.0f, __builtin_inff()));
    snprintf(d, sizeof d, "0*inf = %08x, x86 ffc00000, ARM 7fc00000", n);
    check("default NaN has the x86 sign", n == 0xFFC00000, d);

    n = bits(add(fl(0x7FC00001), fl(0x7FC00002)));
    snprintf(d, sizeof d, "NaN1+NaN2 = %08x, x86 keeps the first", n);
    check("NaN propagation picks the first operand", n == 0x7FC00001, d);

    n = bits(add(fl(0x7F800001), 1.0f));
    snprintf(d, sizeof d, "sNaN+1 = %08x, x86 quiets to 7fc00001", n);
    check("signaling NaN is quieted, payload kept", n == 0x7FC00001, d);

    float mn = mins(__builtin_nanf(""), 1.0f), mz = mins(0.0f, -0.0f);
    snprintf(d, sizeof d, "min(NaN,1)=%g min(0,-0)=%08x", mn, bits(mz));
    check("minss returns the second operand on NaN/0", mn == 1.0f && bits(mz) == 0x80000000, d);

    int32_t c = cvtt(3e9f), cn = cvtt(__builtin_nanf(""));
    snprintf(d, sizeof d, "(int)3e9=%08x (int)NaN=%08x, x86 80000000", (uint32_t)c, (uint32_t)cn);
    check("out-of-range float->int gives 0x80000000", (uint32_t)c == 0x80000000 && (uint32_t)cn == 0x80000000, d);

    set_mxcsr(m0 | 0x8000);  // FTZ
    r = mul(fl(0x00800000), 0.5f);
    set_mxcsr(m0);
    snprintf(d, sizeof d, "min_normal*0.5 = %08x", bits(r));
    check("MXCSR flush-to-zero is honoured", bits(r) == 0, d);

    set_mxcsr(m0 | 0x40);  // DAZ
    r = add(fl(0x00000001), 0.0f);
    set_mxcsr(m0);
    snprintf(d, sizeof d, "denormal+0 = %08x", bits(r));
    check("MXCSR denormals-are-zero is honoured", bits(r) == 0, d);

    set_mxcsr((m0 & ~0x6000u) | 0x4000);  // round up
    float up = divs(1.0f, 3.0f);
    set_mxcsr((m0 & ~0x6000u) | 0x6000);  // toward zero
    float dn = divs(1.0f, 3.0f);
    set_mxcsr(m0);
    snprintf(d, sizeof d, "1/3 up=%08x zero=%08x", bits(up), bits(dn));
    check("MXCSR rounding modes are honoured", bits(up) == bits(dn) + 1, d);

    set_mxcsr(m0 & ~0x3Fu);
    divs(1.0f, 0.0f);
    uint32_t flags = get_mxcsr() & 0x3F;
    set_mxcsr(m0);
    snprintf(d, sizeof d, "flags after 1/0 = %02x, x86 04 (ZE)", flags);
    check("MXCSR exception flags are set", flags & 4, d);
}

// Message passing: thread A writes data then flag; if B sees the flag it must see the data. x86 (TSO)
// guarantees it with plain stores and loads; ARM's own model does not.
enum { kIters = 2000000 };
static volatile int data, flag, go;
static long mp_bad;
static void* mp_writer(void* _) {
    for (int i = 1; i <= kIters; i++) {
        while (go != i) {}
        data = i;
        flag = i;
    }
    return 0;
}
static void memory_order(void) {
    pthread_t t;
    pthread_create(&t, 0, mp_writer, 0);
    for (int i = 1; i <= kIters; i++) {
        go = i;
        while (flag != i) {
            if (flag == i) break;
        }
        if (data != i) mp_bad++;
    }
    pthread_join(t, 0);
    char d[96];
    snprintf(d, sizeof d, "%ld of %d rounds saw the flag without the data", mp_bad, kIters);
    check("stores stay ordered between threads (TSO)", mp_bad == 0, d);
}

int main(void) {
    floats();
    memory_order();
    printf("%d check(s) differ from x86\n", fails);
    return fails != 0;
}

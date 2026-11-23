// Prints what CPUID reports to a guest: the host re-execs under Rosetta with ROSETTA_ADVERTISE_AVX=1 so AVX2 shows.
#include <intrin.h>
#include <stdio.h>
int main(void) {
    int r[4];
    __cpuid(r, 1);
    int avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1;
    __cpuidex(r, 7, 0);
    int avx2 = (r[1] >> 5) & 1;
    printf("avx %d fma %d avx2 %d\n", avx, fma, avx2);
    return avx2 ? 0 : 1;
}

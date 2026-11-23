// Vector throughput under Rosetta 2: SSE vs 256-bit AVX, with 8 independent chains so latency does not cap it.
// Build: clang -arch x86_64 -O1 isa_throughput.c   Run: ROSETTA_ADVERTISE_AVX=1 ./a.out
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){ const long n=100000000; double t0;
#define RUN(name, flops, body) t0=now(); __asm__ volatile("mov %0,%%rcx; 1:" body "dec %%rcx; jnz 1b; vzeroupper"::"r"(n):"rcx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","cc"); printf("%-34s %6.1f GFLOP/s\n", name, (double)flops*n/(now()-t0)/1e9);
RUN("SSE addps, 8 independent", 8*4, "addps %%xmm8,%%xmm0; addps %%xmm8,%%xmm1; addps %%xmm8,%%xmm2; addps %%xmm8,%%xmm3; addps %%xmm8,%%xmm4; addps %%xmm8,%%xmm5; addps %%xmm8,%%xmm6; addps %%xmm8,%%xmm7;")
RUN("AVX vaddps ymm, 8 independent", 8*8, "vaddps %%ymm8,%%ymm0,%%ymm0; vaddps %%ymm8,%%ymm1,%%ymm1; vaddps %%ymm8,%%ymm2,%%ymm2; vaddps %%ymm8,%%ymm3,%%ymm3; vaddps %%ymm8,%%ymm4,%%ymm4; vaddps %%ymm8,%%ymm5,%%ymm5; vaddps %%ymm8,%%ymm6,%%ymm6; vaddps %%ymm8,%%ymm7,%%ymm7;")
RUN("AVX vaddps xmm (VEX-128), 8 indep", 8*4, "vaddps %%xmm8,%%xmm0,%%xmm0; vaddps %%xmm8,%%xmm1,%%xmm1; vaddps %%xmm8,%%xmm2,%%xmm2; vaddps %%xmm8,%%xmm3,%%xmm3; vaddps %%xmm8,%%xmm4,%%xmm4; vaddps %%xmm8,%%xmm5,%%xmm5; vaddps %%xmm8,%%xmm6,%%xmm6; vaddps %%xmm8,%%xmm7,%%xmm7;")
RUN("AVX2 vfmadd ymm, 8 indep", 8*16, "vfmadd231ps %%ymm8,%%ymm8,%%ymm0; vfmadd231ps %%ymm8,%%ymm8,%%ymm1; vfmadd231ps %%ymm8,%%ymm8,%%ymm2; vfmadd231ps %%ymm8,%%ymm8,%%ymm3; vfmadd231ps %%ymm8,%%ymm8,%%ymm4; vfmadd231ps %%ymm8,%%ymm8,%%ymm5; vfmadd231ps %%ymm8,%%ymm8,%%ymm6; vfmadd231ps %%ymm8,%%ymm8,%%ymm7;")
return 0;}

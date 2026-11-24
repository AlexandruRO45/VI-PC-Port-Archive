// Wave64 emulation (src/wave64_emulate.hlsli) over one 64-thread group: the reads of
// tests/wave_shuffle.hlsl at a width of 64 whatever the device's waves are, plus a first-lane read, a sum
// and a ballot. WAVE_SIZE forces the native width (SM 6.6). SKEW is the negative control: the affine
// read asks for the next lane, which the CPU check must report on every thread.
#include "wave64_emulate.hlsli"
RWByteAddressBuffer output : register(u0);

uint Value(uint t) { return t * 2654435761u + 12345u; }

#ifdef WAVE_SIZE
[WaveSize(WAVE_SIZE)]
#endif
[RootSignature("UAV(u0)")] [numthreads(64,1,1)]
void Emulate(uint t : SV_GroupIndex)
{
    uint v = Value(t);
#ifdef SKEW
    uint gathered = Wave64ReadLaneAt(v, t, (t * 7 + 3) % 64 + 1);
#else
    uint gathered = Wave64ReadLaneAt(v, t, (t * 7 + 3) % 64);
#endif
    uint rotated = Wave64ReadLaneAt(v, t, (t + 1) % 64);
    uint mirrored = Wave64ReadLaneAt(v, t, 63 - t);
    uint first = Wave64ReadLaneFirst(v, t);
    uint sum = Wave64ActiveSum(v, t);
    uint2 ballot = Wave64ActiveBallot(t % 3 == 0, t);
    output.Store4(t * 32, uint4(gathered, rotated, mirrored, first));
    output.Store4(t * 32 + 16, uint4(sum, ballot, 0));
}

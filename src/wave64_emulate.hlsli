// 64-lane wave operations for one 64-thread group, built from group-shared memory and group barriers
// instead of wave intrinsics. The result is the 64-lane result on any native wave width (WARP's 8,
// NVIDIA's 32) and does not depend on WARP 1.0.22's forced-wave64 WaveReadLaneAt (issue #145).
// Requirements: `lane` is SV_GroupIndex of a group of exactly 64 threads, and every call is reached by
// all 64 threads in uniform control flow, because each call waits on group barriers. All 64 lanes are
// active, so the first active lane is lane 0. Values are uint.
groupshared uint g_wave64[64];

uint Wave64ReadLaneAt(uint value, uint lane, uint source)
{
    g_wave64[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    uint result = g_wave64[source & 63];
    GroupMemoryBarrierWithGroupSync();  // the next call reuses the slots
    return result;
}

uint Wave64ReadLaneFirst(uint value, uint lane) { return Wave64ReadLaneAt(value, lane, 0); }

uint Wave64ActiveSum(uint value, uint lane)
{
    g_wave64[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    uint sum = 0;
    for (uint i = 0; i < 64; ++i) sum += g_wave64[i];
    GroupMemoryBarrierWithGroupSync();
    return sum;
}

// Lanes 0..31 in .x and 32..63 in .y, like WaveActiveBallot's first two words.
uint2 Wave64ActiveBallot(bool predicate, uint lane)
{
    g_wave64[lane] = predicate ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    uint2 mask = 0;
    for (uint i = 0; i < 32; ++i) {
        mask.x |= g_wave64[i] << i;
        mask.y |= g_wave64[i + 32] << i;
    }
    GroupMemoryBarrierWithGroupSync();
    return mask;
}

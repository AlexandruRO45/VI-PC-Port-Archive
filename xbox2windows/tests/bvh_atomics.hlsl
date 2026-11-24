ByteAddressBuffer points : register(t0);
RWByteAddressBuffer result : register(u0);
cbuffer Settings : register(b0) { uint count; uint outputOffset; uint bucketCount; }
#define LAYOUT "SRV(t0), UAV(u0), RootConstants(num32BitConstants=3,b0)"

[RootSignature(LAYOUT)] [numthreads(32, 1, 1)]
void InitBounds(uint i : SV_DispatchThreadID)
{
    if (i >= bucketCount) return;
    result.Store3(i * 32, 0x7f7fffffu.xxx);
    result.Store3(i * 32 + 12, 0xff7fffffu.xxx);
}

// Compare finite IEEE-754 values, preserving raw float bits in the six bound fields.
// This also gives a deterministic order to signed zeros.
uint orderKey(uint bits)
{
    return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

void extendBound(uint address, uint value, bool minimum)
{
    uint seen = minimum ? 0x7f7fffffu : 0xff7fffffu;
    [allow_uav_condition]
    while (minimum ? orderKey(value) < orderKey(seen) : orderKey(value) > orderKey(seen)) {
        uint previous;
        result.InterlockedCompareExchange(address, seen, value, previous);
        if (previous == seen) break;
        seen = previous;
    }
}

[RootSignature(LAYOUT)]
[numthreads(32, 1, 1)]
void AtomicBounds(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= count) return;
    uint4 sampleBits = points.Load4(id.x * 16);
    if (any((sampleBits.xyz & 0x7f800000u) == 0x7f800000u)) return;
    uint base = sampleBits.w * 32;
    [unroll] for (uint axis = 0; axis < 3; ++axis) {
        extendBound(base + 4 * axis, sampleBits[axis], true);
        extendBound(base + 12 + 4 * axis, sampleBits[axis], false);
    }
    uint beforeAdd, beforeSub;
    result.InterlockedAdd(base + 24, 1u, beforeAdd);
    result.InterlockedAdd(base + 28, 0xffffffffu, beforeSub);
    result.Store2(outputOffset + id.x * 8, uint2(beforeAdd, beforeSub));
}

// Axis-aligned forward rays with a closed interval at the origin.
[RootSignature(LAYOUT)]
[numthreads(32, 1, 1)]
void QueryBounds(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= count) return;
    uint4 ray = points.Load4(id.x * 16);
    float3 origin = asfloat(ray.xyz);
    uint axis = ray.w % 3, mask = 0;
    for (uint b = 0; b < bucketCount; ++b) {
        float3 lo = asfloat(result.Load3(b * 32));
        float3 hi = asfloat(result.Load3(b * 32 + 12));
        if (any(lo > hi)) continue;
        bool3 slab = (lo <= origin) & (origin <= hi);
        slab[axis] = ray.w < 3 ? origin[axis] <= hi[axis] : origin[axis] >= lo[axis];
        if (all(slab)) mask |= 1u << b;
    }
    result.Store(outputOffset + id.x * 4, mask);
}

#define ROOT "SRV(t0),SRV(t1),UAV(u0),RootConstants(num32BitConstants=4,b0)"
ByteAddressBuffer data : register(t0);
StructuredBuffer<uint4> queries : register(t1);
RWByteAddressBuffer output : register(u0);
cbuffer Settings : register(b0) { uint baseLo, baseHi, extent, epoch; }

[RootSignature("UAV(u0),RootConstants(num32BitConstants=4,b0)")] [numthreads(64,1,1)]
void Produce(uint i : SV_DispatchThreadID)
{
    uint value = (i * 0x9e3779b9u) ^ (epoch * 0x51f3a19du);
    output.Store(4*i, i == 3 ? (epoch == 1 ? 37u : 97u) : value);
}

uint4 readAddress(uint2 address, uint width, out bool valid)
{
    uint offset = address.x - baseLo;
    uint high = address.y - baseHi - uint(address.x < baseLo);
    valid = high == 0 && (offset & 3) == 0 && width >= 1 && width <= 4 &&
                 extent >= width * 4 && offset <= extent - width * 4;
    uint4 v = 0xdeadbeefu.xxxx;
    if (valid) {
        switch (width) {
        case 1: v.x = data.Load(offset); break;
        case 2: v.xy = data.Load2(offset); break;
        case 3: v.xyz = data.Load3(offset); break;
        case 4: v = data.Load4(offset); break;
        }
    }
    return v;
}

[RootSignature(ROOT)] [numthreads(64,1,1)]
void ReadValues(uint i : SV_DispatchThreadID)
{
    uint4 q = queries[i];
    bool valid;
    uint4 v = readAddress(q.xy, q.z, valid);
    uint uniformOffset = ((i / 64) << 8) | 44u;
    uint sharedWord = WaveReadLaneFirst(data.Load(uniformOffset));
    uint limit = data.Load(12);
    uint copied = 0xdeadbeefu;
    if (i < limit) copied = data.Load(4*i);
    output.Store4(32*i, v);
    output.Store4(32*i+16, uint4(valid, sharedWord, limit, copied));
}

groupshared uint values[64], scans[64], votes[64];
void scanHalf(uint logical, uint value)
{
    scans[logical] = value;
    GroupMemoryBarrierWithGroupSync();
    // Five candidate routing stages within each logical set of 32.
    for (uint step=1; step<=16; step*=2) {
        uint lane = logical & 31;
        uint mask = 30 * step;
        uint from = (lane & (mask & 31)) | ((mask >> 5) & 31);
        uint addend = scans[(logical & 32) + from];
        GroupMemoryBarrierWithGroupSync();
        if ((lane & step) != 0) scans[logical] += addend;
        GroupMemoryBarrierWithGroupSync();
    }
}
#ifdef REQUIRE_WAVE64
[WaveSize(64)]
#endif
[RootSignature(ROOT)] [numthreads(64,1,1)]
void WaveValues(uint t : SV_GroupIndex, uint g : SV_GroupID)
{
    uint logical = (g & 1) ? 63-t : t;
    uint v = queries[g*64+logical].x;
    bool vote = queries[g*64+logical].y != 0;
    values[logical] = v;
    votes[logical] = vote;
    scanHalf(logical, v);
    uint2 ballot = 0;
    for (uint j=0; j<64; ++j) ballot[j/32] |= votes[j] << (j%32);
    uint rank = 0;
    for (uint k=0; k<logical; ++k) rank += votes[k];
    uint p = (g*64+logical)*128;
    output.Store4(p, uint4(scans[logical], scans[logical]-v, scans[(logical & 32)+31], values[(logical*7+3)%64]));
    output.Store4(p+16, uint4(ballot, rank, values[0]));
    // Independently expose the actual native lane assignment for the CPU oracle.
    uint nativeLane = WaveGetLaneIndex();
    uint last = WaveActiveMax(nativeLane);
    output.Store4(p+32,uint4(nativeLane,WaveGetLaneCount(),WaveReadLaneFirst(logical),WaveReadLaneAt(v,last)));
    output.Store4(p+48,WaveActiveBallot(vote));
    output.Store4(p+64,uint4(WavePrefixCountBits(vote),WaveActiveMin(v),WavePrefixSum(v),WaveReadLaneFirst(v)));
    uint4 subset = 0xffffffffu.xxxx;
    if (vote) {
        subset = uint4(WavePrefixCountBits(true),WaveActiveMin(v),WaveReadLaneFirst(v),WaveActiveCountBits(true));
    }
    output.Store4(p+80,subset);
    uint4 routed = 0xffffffffu.xxxx;
    uint width = WaveGetLaneCount();
    if (WaveActiveCountBits(true) == width) {
        uint sum = v;
        for (uint step=1; step<=16 && step<width; step*=2) {
            uint mask=30*step;
            uint from=(nativeLane & ~31u) | (nativeLane & (mask & 31)) | ((mask >> 5) & 31);
            uint other=WaveReadLaneAt(sum,from);
            if (nativeLane & step) sum+=other;
        }
        routed=uint4(sum, WaveReadLaneAt(v,(nativeLane*7+3)%width),width,1);
    }
    output.Store4(p+96,routed);
    // Mirrors exchange the middle pair (31,32). The half-local mirror is the fault control.
    uint nativeMirror = 0xffffffffu;
    if (width == 64 && WaveActiveCountBits(true) == 64)
        nativeMirror = WaveReadLaneAt(v, 63 - nativeLane);
    output.Store4(p+112, uint4(values[63-logical], values[logical ^ 31], nativeMirror,
                               scans[logical] + (logical >= 32 ? scans[31] : 0)));
}

// One workgroup per ray. The synthetic root has 64 bounds and each child
// points at a separate triangle record using a logical 64-bit address.
[RootSignature("SRV(t1),UAV(u0)")] [numthreads(64,1,1)]
void WriteScene(uint i : SV_DispatchThreadID)
{
    output.Store4(i * 16, queries[i]);
}

uint2 offsetAddress(uint2 address, uint bytes)
{
    uint lo = address.x + bytes;
    return uint2(lo, address.y + uint(lo < address.x));
}

// Inputs are source-built 32-byte records. The six coordinate/metadata words
// may start at any byte in the first dword. The final dword holds the mask.
uint alignedField(uint low, uint high, uint shift)
{
    if (shift == 0) return low;
    return (low >> shift) | (high << (32 - shift));
}

[RootSignature("SRV(t0),UAV(u0),RootConstants(num32BitConstants=4,b0)")] [numthreads(32,1,1)]
void ExpandLeaves(uint i : SV_DispatchThreadID)
{
    uint start = i * 32;
    uint4 a = data.Load4(start);
    uint4 b = data.Load4(start + 16);
    uint shift = ((i + epoch) & 3) * 8;
    float3 lo = asfloat(uint3(alignedField(a.x, a.y, shift), alignedField(a.y, a.z, shift),
                              alignedField(a.z, a.w, shift)));
    uint first = alignedField(a.w, b.x, shift), second = alignedField(b.x, b.y, shift);
    uint header = (first & b.w) | (second & ~b.w);
    float3 hi = lo + float3(2, 2, 0);
    uint slot = alignedField(b.y, b.z, shift);
    if (slot >= 64) return;
    uint triangleOffset = (128 + slot * 3) * 16;
    uint2 address = offsetAddress(uint2(baseLo, baseHi), triangleOffset);
    if (header & (1u << 16)) address.y++;
    if (header & (1u << 17)) address = offsetAddress(uint2(baseLo, baseHi), extent - 8);
    output.Store4(i * 32, uint4(asuint(lo), address.x));
    output.Store4(i * 32 + 16, uint4(asuint(hi), address.y));
    output.Store4(triangleOffset, uint4(asuint(lo), 0));
    output.Store4(triangleOffset + 16, uint4(asuint(float3(hi.x, lo.yz)), 0));
    float y = (header & (1u << 18)) ? lo.y : hi.y;
    output.Store3(triangleOffset + 32, asuint(float3(lo.x, y, lo.z)));
    output.Store(triangleOffset + 44, header & 1023);
}

bool intersectsBounds(float3 origin, float3 direction, float3 lo, float3 hi, float nearT, float farT)
{
    for (uint axis = 0; axis < 3; ++axis) {
        if (direction[axis] == 0) {
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) return false;
        } else {
            float a = (lo[axis] - origin[axis]) / direction[axis];
            float b = (hi[axis] - origin[axis]) / direction[axis];
            nearT = max(nearT, min(a, b));
            farT = min(farT, max(a, b));
        }
    }
    return nearT <= farT;
}

static const uint miss = 0xffffffffu;
// The fixture uses nonnegative ray intervals, so float distance bits are
// ordered as unsigned integers. All triangle intersections are two-sided.
uint triangleDistance(float4 ray, float4 direction, float3 a, float3 b, float3 c)
{
    float3 e1 = b-a, e2 = c-a, delta = ray.xyz-a;
    float3 p = cross(direction.xyz, e2), q = cross(delta, e1);
    float determinant = dot(e1, p);
    if (determinant == 0) return miss;
    float u = dot(delta, p) / determinant;
    float v = dot(direction.xyz, q) / determinant;
    float t = dot(e2, q) / determinant;
    if (u < 0 || v < 0 || u+v > 1 || t < ray.w || t > direction.w) return miss;
    return t == 0 ? 0 : asuint(t);
}

groupshared uint2 childAddress[64];
groupshared uint closest, primitive, hitCount;
[RootSignature(ROOT)] [numthreads(64,1,1)]
void TraceHits(uint lane : SV_GroupIndex, uint rayIndex : SV_GroupID)
{
    float4 ray = asfloat(queries[2 * rayIndex]);
    float4 direction = asfloat(queries[2 * rayIndex + 1]);
    uint2 bounds = offsetAddress(uint2(baseLo, baseHi), lane * 32);
    bool validLo, validHi;
    uint4 lo = readAddress(bounds, 4, validLo);
    uint4 hi = readAddress(offsetAddress(bounds, 16), 4, validHi);
    bool candidate = validLo && validHi &&
        intersectsBounds(ray.xyz, direction.xyz, asfloat(lo.xyz), asfloat(hi.xyz), ray.w, direction.w);
    if (lane == 0) {
        closest = primitive = miss;
        hitCount = 0;
    }
    scanHalf(lane, uint(candidate));
    uint count = scans[31] + scans[63];
    if (candidate) {
        uint rank = scans[lane] - 1 + (lane < 32 ? 0 : scans[31]);
        childAddress[rank] = uint2(lo.w, hi.w);
    }
    GroupMemoryBarrierWithGroupSync();
    uint distance = miss, id = miss;
    if (lane < count) {
        uint2 address = childAddress[lane];
        bool va, vb, vc, vi;
        float3 a = asfloat(readAddress(address, 3, va).xyz);
        float3 b = asfloat(readAddress(offsetAddress(address, 16), 3, vb).xyz);
        float3 c = asfloat(readAddress(offsetAddress(address, 32), 3, vc).xyz);
        id = readAddress(offsetAddress(address, 44), 1, vi).x;
        if (va && vb && vc && vi) distance = triangleDistance(ray, direction, a, b, c);
    }
    uint waveDistance = WaveActiveMin(distance);
    uint waveHits = WaveActiveCountBits(distance != miss);
    if (WaveIsFirstLane()) {
        InterlockedMin(closest, waveDistance);
        InterlockedAdd(hitCount, waveHits);
    }
    GroupMemoryBarrierWithGroupSync();
    uint waveId = WaveActiveMin(distance != miss && distance == closest ? id : miss);
    if (WaveIsFirstLane()) InterlockedMin(primitive, waveId);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) output.Store4(rayIndex * 16, uint4(primitive, closest, count, hitCount));
}

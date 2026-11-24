#define LAYOUT "SRV(t0),UAV(u0),RootConstants(num32BitConstants=16,b0)"
ByteAddressBuffer source : register(t0);
RWByteAddressBuffer target : register(u0);
cbuffer Settings : register(b0) { uint stride; uint countOrReset; uint seed; }

[RootSignature(LAYOUT)] [numthreads(32, 1, 1)]
void CopyBlocks(uint3 id : SV_DispatchThreadID)
{
    if (id.x < countOrReset) target.Store4(16 * id.x, source.Load4(16 * id.x));
}

[RootSignature(LAYOUT)] [numthreads(32, 1, 1)]
void FillBlocks(uint3 id : SV_DispatchThreadID)
{
    if (id.x < countOrReset) target.Store4(16 * id.x, seed.xxxx);
}

[RootSignature(LAYOUT)] [numthreads(32, 1, 1)]
void NumberBlocks(uint3 id : SV_DispatchThreadID)
{
    uint base = seed + 4u * id.x;
    if (id.x < countOrReset) target.Store4(16 * id.x, base + uint4(0, 1, 2, 3));
}

[RootSignature(LAYOUT)] [numthreads(32, 1, 1)]
void CopySpan(uint3 id : SV_DispatchThreadID)
{
    uint bytes = 64u * (1u + source.Load(4));
    for (uint offset = 16 * id.x; offset < bytes; offset += 16 * stride)
        target.Store4(offset, source.Load4(offset));
}

// A separately ordered desktop pass for the six observed header fields.
[RootSignature(LAYOUT)] [numthreads(1, 1, 1)]
void ResetBounds()
{
    target.Store3(16, 0x7f7fffffu.xxx);
    target.Store3(28, 0xff7fffffu.xxx);
}

ByteAddressBuffer colors : register(t1);
RWByteAddressBuffer decoded : register(u1);

#if STORAGE_Packed
RWTexture2D<uint> target : register(u0);
Texture2D<uint> source : register(t0);

#include "../src/rgb9e5.hlsli"
#else
RWTexture2D<float4> target : register(u0);
Texture2D<float4> source : register(t0);
#endif

[RootSignature("DescriptorTable(UAV(u0)),SRV(t1)")]
[numthreads(32,1,1)]
void RcasWrite(uint x : SV_DispatchThreadID)
{
    float3 rgb = asfloat(colors.Load3(x * 12));
#if STORAGE_Packed
    target[uint2(x, 0)] = pack_rgb9e5(rgb);
#else
    target[uint2(x, 0)] = float4(rgb, 1.0);
#endif
}

[RootSignature("DescriptorTable(SRV(t0)),UAV(u1)")]
[numthreads(32,1,1)]
void RcasRead(uint x : SV_DispatchThreadID)
{
#if STORAGE_Packed
    float4 color = unpack_rgb9e5(source.Load(int3(x, 0, 0)));
#else
    float4 color = source.Load(int3(x, 0, 0));
#endif
    decoded.Store4(x * 16, asuint(color));
}

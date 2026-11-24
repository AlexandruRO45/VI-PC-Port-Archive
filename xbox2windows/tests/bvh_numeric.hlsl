ByteAddressBuffer inputs : register(t0);
RWByteAddressBuffer outputs : register(u0);

[RootSignature("SRV(t0), UAV(u0)")]
[numthreads(64, 1, 1)]
void NumericBounds(uint3 id : SV_DispatchThreadID)
{
    uint offset = id.x * 32;
    uint4 first = inputs.Load4(offset);
    uint4 second = inputs.Load4(offset + 16);
    float2 a = f16tof32(uint2(first.x & 65535, first.x >> 16));
    float2 b = f16tof32(uint2(first.y & 65535, first.y >> 16));
    uint2 lo = f32tof16(min(a, b)), hi = f32tof16(max(a, b));
    float3 f = asfloat(uint3(first.zw, second.x));
    uint3 u = second.yzw;
    uint base = id.x * 40;
    outputs.Store4(base, uint4(lo.x | (lo.y << 16), hi.x | (hi.y << 16),
                               asuint(min(f.x, min(f.y, f.z))), asuint(max(f.x, max(f.y, f.z)))));
    outputs.Store3(base + 16, asuint(float3(
        clamp(f.x, asfloat(0xff7fffffu), asfloat(0x7f7fffffu)),
        clamp(f.x, asfloat(0x0f000000u), asfloat(0x7f7fffffu)),
        clamp(f.x, 0.0f, asfloat(0x3f7ff972u)))));
    outputs.Store3(base + 28, uint3(min(u.x, min(u.y, u.z)),
        max(min(u.x, u.y), min(max(u.x, u.y), u.z)), max(u.x, max(u.y, u.z))));
}

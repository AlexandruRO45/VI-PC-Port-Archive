#ifndef XBOX2WINDOWS_RGB9E5_HLSLI
#define XBOX2WINDOWS_RGB9E5_HLSLI

uint pack_rgb9e5(float3 rgb)
{
    rgb = clamp(rgb, 0.0, 65408.0);
    float peak = max(rgb.x, max(rgb.y, rgb.z));
    int exponent = max(0, int(asuint(peak) >> 23) - 127 + 16);
    float scale = asfloat(uint(127 + 24 - exponent) << 23);
    if (round(peak * scale) == 512.0) {
        ++exponent;
        scale *= 0.5;
    }
    uint3 m = uint3(round(rgb * scale));
    return (uint(exponent) << 27) | (m.z << 18) | (m.y << 9) | m.x;
}

float4 unpack_rgb9e5(uint bits)
{
    uint3 m = uint3(bits, bits >> 9, bits >> 18) & 511u;
    return float4(float3(m) * asfloat((127u + (bits >> 27) - 24u) << 23), 1.0);
}

#endif

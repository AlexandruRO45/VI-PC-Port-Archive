#if defined(INITIALIZE_TEXTURE)
RWTexture2D<uint> pixels : register(u0);
cbuffer Update : register(b0) { uint partial; };

[numthreads(2, 2, 1)]
void main(uint3 lane : SV_DispatchThreadID)
{
    // A select chain, not an indexed constant array: the Windows 10 in-box
    // WARP (10.0.19041) compiler crashes on the array when creating the PSO.
    const uint texel = 2 * lane.y + lane.x;
    if (!partial)
        pixels[lane.xy] = texel == 0 ? 0xff0000ff : texel == 1 ? 0xff00ff00
                        : texel == 2 ? 0xffff0000 : 0xff00ffff;
    else if (all(lane.xy == uint2(1, 0)))
        pixels[lane.xy] = 0xffff00ff;
}
#else
Texture2D<uint> pixels : register(t0);

float4 main(float4 position : SV_Position) : SV_Target
{
    uint value = pixels.Load(int3(position.x >= 24, position.y >= 24, 0));
    uint4 rgba = (value >> uint4(0, 8, 16, 24)) & 255;
#if defined(SWAP_CHANNELS)
    rgba = rgba.bgra;
#endif
    return float4(rgba) / 255.0;
}
#endif

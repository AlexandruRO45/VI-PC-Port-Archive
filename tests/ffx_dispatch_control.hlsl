// Source-built controls for public CI using the same descriptor and dispatch contracts.
#include "ffx_dispatch_io.hlsli"

[RootSignature(PRODUCER_ROOT)]
[numthreads(256,1,1)]
void producer(uint lane : SV_GroupIndex) {
    uint2 p = uint2(lane % 8, lane / 8);
    if (p.y >= 4) return;
    float3 h = asfloat(historyInput.Load3(lane*12));
    float3 color = asfloat(sceneInput.Load3(lane*12));
    recurrent[p] = float4(color,1);
    historyOutput[p] = float4(h,1);
    write16(p, float4(h,1));
}

[RootSignature(CONSUMER_ROOT)]
[numthreads(64,1,1)]
void rcas(uint lane : SV_GroupIndex) {
    uint2 base = uint2(lane % 8, lane / 8);
    [unroll] for (uint y=0; y<16; y+=8)
        [unroll] for (uint x=0; x<16; x+=8) {
            uint2 p = base + uint2(x,y);
            float3 color = (exp2(12.288 * load16(p).rgb)-1)/150;
            finalOutput[p] = float4(color,color.r);
        }
}

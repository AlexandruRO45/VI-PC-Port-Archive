cbuffer Config : register(b0) { uint4 config[4]; }
#ifdef TABLE_BINDINGS
#define SIGNATURE "DescriptorTable(UAV(u0,numDescriptors=13)),DescriptorTable(SRV(t0,numDescriptors=21))," \
                  "CBV(b0),CBV(b1),CBV(b2),CBV(b3),StaticSampler(s0,filter=FILTER_MIN_MAG_MIP_LINEAR," \
                  "addressU=TEXTURE_ADDRESS_CLAMP,addressV=TEXTURE_ADDRESS_CLAMP,addressW=TEXTURE_ADDRESS_CLAMP," \
                  "comparisonFunc=COMPARISON_LESS_EQUAL,borderColor=STATIC_BORDER_COLOR_OPAQUE_WHITE,minLOD=0)"
cbuffer Extra1 : register(b1) { uint first; }
cbuffer Extra2 : register(b2) { uint second; }
cbuffer Extra3 : register(b3) { uint third; }
ByteAddressBuffer sourceData : register(t20);
RWByteAddressBuffer resultData : register(u12);
#else
#define SIGNATURE "SRV(t0),UAV(u0),RootConstants(num32BitConstants=16,b0)"
ByteAddressBuffer sourceData : register(t0);
RWByteAddressBuffer resultData : register(u0);
#endif

[RootSignature(SIGNATURE)]
[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= config[0].x) return;
#ifdef TABLE_BINDINGS
    uint key = first + 3u * second + 5u * third;
#else
    uint key = config[3].y + 3u * config[3].z + 5u * config[3].w;
#endif
    uint value = (sourceData.Load(4u * i) * 33u) ^ (key + i * config[0].y);
    resultData.Store(4u * i, value);
}

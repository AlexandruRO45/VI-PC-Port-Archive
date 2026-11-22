struct Vertex { float2 uv : TEXCOORD0; float4 position : SV_Position; };
[RootSignature("DescriptorTable(SRV(t0,numDescriptors=2),visibility=SHADER_VISIBILITY_PIXEL),CBV(b0,visibility=SHADER_VISIBILITY_PIXEL),StaticSampler(s0,filter=FILTER_MIN_MAG_MIP_POINT,"
               "addressU=TEXTURE_ADDRESS_CLAMP,addressV=TEXTURE_ADDRESS_CLAMP,addressW=TEXTURE_ADDRESS_CLAMP)")]
Vertex main(uint id : SV_VertexID)
{
    Vertex v;
    const bool partial = id >= 3;
    const uint corner = partial ? id - 3 : id;
    v.uv = float2((corner << 1) & 2, corner & 2) * (partial ? 7.75f / 16.0f : 1.0f);
    v.position = float4(v.uv * float2(2,-2) + float2(-1,1), 0, 1);
    return v;
}

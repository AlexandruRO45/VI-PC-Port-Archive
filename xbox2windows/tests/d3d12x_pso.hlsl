// Source-built compute shader for the D3D12.X static-sampler PSO fixture.
// The test root signature supplies s0; the shader is compiled with
// -Qstrip_rootsignature so the pipeline uses the signature created by the test.
Texture2D<float4> t0 : register(t0);
RWTexture2D<float4> u0 : register(u0);
SamplerState s0 : register(s0);

[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    u0[id.xy] = t0.SampleLevel(s0, float2(0.5, 0.5), 0.0);
}

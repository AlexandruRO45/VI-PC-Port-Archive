RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<uint4> results : register(u0);
struct Query { float4 originMin; float4 directionMax; uint mask; uint flags; };
StructuredBuffer<Query> queries : register(t1);

[RootSignature("SRV(t0),UAV(u0),SRV(t1)")]
[numthreads(1, 1, 1)]
void CSMain(uint3 thread : SV_DispatchThreadID)
{
    uint i = thread.x;
    RayDesc ray;
    Query query = queries[i];
    ray.Origin = query.originMin.xyz;
    ray.Direction = query.directionMax.xyz;
    ray.TMin = query.originMin.w;
    ray.TMax = query.directionMax.w;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    // Bit 31 is a fixture request to report facing; the low bits are ray flags.
    q.TraceRayInline(scene, query.flags & 0x3ff, query.mask, ray);
    while (q.Proceed()) {}
    uint4 value = uint4(0, 0, 0, 0);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        uint status = 1;
        if ((query.flags >> 31) != 0 && q.CommittedTriangleFrontFace()) status = 3;
        value = uint4(status, q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), asuint(q.CommittedRayT()));
    }
    results[i] = value;
}

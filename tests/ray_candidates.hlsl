RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<uint4> results : register(u0);
struct Query { float4 originMin; float4 directionMax; uint mask; uint flags; };
StructuredBuffer<Query> queries : register(t1);

// Traces without template flags so geometry, instance and dynamic ray flags
// alone decide opacity, and handles every non-opaque candidate in the shader.
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
    // Bits 16-19 are a fixture convention: geometries whose non-opaque
    // candidates this shader commits. The low bits are ray flags.
    uint accept = (query.flags >> 16) & 0xf;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(scene, query.flags & 0x3ff, query.mask, ray);
    uint seen = 0, afterCommit = 0, afterShaderCommit = 0, anomaly = 0, shaderCommitted = 0;
    while (q.Proceed()) {
        // Legal while searching for the closest hit, but a first-hit search
        // must end at its first commit. Bit 3 marks a candidate after this
        // shader committed; bit 1 one after an automatic opaque commit only.
        if (shaderCommitted) afterShaderCommit = 1;
        else if (q.CommittedStatus() != COMMITTED_NOTHING) afterCommit = 1;
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { anomaly = 1; continue; }
        // A candidate must lie inside the ray extents and in front of any committed hit.
        float t = q.CandidateTriangleRayT();
        float limit = q.CommittedStatus() == COMMITTED_NOTHING ? ray.TMax : q.CommittedRayT();
        uint geometry = q.CandidateGeometryIndex();
        if (!(t > ray.TMin && t < limit) || q.CandidatePrimitiveIndex() != 0 || geometry >= 4) { anomaly = 1; continue; }
        seen |= 1u << geometry;
        if ((accept >> geometry) & 1) { q.CommitNonOpaqueTriangleHit(); shaderCommitted = 1; }
    }
    uint4 value = uint4((afterCommit << 1) | (anomaly << 2) | (afterShaderCommit << 3) | (seen << 8), 0, 0, 0);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        value = uint4(value.x | 1, q.CommittedInstanceID(),
                      q.CommittedGeometryIndex() | (q.CommittedPrimitiveIndex() << 16), asuint(q.CommittedRayT()));
    results[i] = value;
}

// Reads words through one root UAV and writes transformed words through another. Both views
// are bound at desktop GPU addresses translated from title addresses inside one aliased
// allocation (tests/gpu_alias.cpp).
#define AliasRoot "UAV(u0), UAV(u1), RootConstants(num32BitConstants=1, b0)"

RWByteAddressBuffer Source : register(u0);
RWByteAddressBuffer Destination : register(u1);
cbuffer Count : register(b0) { uint count; };

[RootSignature(AliasRoot)]
[numthreads(64, 1, 1)]
void AliasTransform(uint id : SV_DispatchThreadID)
{
    if (id >= count) return;
    const uint value = Source.Load(id * 4);
    Destination.Store(id * 4, (value ^ 0xa5a5a5a5u) + id);
}

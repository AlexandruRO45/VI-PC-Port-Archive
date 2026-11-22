// Stores and atomics through 64-bit title addresses, the other memory operations a shader that
// reads addresses from its inputs performs on the console (global_store, global_atomic_add and
// _sub, and float min and max). TitleLookup finds the aliased buffer and byte offset. Float
// min and max use a compare-exchange loop on the word's bits. Each record's result is 1 when one
// range holds the whole aligned DWORD and 0 otherwise; a rejected record writes nothing else.
#include "title_va_lookup.hlsli"

#define WritesRoot "DescriptorTable(UAV(u0, space = 1, numDescriptors = 8)), SRV(t0), SRV(t1), UAV(u0), " \
                   "RootConstants(num32BitConstants = 2, b0)"

RWByteAddressBuffer Aliases[8] : register(u0, space1);
StructuredBuffer<TitleRange> Ranges : register(t0);
ByteAddressBuffer Records : register(t1);  // 16 bytes each: address (uint2), operation, operand
RWByteAddressBuffer Results : register(u0);
cbuffer Counts : register(b0) { uint rangeCount; uint recordCount; };

static const uint OpStore = 0;
static const uint OpAdd = 1;
static const uint OpSub = 2;
static const uint OpMinFloat = 3;
static const uint OpMaxFloat = 4;

bool Replaces(uint operation, float candidate, float current)
{
    return operation == OpMinFloat ? candidate < current : candidate > current;
}

[RootSignature(WritesRoot)]
[numthreads(64, 1, 1)]
void WriteTitleAddresses(uint id : SV_DispatchThreadID)
{
    if (id >= recordCount) return;
    const uint4 record = Records.Load4(id * 16);
    uint slot;
    uint offset;
    if (!TitleLookup(Ranges, rangeCount, record.xy, 4, slot, offset)) {
        Results.Store(id * 4, 0);
        return;
    }
    const uint index = NonUniformResourceIndex(slot);
    const uint operation = record.z;
    const uint operand = record.w;
    uint previous;
    if (operation == OpStore) {
        Aliases[index].Store(offset, operand);
    } else if (operation == OpAdd) {
        Aliases[index].InterlockedAdd(offset, operand, previous);
    } else if (operation == OpSub) {
        Aliases[index].InterlockedAdd(offset, 0u - operand, previous);
    } else {
        uint current = Aliases[index].Load(offset);
        [allow_uav_condition]
        while (Replaces(operation, asfloat(operand), asfloat(current))) {
            Aliases[index].InterlockedCompareExchange(offset, current, operand, previous);
            if (previous == current) break;
            current = previous;
        }
    }
    Results.Store(id * 4, 1);
}

// Correctness probe for the atomics above, run on plain buffers before the fixture. Each of
// recordCount threads adds 1 to word 0 of Aliases[id % rangeCount], the same per-thread index
// into the UAV array that WriteTitleAddresses uses, so buffer i must gain exactly the number of
// ids congruent to i.
#define ProbeRoot "DescriptorTable(UAV(u0, space = 1, numDescriptors = 8)), RootConstants(num32BitConstants = 2, b0)"

[RootSignature(ProbeRoot)]
[numthreads(64, 1, 1)]
void ProbeArrayAtomics(uint id : SV_DispatchThreadID)
{
    if (id >= recordCount) return;
    uint previous;
    Aliases[NonUniformResourceIndex(id % rangeCount)].InterlockedAdd(0, 1, previous);
}

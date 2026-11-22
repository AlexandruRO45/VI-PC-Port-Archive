// Loads one word through each 64-bit title address in Records, as a shader that reads addresses
// from its inputs does on the console. TitleLookup finds the aliased buffer and byte offset.
// Each result is (value, slot + 1), or (0xdeadbeef, 0) when the address is misaligned or no
// single range holds the whole DWORD.
#include "title_va_lookup.hlsli"

#define LookupRoot "DescriptorTable(UAV(u0, space = 1, numDescriptors = 8)), SRV(t0), SRV(t1), UAV(u0), " \
                   "RootConstants(num32BitConstants = 2, b0)"

RWByteAddressBuffer Aliases[8] : register(u0, space1);
StructuredBuffer<TitleRange> Ranges : register(t0);
ByteAddressBuffer Records : register(t1);
RWByteAddressBuffer Results : register(u0);
cbuffer Counts : register(b0) { uint rangeCount; uint recordCount; };

[RootSignature(LookupRoot)]
[numthreads(64, 1, 1)]
void LookupTitleAddresses(uint id : SV_DispatchThreadID)
{
    if (id >= recordCount) return;
    const uint2 address = Records.Load2(id * 8);
    uint slot;
    uint offset;
    if (TitleLookup(Ranges, rangeCount, address, 4, slot, offset)) {
        Results.Store2(id * 8, uint2(Aliases[NonUniformResourceIndex(slot)].Load(offset), slot + 1));
    } else {
        Results.Store2(id * 8, uint2(0xdeadbeefu, 0));
    }
}

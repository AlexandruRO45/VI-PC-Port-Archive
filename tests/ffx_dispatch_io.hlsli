#include "../src/rgb9e5.hlsli"

#define PRODUCER_ROOT "DescriptorTable(UAV(u2), UAV(u1), UAV(u6)), SRV(t3), SRV(t9)"
#define CONSUMER_ROOT "DescriptorTable(SRV(t18), UAV(u11))"

// The two read-only producer fixtures use RGB DWORD triples at integer coordinates.
ByteAddressBuffer sceneInput : register(t3);
ByteAddressBuffer historyInput : register(t9);
RWTexture2D<float4> historyOutput : register(u1);
RWTexture2D<float4> recurrent : register(u6);
RWTexture2D<float4> finalOutput : register(u11);
#if STORAGE_Packed
RWTexture2D<uint> target16 : register(u2);
Texture2D<uint> source16 : register(t18);
void write16(uint2 p, float4 value) { target16[p] = pack_rgb9e5(value.rgb); }
float4 load16(uint2 p) { return unpack_rgb9e5(source16.Load(int3(p, 0))); }
#else
RWTexture2D<float4> target16 : register(u2);
Texture2D<float4> source16 : register(t18);
void write16(uint2 p, float4 value) { target16[p] = value; }
float4 load16(uint2 p) { return source16.Load(int3(p, 0)); }
#endif

int median_i(int a, int b, int c) { return max(min(a,b), min(max(a,b),c)); }
float median_f(float a, float b, float c) { return max(min(a,b), min(max(a,b),c)); }
uint insert_byte(uint old, uint lane, uint value) {
    return (old & ~(255u << (8*lane))) | (value << (8*lane));
}
int idot(uint a, uint b, int sum) {
    [unroll] for (int i=0; i<4; ++i) {
        sum += (asint(a << (24-8*i)) >> 24) * (asint(b << (24-8*i)) >> 24);
    }
    return sum;
}

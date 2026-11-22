RWStructuredBuffer<uint> result : register(u0);
groupshared uint active[32];
groupshared int2 endpoints[32];
groupshared uint tile[576];

[numthreads(32, 1, 1)]
void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint4 inputCase = inputs[group.x];
    uint2 cell = uint2((lane & 1) | ((lane >> 1) & 14), (lane >> 1) & 1);
    active[lane] = all(cell < inputCase.xy / 2);
    float2 scale = float2(inputCase.zw) / 256.0;
    uint2 pixel = 2 * cell + (lane == 31 ? 1 : 0);
    endpoints[lane] = int2(round((float2(pixel) + 0.5) * scale - 0.5));
#if STORES
    for (uint i = lane; i < 576; i += 32) tile[i] = 0;
#endif
    GroupMemoryBarrierWithGroupSync();
    uint count = 0;
    for (uint j = 0; j < 32; ++j) count += active[j];
    int2 extent = 3 + endpoints[31] - endpoints[0];
    bool enter = count == 32 && (INCLUSIVE_X ? extent.x <= 33 : extent.x < 33) && extent.y < 7;
#if STORES
    if (enter) {
        for (uint row = 0; row < 6; ++row) {
            uint address = (32 * row + lane) * 3;
            uint tag = 1 + 96 * row + 3 * lane;
            tile[address] = tag;
            tile[address + 1] = tag + 1;
            tile[address + 2] = tag + 2;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = lane; k < 576; k += 32) result[group.x * 576 + k] = tile[k];
#else
    if (lane == 0) {
        uint start = group.x * 4;
        result[start] = count;
        result[start + 1] = extent.x;
        result[start + 2] = extent.y;
        result[start + 3] = enter;
    }
#endif
}

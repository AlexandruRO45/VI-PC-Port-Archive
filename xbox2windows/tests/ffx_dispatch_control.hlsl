// Source-built controls for public CI using the same descriptor and dispatch contracts.
#include "ffx_dispatch_io.hlsli"

[RootSignature(SCRATCH_ROOT)]
[numthreads(64,1,1)]
void pass24(uint3 p : SV_DispatchThreadID) {
    uint offset = p.y*0x3c20 + p.x*16;
    work.Store4(0x7f4bc0 + 0x3c20 + 16 + offset,work.Load4(offset));
}

[RootSignature(SCRATCH_ROOT)]
[numthreads(32,1,1)]
void pass25(uint i : SV_DispatchThreadID) {
    // Row-major enumeration of the 44 cells around the 4x4 interior.
    if (i>=44) return;
    uint row = i<10 ? 0 : (i<34 ? 1+(i-10)/6 : 5);
    uint col = i<10 ? i : (i<34 ? (i-10)%6 : i-34);
    if (row>0 && row<5 && col>0) col+=4;
    work.Store4(0x7f4bc0+row*0x3c20+col*16,0);
}

[RootSignature(PRODUCER_ROOT)]
[numthreads(256,1,1)]
void producer(uint lane : SV_GroupIndex) {
    if (lane>=8) return;
    uint2 cell = uint2(lane%4,lane/4);
    uint center=0;
    for (uint tap=0; tap<9; ++tap) {
        uint address = 0x7f4bc0+(cell.y+tap/3)*0x3c20+(cell.x+tap%3)*16;
        uint4 v=work.Load4(address);
        audit_load(cell,tap,address,v);
        if (tap==4) center=v.x;
    }
    for (uint i=0;i<4;++i) {
        uint2 p=cell*2+uint2(i%2,i/2);
        float3 h=asfloat(historyInput.Load3((p.y*8+p.x)*12));
        float3 color=asfloat(sceneInput.Load3((p.y*8+p.x)*12));
        recurrent[p]=float4(color,1);
        historyOutput[p]=float4(h,1);
        write16(p,float4(h*(center==0 ? 0.75 : 1.0),1));
    }
}

[RootSignature(CONSUMER_ROOT)]
[numthreads(64,1,1)]
void rcas(uint lane : SV_GroupIndex) {
    uint2 base = uint2(lane % 8, lane / 8);
    [unroll] for (uint y=0; y<16; y+=8)
        [unroll] for (uint x=0; x<16; x+=8) {
            uint2 p = base + uint2(x,y);
            float3 color = (exp2(12.288 * load16(p).rgb)-1)/150;
            finalOutput[p] = float4(color,color.r);
        }
}

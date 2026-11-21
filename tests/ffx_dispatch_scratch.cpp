#include "ffx_dispatch_scratch.h"
#include "pass24Raw.h"
#include "pass25Raw.h"
#include <cmath>
#ifdef FFX_DISPATCH_AOT
#include "scratch_reference.h"
#endif

namespace {
constexpr UINT base = 0x7f4bc0, stride = 0x3c20;
constexpr UINT plane_bytes = 6*stride, window = 6*4*16, trace_bytes = 8*9*20;
constexpr uint32_t poison = 0xcdcdcdcdu;
constexpr auto uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
constexpr auto src = D3D12_RESOURCE_STATE_COPY_SOURCE;
constexpr auto dst = D3D12_RESOURCE_STATE_COPY_DEST;

uint32_t patterned(UINT x, UINT y, UINT c)
{
    return (x+13*y+19*c+1)*0x19660du+0x3c6ef35fu;
}

void snapshot(ID3D12GraphicsCommandList* list, ID3D12Resource* work, ID3D12Resource* back, UINT offset)
{
    transition(list,work,uav,src);
    for (UINT y=0; y<4; ++y)
        list->CopyBufferRegion(back,offset+y*96,work,base+y*stride,96);
    transition(list,work,src,uav);
}
}

DispatchScratch::DispatchScratch(ID3D12Device* device, ID3D12Resource* provider_scratch, ScratchInput input)
    : mode(input), work(provider_scratch), tensor(device,{pass24Raw,sizeof(pass24Raw)}),
      padding(device,{pass25Raw,sizeof(pass25Raw)})
{
    if (work->GetDesc().Width < base+plane_bytes)
        throw std::runtime_error("Short provider scratch buffer");
    std::vector<uint32_t> seed((2*plane_bytes+trace_bytes)/4,poison);
    for (UINT y=0; y<6; ++y)
        for (UINT x=0; x<66; ++x)
            for (UINT c=0; c<4; ++c) {
                const UINT i=y*stride/4+4*x+c;
                seed[i] = mode==ScratchInput::pattern ? patterned(x,y,c) : 0;
                if (mode==ScratchInput::bypass) seed[plane_bytes/4+i]=0;
            }
    upload=bvh_fixture::upload(device,seed);
    back=make_buffer(device,2*window+trace_bytes,D3D12_HEAP_TYPE_READBACK,dst);
    trace=make_buffer(device,trace_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COMMON,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
}

void DispatchScratch::produce(ID3D12GraphicsCommandList* list)
{
    transition(list,work,D3D12_RESOURCE_STATE_COMMON,dst);
    transition(list,trace.Get(),D3D12_RESOURCE_STATE_COMMON,dst);
    list->CopyBufferRegion(work,0,upload.Get(),0,plane_bytes);
    list->CopyBufferRegion(work,base,upload.Get(),plane_bytes,plane_bytes);
    list->CopyBufferRegion(trace.Get(),0,upload.Get(),2*plane_bytes,trace_bytes);
    transition(list,work,dst,uav);
    transition(list,trace.Get(),dst,uav);
    if (mode!=ScratchInput::bypass) {
        list->SetComputeRootSignature(tensor.root.Get());
        list->SetComputeRootUnorderedAccessView(0,work->GetGPUVirtualAddress());
        list->SetPipelineState(tensor.pipeline.Get());
        list->Dispatch(1,4,1);
    }
    snapshot(list,work,back.Get(),0);
    if (mode!=ScratchInput::bypass) {
        list->SetComputeRootSignature(padding.root.Get());
        list->SetComputeRootUnorderedAccessView(0,work->GetGPUVirtualAddress());
        list->SetPipelineState(padding.pipeline.Get());
        list->Dispatch(2,1,1);
    }
    snapshot(list,work,back.Get(),window);
}

void DispatchScratch::bind_consumer(ID3D12GraphicsCommandList* list) const
{
    list->SetComputeRootUnorderedAccessView(3,work->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(4,trace->GetGPUVirtualAddress());
}

void DispatchScratch::finish(ID3D12GraphicsCommandList* list)
{
    transition(list,trace.Get(),uav,src);
    list->CopyBufferRegion(back.Get(),2*window,trace.Get(),0,trace_bytes);
    transition(list,work,uav,D3D12_RESOURCE_STATE_COMMON);
}

void DispatchScratch::verify() const
{
    auto words=bvh_fixture::read(back.Get(),(2*window+trace_bytes)/4);
    for (UINT y=0; y<4; ++y)
        for (UINT x=0; x<6; ++x)
            for (UINT c=0; c<4; ++c) {
                const UINT i=(6*y+x)*4+c;
                uint32_t before=0;
                if (mode!=ScratchInput::bypass) {
                    before=poison;
                    if (x && y) {
#ifdef FFX_DISPATCH_AOT
                        before=before_padding[unsigned(mode)*window/4+i];
#else
                        before=mode==ScratchInput::pattern ? patterned(x-1,y-1,c) : 0;
#endif
                    }
                }
                bvh_fixture::equal(words[i],before,i,24);
                bvh_fixture::equal(words[window/4+i],(x>=1 && x<=4 && y) ? before : 0,i,25);
            }
    std::array<unsigned,9> tensor_reads{};
    for (UINT lane=0; lane<8; ++lane)
        for (UINT tap=0; tap<9; ++tap) {
            const UINT x=lane%4+tap%3, y=lane/4+tap/3;
            const UINT i=2*window/4+5*(lane*9+tap);
            bvh_fixture::equal(words[i],base+y*stride+16*x,lane,tap);
            for (UINT c=0; c<4; ++c)
                bvh_fixture::equal(words[i+1+c],words[window/4+4*(6*y+x)+c],lane,tap);
            tensor_reads[tap] += x>=1 && x<=4 && y>0;
        }
    if (tensor_reads!=std::array<unsigned,9>{3,4,3,6,8,6,6,8,6})
        throw std::runtime_error("Unexpected per-instruction scratch ownership");
}

float DispatchScratch::expected_history_fraction(unsigned x,unsigned y) const
{
#ifdef FFX_DISPATCH_AOT
    const auto logit=history_logits[unsigned(mode)*32+8*y+x];
    uint32_t scale_bits=0xba38aa3b;
    float scale;
    std::memcpy(&scale,&scale_bits,sizeof scale);
    return 1.f/(1.f+std::exp2(scale*logit));
#else
    return mode==ScratchInput::pattern ? 1.f : 0.5f;
#endif
}

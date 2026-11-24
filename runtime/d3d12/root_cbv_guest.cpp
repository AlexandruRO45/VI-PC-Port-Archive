// Verify actual root CBV window bytes on a Scarlett device, without
// pre-registering a placed resource at the guest address.
#include "compute_fixture.h"
#include "frame_cli.h"
#include "root_cbv.h"
namespace {
using namespace frame_abi;
using namespace compute_fixture;
using Pipeline=HRESULT(WINAPI*)(ID3D12Device*,const D3D12_COMPUTE_PIPELINE_STATE_DESC*,UINT,const void*,REFIID,void**);
using SetRoot=void(WINAPI*)(ID3D12GraphicsCommandList*,ID3D12RootSignature*);
using SetBuffer=void(WINAPI*)(ID3D12GraphicsCommandList*,UINT,UINT64);
using Signal=HRESULT(WINAPI*)(ID3D12CommandQueue*,ID3D12Fence*,UINT64);
constexpr UINT page=65536;
struct Pages {
    UINT* ptr=static_cast<UINT*>(VirtualAlloc(nullptr,3*page,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
    ~Pages(){if(ptr)VirtualFree(ptr,0,MEM_RELEASE);}
};
std::string run(const wchar_t* adapter,bool){
    DXGI_ADAPTER_DESC1 selected{};
    constexpr std::array<UINT,18> profile{0x20004,0,UINT_MAX,0,0,0,UINT_MAX,UINT_MAX,0,1,0,0,0,0,0,0,0,0};
    auto device=create_device(adapter,selected,profile.data(),__uuidof(ID3D12Device11));
    Pages pages;if(!pages.ptr)throw std::runtime_error("allocate CBV backing");
    for(UINT i=0;i<3*page/4;++i)pages.ptr[i]=0x13570000u^(i*0x1020305u);
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC desc{2,params,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> serialized;
    check(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,nullptr),"serialize root CBV");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&root)),"root CBV signature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature=root.Get();pd.CS={root_cbv_dxil,sizeof(root_cbv_dxil)};
    ComPtr<ID3D12PipelineState> pso;
    check(method<Pipeline>(device.Get(),62)(device.Get(),&pd,0,nullptr,IID_PPV_ARGS(&pso)),"root CBV pipeline");
    D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ComPtr<ID3D12CommandQueue> queue;
    check(method<CreateQueue>(device.Get(),48)(device.Get(),&q,IID_PPV_ARGS(&queue)),"root CBV queue");
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"root CBV fence");
    constexpr std::array<UINT,6> seed{0xdeadbeef,0,0,0,0,0xcafebabe};
    auto source=buffer(device.Get(),D3D12_HEAP_TYPE_UPLOAD,sizeof(seed),D3D12_RESOURCE_STATE_GENERIC_READ);
    auto output=buffer(device.Get(),D3D12_HEAP_TYPE_DEFAULT,sizeof(seed),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback=buffer(device.Get(),D3D12_HEAP_TYPE_READBACK,sizeof(seed),D3D12_RESOURCE_STATE_COPY_DEST);
    upload(source.Get(),seed.data(),sizeof(seed));
    constexpr std::array<UINT,4> offsets{page-256,0,page,page+256};
    for(UINT i=0;i<offsets.size();++i){
        ComPtr<ID3D12CommandAllocator> allocator;
        check(device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&allocator)),"root CBV allocator");
        const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_COMPUTE,0,1,0};
        ComPtr<ID3D12GraphicsCommandList> list;
        check(method<CreateList>(device.Get(),73)(device.Get(),list_desc,allocator.Get(),pso.Get(),IID_PPV_ARGS(&list)),"root CBV list");
        if(i)transition(list.Get(),output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(output.Get(),0,source.Get(),0,sizeof(seed));
        transition(list.Get(),output.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        method<SetRoot>(list.Get(),33)(list.Get(),root.Get());
        method<SetBuffer>(list.Get(),41)(list.Get(),0,reinterpret_cast<UINT64>(pages.ptr)+offsets[i]);
        method<SetBuffer>(list.Get(),45)(list.Get(),1,output->GetGPUVirtualAddress());
        list->Dispatch(1,1,1);
        transition(list.Get(),output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(),0,output.Get(),0,sizeof(seed));
        check(list->Close(),"close root CBV");
        ID3D12CommandList* batch[]{list.Get()};queue->ExecuteCommandLists(1,batch);
        check(method<Signal>(queue.Get(),16)(queue.Get(),fence.Get(),i+1),"signal root CBV");
        wait_for_fence(fence.Get(),i+1);
        auto expected=seed;auto ptr=pages.ptr+offsets[i]/4;
        expected[1]=ptr[0];expected[2]=ptr[63];expected[3]=ptr[8193];expected[4]=ptr[16383];
        D3D12_RANGE range{0,sizeof(seed)},empty{};void* data{};
        check(readback->Map(0,&range,&data),"root CBV readback");
        const bool equal=std::memcmp(data,expected.data(),sizeof(seed))==0;
        readback->Unmap(0,&empty);
        if(!equal)throw std::runtime_error("root CBV data mismatch at offset "+std::to_string(offsets[i]));
    }
    char receipt[160];
    std::snprintf(receipt,sizeof(receipt),"d3d12-root-cbv PASS pci=%04x:%04x slots=41,45 windows=4 words=16/16 guards=8",selected.VendorId,selected.DeviceId);
    return receipt;
}
}
int wmain(int argc,wchar_t** argv){return tuxbox_test::frame_main(argc,argv,"d3d12-root-cbv",false,run);}

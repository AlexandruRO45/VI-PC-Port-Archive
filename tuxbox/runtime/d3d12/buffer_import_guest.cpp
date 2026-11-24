// Read actual CPU-backed page data, not a successful device-local stand-in.
#include "compute_fixture.h"
#include "frame_cli.h"
namespace {
using namespace frame_abi;
using Place = HRESULT (WINAPI*)(ID3D12Device*, UINT64, const D3D12_RESOURCE_DESC*,
    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
using Register = HRESULT (WINAPI*)(ID3D12Device*, UINT64, UINT, SIZE_T*);
using Unregister = void (WINAPI*)(ID3D12Device*, SIZE_T);
constexpr UINT page=65536, words=256;
struct Memory {
    UINT* base=static_cast<UINT*>(VirtualAlloc(nullptr,2*page,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
    ~Memory(){if(base)VirtualFree(base,0,MEM_RELEASE);}
};
std::string run(const wchar_t* requested, bool negative) {
    DXGI_ADAPTER_DESC1 selected{};
    auto device=create_device(requested,selected);
    auto place=method<Place>(device.Get(),44);
    auto reg=method<Register>(device.Get(),74);
    auto unreg=method<Unregister>(device.Get(),75);
    Memory memory;
    if(!memory.base)throw std::runtime_error("allocate two guest pages");
    const auto base=reinterpret_cast<UINT64>(memory.base);
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width=page;
    desc.Height=desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1;
    desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if(negative){
        // A no-access reservation is deliberately non-importable. The strict
        // mode must fail rather than silently create unrelated GPU storage.
        // Obtain an address known to be usable, then release its entire
        // reservation. MEM_COMMIT alone cannot recommit an unreserved range.
        void* reservation=VirtualAlloc(nullptr,page,MEM_RESERVE,PAGE_READWRITE);
        if(!reservation || !VirtualFree(reservation,0,MEM_RELEASE))
            throw std::runtime_error("prepare free negative address");
        MEMORY_BASIC_INFORMATION info{};
        if(!VirtualQuery(reservation,&info,sizeof(info)) || info.State!=MEM_FREE)
            throw std::runtime_error("negative address not free");
        ComPtr<ID3D12Resource> result;
        auto hr=place(device.Get(),reinterpret_cast<UINT64>(reservation),&desc,
                     D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&result));
        if(SUCCEEDED(hr) || result)throw std::runtime_error("strict mode accepted stand-in");
        char receipt[128];
        std::snprintf(receipt,sizeof(receipt),"d3d12-buffer-import-reject PASS pci=%04x:%04x rejected=1",selected.VendorId,selected.DeviceId);
        return receipt;
    }
    for(UINT i=0;i<words;++i){
        memory.base[i]=0x12340000u+i;
        memory.base[page/4+i]=0xfedc0000u+i*3u;
    }
    SIZE_T token=0;
    check(reg(device.Get(),base,2,&token),"register page pool");
    if(!token)throw std::runtime_error("zero pool token");
    auto make=[&](UINT64 address){
        ComPtr<ID3D12Resource> out;
        check(place(device.Get(),address,&desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&out)),"import page");
        if(!out)throw std::runtime_error("missing imported resource");
        return out;
    };
    auto a=make(base),b=make(base+page),alias=make(base);
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&queue_desc,IID_PPV_ARGS(&queue)),"queue");
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"fence");
    auto readback=compute_fixture::buffer(device.Get(),D3D12_HEAP_TYPE_READBACK,3*words*4,D3D12_RESOURCE_STATE_COPY_DEST);
    auto observe=[&](UINT pass){
        ComPtr<ID3D12CommandAllocator> allocator;
        check(device->CreateCommandAllocator(queue_desc.Type,IID_PPV_ARGS(&allocator)),"allocator");
        ComPtr<ID3D12GraphicsCommandList> list;
        const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT,0,1,0};
        check(method<CreateList>(device.Get(),73)(device.Get(),list_desc,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"list");
        // Buffers in COMMON implicitly promote to COPY_SOURCE.
        for(UINT i=0;i<3;++i)
            list->CopyBufferRegion(readback.Get(),i*words*4, (i==0?a:i==1?b:alias).Get(),0,words*4);
        check(list->Close(),"close");
        ID3D12CommandList* batch[]={list.Get()};
        queue->ExecuteCommandLists(1,batch);
        check(queue->Signal(fence.Get(),pass+1),"signal");
        compute_fixture::wait_for_fence(fence.Get(),pass+1);
        D3D12_RANGE range{0,3*words*4},empty{};
        void* ptr{};
        check(readback->Map(0,&range,&ptr),"readback");
        bool equal=true;
        auto actual=static_cast<const UINT*>(ptr);
        for(UINT i=0;i<words;++i){
            equal &= actual[i]==memory.base[i];
            equal &= actual[words+i]==memory.base[page/4+i];
            equal &= actual[2*words+i]==memory.base[i];
        }
        readback->Unmap(0,&empty);
        if(!equal)throw std::runtime_error("imported buffer alias readback differs");
    };
    observe(0);
    // Drop the A aliases after GPU completion, then recreate the page. The
    // weak cache must not own it. B remains live and must retain its data.
    a.Reset();alias.Reset();
    for(UINT i=0;i<words;++i)memory.base[i]=0x98760000u+i*7u;
    a=make(base);alias=make(base);
    observe(1);
    a.Reset();alias.Reset();b.Reset();
    unreg(device.Get(),token);
    char receipt[160];
    std::snprintf(receipt,sizeof(receipt),"d3d12-buffer-import PASS pci=%04x:%04x pages=2 words=1536/1536 passes=2 aliases=2",selected.VendorId,selected.DeviceId);
    return receipt;
}
std::string layout(const wchar_t* requested, bool) {
    DXGI_ADAPTER_DESC1 selected{};
    constexpr std::array<UINT,18> scarlett{
        0x20004,0,UINT_MAX,0,0,0,UINT_MAX,UINT_MAX,0,1,0,0,0,0,0,0,0,0};
    auto device=create_device(requested,selected,scarlett.data(),__uuidof(ID3D12Device11));
    using Layout = HRESULT (WINAPI*)(ID3D12Device*, UINT64, const D3D12_RESOURCE_DESC*,
        D3D12_BARRIER_LAYOUT,const D3D12_CLEAR_VALUE*,REFIID,void**);
    auto place=method<Layout>(device.Get(),133);
    struct Reservation {
        UINT* base=static_cast<UINT*>(VirtualAlloc(nullptr,2*page,MEM_RESERVE,PAGE_READWRITE));
        ~Reservation(){if(base)VirtualFree(base,0,MEM_RELEASE);}
    } memory;
    if(!memory.base)throw std::runtime_error("reserve pages for layout retry");
    auto base=reinterpret_cast<UINT64>(memory.base);
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width=words*4;
    desc.Height=desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1;
    desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    std::array<ComPtr<ID3D12Resource>,2> resources;
    std::array<UINT64,2> addresses{base,base+page+4};
    for(UINT i=0;i<resources.size();++i){
        check(place(device.Get(),addresses[i],&desc,D3D12_BARRIER_LAYOUT_COMMON,nullptr,
                    IID_PPV_ARGS(&resources[i])),"place late committed layout buffer");
        if(!resources[i] || resources[i]->GetGPUVirtualAddress()!=addresses[i]
                || resources[i]->GetDesc().Width!=desc.Width)
            throw std::runtime_error("layout retry lost guest buffer identity at index "+std::to_string(i));
        MEMORY_BASIC_INFORMATION info{};
        if(!VirtualQuery(reinterpret_cast<void*>(addresses[i]),&info,sizeof(info)) || info.State!=MEM_COMMIT)
            throw std::runtime_error("layout retry did not commit pages");
        auto data=reinterpret_cast<UINT*>(addresses[i]);
        for(UINT j=0;j<words;++j)data[j]=0x98760000u+i*0x10000u+j*13u;
    }
    D3D12_COMMAND_QUEUE_DESC q{};
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)),"layout queue");
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"layout fence");
    ComPtr<ID3D12CommandAllocator> allocator;
    check(device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&allocator)),"layout allocator");
    ComPtr<ID3D12GraphicsCommandList> list;
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT,0,1,0};
    check(method<CreateList>(device.Get(),73)(device.Get(),list_desc,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"layout list");
    auto readback=compute_fixture::buffer(device.Get(),D3D12_HEAP_TYPE_READBACK,2*words*4,D3D12_RESOURCE_STATE_COPY_DEST);
    for(UINT i=0;i<resources.size();++i)list->CopyBufferRegion(readback.Get(),i*words*4,resources[i].Get(),0,words*4);
    check(list->Close(),"layout close");
    ID3D12CommandList* batch[]={list.Get()};
    queue->ExecuteCommandLists(1,batch);
    using Signal = HRESULT (WINAPI*)(ID3D12CommandQueue*,ID3D12Fence*,UINT64);
    check(method<Signal>(queue.Get(),16)(queue.Get(),fence.Get(),1),"layout signal");
    compute_fixture::wait_for_fence(fence.Get(),1);
    D3D12_RANGE range{0,2*words*4},empty{};
    void* ptr{};
    check(readback->Map(0,&range,&ptr),"layout readback");
    bool equal=true;
    auto actual=static_cast<const UINT*>(ptr);
    for(UINT i=0;i<resources.size();++i)
        for(UINT j=0;j<words;++j)equal &= actual[i*words+j]==0x98760000u+i*0x10000u+j*13u;
    readback->Unmap(0,&empty);
    if(!equal)throw std::runtime_error("layout retry readback differs");
    char receipt[160];
    std::snprintf(receipt,sizeof(receipt),"d3d12-buffer-layout PASS pci=%04x:%04x slot=133 words=512/512 VA-offsets=0,4",selected.VendorId,selected.DeviceId);
    return receipt;
}
}
int wmain(int argc,wchar_t** argv){
    if(argc>1 && !std::wcscmp(argv[1],L"--layout"))
        return tuxbox_test::frame_main(argc-1,argv+1,"d3d12-buffer-layout",false,layout);
    return tuxbox_test::frame_main(argc,argv,"d3d12-buffer-import",true,run,L"--reject"," [--reject]");
}

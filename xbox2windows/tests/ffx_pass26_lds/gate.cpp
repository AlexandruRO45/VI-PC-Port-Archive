#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr, const char* stage) {
    if (FAILED(hr)) { std::fprintf(stderr,"%s: 0x%08lx\n",stage,(unsigned long)hr); std::exit(2); }
}
int wmain(int argc, wchar_t** argv) {
    if (argc != 4 && argc != 5) return 2;
    const bool warp = wcscmp(argv[2],L"warp")==0;
    ComPtr<ID3D11Device> dev; ComPtr<ID3D11DeviceContext> ctx;
    check(D3D11CreateDevice(nullptr,warp?D3D_DRIVER_TYPE_WARP:D3D_DRIVER_TYPE_HARDWARE,nullptr,0,
        nullptr,0,D3D11_SDK_VERSION,&dev,nullptr,&ctx),"device");
    ComPtr<IDXGIDevice> dxgi; ComPtr<IDXGIAdapter> adapter; DXGI_ADAPTER_DESC ad{};
    check(dev.As(&dxgi),"dxgi"); check(dxgi->GetAdapter(&adapter),"adapter"); check(adapter->GetDesc(&ad),"desc");
    ComPtr<ID3DBlob> code,errors;
    HRESULT hr=D3DCompileFromFile(argv[1],nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,"main","cs_5_0",0,0,&code,&errors);
    if(FAILED(hr)&&errors) std::fprintf(stderr,"%s",(char*)errors->GetBufferPointer()); check(hr,"compile");
    ComPtr<ID3D11ComputeShader> shader;
    check(dev->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&shader),"shader");
    const unsigned groups = unsigned(_wtoi(argv[3]));
    const unsigned words = argc == 5 ? unsigned(_wtoi(argv[4])) : 4;
    if (!groups || groups > 65535 || !words || words > 1024) return 2;
    std::vector<unsigned> initial(groups*words, 0xdeadbeefu);
    D3D11_BUFFER_DESC bd{}; bd.ByteWidth=UINT(initial.size()*4); bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;
    D3D11_SUBRESOURCE_DATA seed{initial.data(),0,0};
    ComPtr<ID3D11Buffer> out,back;ComPtr<ID3D11UnorderedAccessView> uav;
    check(dev->CreateBuffer(&bd,&seed,&out),"buffer"); check(dev->CreateUnorderedAccessView(out.Get(),nullptr,&uav),"uav");
    bd.BindFlags=bd.MiscFlags=bd.StructureByteStride=0;bd.Usage=D3D11_USAGE_STAGING;bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    check(dev->CreateBuffer(&bd,nullptr,&back),"staging");
    ctx->CSSetShader(shader.Get(),nullptr,0);
    ctx->CSSetUnorderedAccessViews(0,1,uav.GetAddressOf(),nullptr);ctx->Dispatch(groups,1,1);
    ctx->ClearState();ctx->CopyResource(back.Get(),out.Get());D3D11_MAPPED_SUBRESOURCE map{};
    check(ctx->Map(back.Get(),0,D3D11_MAP_READ,0,&map),"readback");
    const unsigned* got=(const unsigned*)map.pData;std::printf("adapter=%ls\n",ad.Description);
    for(unsigned i=0;i<groups;i++) {
        for(unsigned j=0;j<words;j++) std::printf("%u%c",got[words*i+j],j+1==words?'\n':' ');
    }
    ctx->Unmap(back.Get(),0);return 0;
}

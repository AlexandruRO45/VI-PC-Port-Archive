#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <cstdio>
#include <cstdlib>
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr, const char* stage) {
    if (FAILED(hr)) { std::fprintf(stderr,"%s: 0x%08lx\n",stage,(unsigned long)hr); std::exit(2); }
}
int wmain(int argc, wchar_t** argv) {
    if (argc != 5) return 2;
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
    std::array<float,64> ids{}; for(unsigned i=0;i<64;i++) ids[i]=float(i);
    D3D11_TEXTURE2D_DESC td{}; td.Width=td.Height=8;td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;
    td.Format=DXGI_FORMAT_R32_FLOAT;td.BindFlags=D3D11_BIND_SHADER_RESOURCE;td.Usage=D3D11_USAGE_IMMUTABLE;
    D3D11_SUBRESOURCE_DATA init{ids.data(),8*sizeof(float),0};
    ComPtr<ID3D11Texture2D> tex; ComPtr<ID3D11ShaderResourceView> srv;
    check(dev->CreateTexture2D(&td,&init,&tex),"texture");check(dev->CreateShaderResourceView(tex.Get(),nullptr,&srv),"srv");
    D3D11_BUFFER_DESC bd{};bd.ByteWidth=256;bd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=4;
    std::array<unsigned,64> sentinel;sentinel.fill(999);D3D11_SUBRESOURCE_DATA seed{sentinel.data(),0,0};
    ComPtr<ID3D11Buffer> out,back;ComPtr<ID3D11UnorderedAccessView> uav;
    check(dev->CreateBuffer(&bd,&seed,&out),"buffer"); check(dev->CreateUnorderedAccessView(out.Get(),nullptr,&uav),"uav");
    bd.BindFlags=bd.MiscFlags=bd.StructureByteStride=0;bd.Usage=D3D11_USAGE_STAGING;bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    check(dev->CreateBuffer(&bd,nullptr,&back),"staging");
    ctx->CSSetShader(shader.Get(),nullptr,0);ctx->CSSetShaderResources(0,1,srv.GetAddressOf());
    ctx->CSSetUnorderedAccessViews(0,1,uav.GetAddressOf(),nullptr);ctx->Dispatch(1,1,1);
    ctx->ClearState();ctx->CopyResource(back.Get(),out.Get());D3D11_MAPPED_SUBRESOURCE map{};
    check(ctx->Map(back.Get(),0,D3D11_MAP_READ,0,&map),"readback");
    const unsigned* got=(const unsigned*)map.pData;unsigned mismatches=0;
    int dx=_wtoi(argv[3]),dy=_wtoi(argv[4]);
    for(int i=0;i<64;i++){int x=i%8+dx,y=i/8+dy;unsigned expected=(x<0||y<0||x>=8||y>=8)?0:8*y+x;
        if(got[i]!=expected)++mismatches;}
    std::printf("adapter=%ls center=%u mismatches=%u/64\n",ad.Description,got[27],mismatches);
    ctx->Unmap(back.Get(),0);return mismatches?1:0;
}

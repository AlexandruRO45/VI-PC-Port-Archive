// Optional Windows test: runs src/dlss_backend.cpp (the DLSS backend the FFX shim reaches when LEONIDA_DLSS=1)
// end to end on a PC D3D12 device with procedural mock buffers. No game data anywhere: the color/depth/
// motion-vector contents are a checker, a near-plane value and zeros.
//
//   LEONIDA_NGX_DLL=<path>   full path to the driver's NGX Core nvngx.dll; found by name without it
// Expected next to the executable: nvngx_dlss.dll (the NVIDIA-distributed feature dll NGX loads; never shipped
// with Leonida).
//
// Steps, each only if the previous worked (77 when NGX or a device is missing):
//   1. init: NGX Core load, Init_ProjectID, capability (available, driver gate)
//   2. create: 1280x720 -> 1920x1080, MaxQuality, the flags the game's FSR4 context carries on 1.0.2.1
//      (HDR + MV low-res + inverted depth + auto exposure)
//   3. eval on three jitters; every call records NGX Success and the (execute+sync) wall time
//   4. destroy: ReleaseFeature, DestroyParameters, Shutdown1
// Nothing from NVIDIA is written to disk beyond what NGX itself logs into the app data path.
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_4.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "../src/dlss_backend.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

void note(const std::string& s) { std::printf("INFO %s\n", s.c_str()); }

constexpr unsigned kW = 1280, kH = 720, kOutW = 1920, kOutH = 1080;

ID3D12Device* create_pc_device() {
    IDXGIFactory1* factory = nullptr;
    ID3D12Device* device = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            std::printf("INFO adapter: %ls\n", d.Description);
            if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
                adapter->Release();
                break;
            }
        }
        adapter->Release();
        adapter = nullptr;
    }
    factory->Release();
    return device;
}

UINT64 g_fence_value = 0;

void wait_idle(ID3D12Device* dev, ID3D12CommandQueue* q) {
    ID3D12Fence* fence = nullptr;
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return;
    const UINT64 v = ++g_fence_value;
    q->Signal(fence, v);
    const HANDLE e = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(v, e);
    WaitForSingleObject(e, 10000);
    CloseHandle(e);
    fence->Release();
}

// __try cannot own objects with destructors (C2712): the guarded body is a free function.
unsigned long long g_fault_addr = 0;
unsigned guarded_backend_init(ID3D12Device* dev, const wchar_t* data_path, bool* ok) {
    __try {
        *ok = leonida::dlss::init(dev, data_path);
    } __except (g_fault_addr = reinterpret_cast<unsigned long long>(
                    GetExceptionInformation()->ExceptionRecord->ExceptionAddress),
                EXCEPTION_EXECUTE_HANDLER) {
        *ok = false;
        return GetExceptionCode();
    }
    return 0;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("dlss_dispatch_test: src/dlss_backend on mock buffers (no game data)\n");

    // SKIP when NGX cannot load (no NVIDIA driver): every later step needs it.
    {
        const char* override_path = std::getenv("LEONIDA_NGX_DLL");
        HMODULE probe = nullptr;
        if (override_path && *override_path) probe = LoadLibraryA(override_path);
        if (!probe && !GetModuleHandleA("nvngx.dll")) probe = LoadLibraryA("nvngx.dll");
        if (!probe) {
            std::printf("SKIP (no NGX Core; set LEONIDA_NGX_DLL=<path to nvngx.dll>)\n");
            return 77;
        }
    }
    if (HMODULE d3d = GetModuleHandleA("d3d12.dll"); !d3d) {  // device creation needs the system D3D12
        const bool have_device = create_pc_device() != nullptr;
        // if a device exists we fall through; only a truly missing D3D12 skips
        if (!have_device) {
            std::printf("SKIP (no D3D12 device)\n");
            return 77;
        }
    }

    ID3D12Device* dev = create_pc_device();
    check(dev != nullptr, "PC D3D12 device created");
    if (!dev) return 1;
    ID3D12CommandQueue* q = nullptr;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
    check(q != nullptr, "command queue created");
    ID3D12CommandAllocator* al = nullptr;
    ID3D12GraphicsCommandList* l = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al));
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, nullptr, IID_PPV_ARGS(&l));
    if (l) l->Close();  // created OPEN

    wchar_t appdata[MAX_PATH] = {};
    GetEnvironmentVariableW(L"LOCALAPPDATA", appdata, MAX_PATH);
    const std::wstring data_path = std::wstring(appdata) + L"\\LeonidaDlssProbe";
    std::error_code ec;
    std::filesystem::create_directories(data_path, ec);  // NGX init requires the data path to exist

    // 1. init + capability.
    std::printf("INFO calling leonida::dlss::init\n");
    bool init_ok = false;
    unsigned seh_code = guarded_backend_init(dev, data_path.c_str(), &init_ok);
    if (seh_code) {
        std::printf("FAIL backend init faulted with 0x%08x at 0x%012llx\n", seh_code,
                    static_cast<unsigned long long>(g_fault_addr));
        return 1;
    }
    check(init_ok, "NGX init + capability");

    // Mock inputs on DEFAULT heaps, staged from row-major upload buffers (one pass, before the evals).
    auto make_tex = [&](unsigned w, unsigned h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = w;
        d.Height = h;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = fmt;
        d.SampleDesc.Count = 1;
        d.Flags = flags;
        ID3D12Resource* r = nullptr;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        if (SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&r))))
            dev->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, nullptr);
        return std::pair<ID3D12Resource*, UINT64>{r, fp.Footprint.RowPitch};
    };
    auto make_upload = [&](UINT64 bytes) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* r = nullptr;
        unsigned char* map = nullptr;
        if (SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr, IID_PPV_ARGS(&r)))) {
            D3D12_RANGE none{0, 0};
            r->Map(0, &none, reinterpret_cast<void**>(&map));
        }
        return std::pair<ID3D12Resource*, unsigned char*>{r, map};
    };

    const uint16_t kHalfOne = 0x3c00;  // 1.0
    const auto [color, color_pitch] = make_tex(kW, kH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    const auto [depth, depth_pitch] = make_tex(kW, kH, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    const auto [mv, mv_pitch] = make_tex(kW, kH, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    const auto [output, output_pitch] =
        make_tex(kOutW, kOutH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    check(color && depth && mv && output, "mock textures created");
    if (!(color && depth && mv && output)) return 1;

    const auto [color_up, color_map] = make_upload(static_cast<UINT64>(color_pitch) * kH);
    const auto [depth_up, depth_map] = make_upload(static_cast<UINT64>(depth_pitch) * kH);
    const auto [mv_up, mv_map] = make_upload(static_cast<UINT64>(mv_pitch) * kH);
    check(color_up && depth_up && mv_up, "upload staging created");
    if (!(color_up && depth_up && mv_up)) return 1;
    if (color_map && depth_map && mv_map) {
        for (unsigned y = 0; y < kH; ++y) {
            auto row = reinterpret_cast<uint16_t*>(color_map + static_cast<size_t>(y) * color_pitch);
            for (unsigned x = 0; x < kW; ++x) {
                const bool on = ((x * 32 / kW) ^ (y * 18 / kH)) & 1;
                const uint16_t v = on ? 0x3abf : 0x2666;  // 0.8f / 0.05f as half
                row[x * 4 + 0] = v;
                row[x * 4 + 1] = v;
                row[x * 4 + 2] = v;
                row[x * 4 + 3] = kHalfOne;
            }
            std::memset(depth_map + static_cast<size_t>(y) * depth_pitch, 0xff, static_cast<size_t>(kW) * 4);
            std::memset(mv_map + static_cast<size_t>(y) * mv_pitch, 0, static_cast<size_t>(kW) * 4);
        }
    }
    {
        l->Reset(al, nullptr);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fc{};
        fc.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, kW, kH, 1, static_cast<UINT>(color_pitch)};
        D3D12_TEXTURE_COPY_LOCATION cs{color_up, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, fc};
        D3D12_TEXTURE_COPY_LOCATION cd{color, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        cd.SubresourceIndex = 0;
        l->CopyTextureRegion(&cd, 0, 0, 0, &cs, nullptr);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fd{};
        fd.Footprint = {DXGI_FORMAT_R32_FLOAT, kW, kH, 1, static_cast<UINT>(depth_pitch)};
        D3D12_TEXTURE_COPY_LOCATION ds{depth_up, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, fd};
        D3D12_TEXTURE_COPY_LOCATION dd{depth, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dd.SubresourceIndex = 0;
        l->CopyTextureRegion(&dd, 0, 0, 0, &ds, nullptr);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fm{};
        fm.Footprint = {DXGI_FORMAT_R16G16_FLOAT, kW, kH, 1, static_cast<UINT>(mv_pitch)};
        D3D12_TEXTURE_COPY_LOCATION ms{mv_up, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, fm};
        D3D12_TEXTURE_COPY_LOCATION md{mv, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        md.SubresourceIndex = 0;
        l->CopyTextureRegion(&md, 0, 0, 0, &ms, nullptr);
        D3D12_RESOURCE_BARRIER bar[3];
        for (int i = 0; i < 3; ++i) {
            bar[i] = {};
            bar[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bar[i].Transition.pResource = i == 0 ? color : i == 1 ? depth : mv;
            bar[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            bar[i].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            bar[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        l->ResourceBarrier(3, bar);
        l->Close();
        q->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList* const*>(&l));
        wait_idle(dev, q);
    }

    // 2. create, in the game's measured FSR4 configuration.
    leonida::dlss::Description desc{};
    desc.render_width = kW;
    desc.render_height = kH;
    desc.upscale_width = kOutW;
    desc.upscale_height = kOutH;
    desc.upscale_ratio = 1.5f;  // the game runs 1280x720 -> 1920x1080 at the FSR Quality slot
    desc.hdr = true;
    desc.depth_inverted = true;
    desc.auto_exposure = true;
    leonida::dlss::Feature f = leonida::dlss::create(l, desc, dev, 1);
    check(f.rc_create == nvngx::RESULT_SUCCESS && f.ngx != nullptr, "create (MaxQuality, game flags)");
    if (!(f.rc_create == nvngx::RESULT_SUCCESS && f.ngx)) return 1;
    note("quality slot " + std::to_string(f.quality) + ", flags " + std::to_string(f.flags));

    // 3. eval on three jitters.
    for (int frame = 0; frame < 3; ++frame) {
        leonida::dlss::FfxDescription fx{};
        fx.color = color;
        fx.output = output;
        fx.depth = depth;
        fx.motion_vectors = mv;
        fx.jitter_x = frame == 0 ? 0.0f : (frame == 1 ? 0.25f : -0.25f);
        fx.jitter_y = frame == 0 ? 0.0f : (frame == 1 ? -0.25f : 0.25f);
        fx.mv_scale_x = 1.0f;
        fx.mv_scale_y = 1.0f;
        fx.sharpness = 0.0f;
        fx.reset = frame == 0;
        l->Reset(al, nullptr);
        if (frame == 0) {  // the output is UAV for the first eval and stays there
            D3D12_RESOURCE_BARRIER ob{};
            ob.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            ob.Transition.pResource = output;
            ob.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            ob.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ob.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            l->ResourceBarrier(1, &ob);
        }
        const LARGE_INTEGER t0 = [] {
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            return t;
        }();
        const bool ok = leonida::dlss::eval_from_ffx(f, l, fx);
        const bool closed = SUCCEEDED(l->Close());
        q->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList* const*>(&l));
        wait_idle(dev, q);
        LARGE_INTEGER t1, freq;
        QueryPerformanceCounter(&t1);
        QueryPerformanceFrequency(&freq);
        char line[192];
        std::snprintf(line, sizeof(line), "eval[%d] NGX ok %d close %d | GPU wall %.3f ms (execute+sync)", frame,
                      ok ? 1 : 0, closed ? 1 : 0, (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart);
        note(line);
        check(ok, "eval frame returns Success");
    }

    // 4. destroy.
    leonida::dlss::destroy(f);
    check(f.ngx == nullptr && f.params == nullptr, "destroy clean");
    leonida::dlss::shutdown(dev);

    if (l) l->Release();
    if (al) al->Release();
    if (output) output->Release();
    if (mv) mv->Release();
    if (depth) depth->Release();
    if (color) color->Release();
    if (color_up) color_up->Release();
    if (depth_up) depth_up->Release();
    if (mv_up) mv_up->Release();
    if (q) q->Release();
    if (dev) dev->Release();

    std::printf("dlss_dispatch_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

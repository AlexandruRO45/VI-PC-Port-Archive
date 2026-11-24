#include "cpu/shim_gpu_device.h"
#include "desktop_gpu.h"
#include "startup_stubs/slice_footprints.h"

#include <array>

namespace {
constexpr UINT width = 17, height = 3;

unsigned char pixel(UINT frame, UINT row, UINT column)
{
    return static_cast<unsigned char>(17 * frame + 7 * row + column);
}

// Three CPU-backed textures rotate between successive frames. Reusing one after three
// presents must publish its new bytes even without an XG write notification.
void run(ID3D12Device* device, bool stale)
{
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_R8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    const x2w_startup::SliceFootprints plan(device, desc, 1);
    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, 3> textures;
    for (auto& texture : textures)
        texture = resource(device, desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto upload = make_buffer(device, plan.size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = make_buffer(device, plan.size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_RANGE none{};
    UINT wrong = 0;
    for (UINT frame = 0; frame < 6; ++frame) {
        ID3D12Resource* texture = textures[frame % textures.size()].Get();
        ComputeCommands commands(device, nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
        auto* list = commands.list.Get();
        const UINT pitch = plan.placed[0].Footprint.RowPitch;
        if (frame < textures.size() || (!stale && x2w_startup::recheck_cpu_texture(desc, plan.size, 3))) {
            void* mapped = nullptr;
            check(upload->Map(0, &none, &mapped), "Map R8 source");
            for (UINT y = 0; y < height; ++y)
                for (UINT x = 0; x < width; ++x)
                    static_cast<unsigned char*>(mapped)[pitch * y + x] = pixel(frame, y, x);
            upload->Unmap(0, nullptr);
            transition(list, texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            plan.upload(list, texture, upload.Get(), 1, 1, 0, 0, 0);
            transition(list, texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        }
        transition(list, texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION source{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION target{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        target.PlacedFootprint = plan.placed[0];
        list->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
        transition(list, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        commands.submit_and_wait(device);
        void* observed = nullptr;
        check(readback->Map(0, nullptr, &observed), "Map R8 result");
        for (UINT y = 0; y < height; ++y)
            for (UINT x = 0; x < width; ++x) {
                const auto actual = static_cast<unsigned char*>(observed)[pitch * y + x];
                const UINT expected_frame = stale ? frame % textures.size() : frame;
                if (actual != pixel(expected_frame, y, x)) throw std::runtime_error("Unexpected R8 pixel");
                wrong += actual != pixel(frame, y, x);
            }
        readback->Unmap(0, &none);
    }
    if (wrong != (stale ? 3 * width * height : 0)) throw std::runtime_error("Unexpected stale pixel count");
    std::printf("stale=%d, six frames, %u of %u pixels differ from the current CPU image\n", stale, wrong, 6 * width * height);

    auto other = desc;
    other.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    if (x2w_startup::recheck_cpu_texture(desc, plan.size, 0) ||
        !x2w_startup::recheck_cpu_texture(desc, plan.size, 1) ||
        x2w_startup::recheck_cpu_texture(desc, (UINT64{32} << 20) + 1, 1) ||
        x2w_startup::recheck_cpu_texture(other, plan.size, 14) ||
        !x2w_startup::recheck_cpu_texture(other, plan.size, 15))
        throw std::runtime_error("Unexpected CPU texture refresh interval");
}
}  // namespace

int main()
{
    try {
        auto device = shim_create_device();
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "Create observation factory");
        check(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)), "Identify test device");
        DXGI_ADAPTER_DESC1 description{};
        check(adapter->GetDesc1(&description), "Read test adapter");
        std::printf("adapter=%ls\n", description.Description);
        run(device.Get(), false);
        run(device.Get(), true);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    } catch (...) { return 1; }
    return 0;
}

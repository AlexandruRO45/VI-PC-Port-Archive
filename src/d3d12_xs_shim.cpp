// Desktop d3d12_xs.dll exporting the two root-signature functions the package DLLs import by name.
// Observed call sites (docs/experiments.md):
//   D3D12SerializeVersionedRootSignature: XboxBvhBuilder.dll (0xc2d0/0xc32f) copies a 48-byte
//     descriptor, sets console flag bits 0x03000000 at +0x28, and passes a null error output.
//   D3D12CreateRootSignatureDeserializer: amd_fidelityfx_upscaler_xbox.dll (0x4f88) reads a
//     serialized root signature from an embedded container and retrieves the descriptor interface.
// D3D12ConfigureVirtualMemoryX: called by the main executable after D3D12XboxCreateDevice.
//   The ERA DLL (GDK 261000) calls VdEraConfigureVirtualMemory then XMemRegisterGraphicsMemoryConfiguration.
//   The desktop shim records the page-table parameters and returns a success code.
//   Struct layout from the ERA d3d12_xs.dll PDB:
//     +0x00  UINT  Flags          (bit 0 observed; must be 0 or 1)
//     +0x04  UINT  GpuPageTablePages  (max 33 = 132 MB; shifted left 22 for XMemRegister)
//     +0x08  UINT  FramePreserveBufferSize
//     +0x10  UINT64 MinGpuAddress  (output, filled by VdEraConfigureVirtualMemory)
//     +0x18  UINT64 MaxGpuAddress  (output, filled by VdEraConfigureVirtualMemory)
#include <d3d12.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
// Observed console root-signature flag bits set by XboxBvhBuilder.dll.
// Unobserved bits are left untouched so desktop d3d12.dll still rejects unknown flags.
constexpr std::uint32_t observed_console_flags = 0x03000000u;

// _XMEM_GRAPHICS_MEMORY_CONFIG from the ERA xmem.dll PDB (24 bytes).
struct XMemGraphicsMemoryConfig {
    std::uint32_t gpu_page_table_size;
    std::uint32_t frame_preserve_buffer_size;
    std::uint64_t min_gpu_address;
    std::uint64_t max_gpu_address;
};
static_assert(sizeof(XMemGraphicsMemoryConfig) == 24);

std::atomic<bool> graphics_memory_configured{false};
XMemGraphicsMemoryConfig graphics_memory_config{};

// D3D12ConfigureVirtualMemoryX input layout from the ERA d3d12_xs.dll PDB.
struct D3D12VirtualMemoryConfig {
    std::uint32_t flags;
    std::uint32_t gpu_page_table_pages;
    std::uint32_t frame_preserve_buffer_size;
    std::uint32_t padding;
    std::uint64_t min_gpu_address;  // output
    std::uint64_t max_gpu_address;  // output
};
static_assert(sizeof(D3D12VirtualMemoryConfig) == 32);

static_assert(offsetof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC, Desc_1_0) +
              offsetof(D3D12_ROOT_SIGNATURE_DESC, Flags) == 0x28);
static_assert(offsetof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC, Desc_1_1) +
              offsetof(D3D12_ROOT_SIGNATURE_DESC1, Flags) == 0x28);
}

extern "C" HRESULT WINAPI d3d12_xs_SerializeVersionedRootSignature(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* root_signature,
    ID3DBlob** blob,
    ID3DBlob** error_blob)
{
    if (!root_signature) {
        return D3D12SerializeVersionedRootSignature(root_signature, blob, error_blob);
    }
    if (root_signature->Version == D3D_ROOT_SIGNATURE_VERSION_1_1) {
        const auto flags = static_cast<std::uint32_t>(root_signature->Desc_1_1.Flags);
        if (flags & observed_console_flags) {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC copy = *root_signature;
            copy.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags & ~observed_console_flags);
            return D3D12SerializeVersionedRootSignature(&copy, blob, error_blob);
        }
    } else if (root_signature->Version == D3D_ROOT_SIGNATURE_VERSION_1_0) {
        const auto flags = static_cast<std::uint32_t>(root_signature->Desc_1_0.Flags);
        if (flags & observed_console_flags) {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC copy = *root_signature;
            copy.Desc_1_0.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags & ~observed_console_flags);
            return D3D12SerializeVersionedRootSignature(&copy, blob, error_blob);
        }
    }
    return D3D12SerializeVersionedRootSignature(root_signature, blob, error_blob);
}

extern "C" HRESULT WINAPI d3d12_xs_CreateRootSignatureDeserializer(
    LPCVOID src_data,
    SIZE_T src_data_size_in_bytes,
    REFIID deserializer_interface,
    void** deserializer)
{
    return D3D12CreateRootSignatureDeserializer(
        src_data, src_data_size_in_bytes, deserializer_interface, deserializer);
}

extern "C" HRESULT WINAPI d3d12_xs_XboxCreateDevice(
    IUnknown* adapter, const void* parameters, REFIID iid, void** device)
{
    if (!device) return E_POINTER;
    *device = nullptr;
    // This entry supports an explicit desktop consumer of the returned interface.
    // The Xbox parameter block and the Xbox device vtable require their own implementation.
    if (parameters) return E_INVALIDARG;
    if (iid != __uuidof(ID3D12Device)) return E_NOINTERFACE;
    return D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, iid, device);
}

extern "C" void WINAPI d3d12_xs_RegisterGraphicsMemoryConfiguration(void* config_ptr)
{
    if (!config_ptr) return;
    if (graphics_memory_configured.exchange(true, std::memory_order_acq_rel)) return;
    std::memcpy(&graphics_memory_config, config_ptr, sizeof(graphics_memory_config));
}

extern "C" HRESULT WINAPI d3d12_xs_ConfigureVirtualMemoryX(void* config_ptr)
{
    if (!config_ptr) return E_INVALIDARG;
    auto* config = static_cast<D3D12VirtualMemoryConfig*>(config_ptr);
    if ((config->flags & ~1u) != 0) return E_INVALIDARG;
    if (config->gpu_page_table_pages > 33) return E_INVALIDARG;

    // On the console, VdEraConfigureVirtualMemory reserves GPU virtual address space and returns
    // the min/max addresses. On desktop, the GPU VA space is managed by the desktop D3D12 runtime.
    // Return a plausible range so callers that inspect these fields can proceed.
    config->min_gpu_address = 0;
    config->max_gpu_address = static_cast<std::uint64_t>(config->gpu_page_table_pages) << 32;

    // Register with xmem as the ERA DLL does: gpuPageTableSize = pages << 22.
    XMemGraphicsMemoryConfig xmem_config{};
    xmem_config.gpu_page_table_size = config->gpu_page_table_pages << 22;
    xmem_config.frame_preserve_buffer_size = config->frame_preserve_buffer_size;
    xmem_config.min_gpu_address = config->min_gpu_address;
    xmem_config.max_gpu_address = config->max_gpu_address;
    d3d12_xs_RegisterGraphicsMemoryConfiguration(&xmem_config);

    return static_cast<HRESULT>(0x10000000u); // ERA success code from the RE
}

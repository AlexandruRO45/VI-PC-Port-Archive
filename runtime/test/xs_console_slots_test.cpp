// The console order of the Xbox-ABI device and command-queue proxies (d3d12_xs_console_slots.h), over synthetic COM
// objects: no D3D12 runtime, no GPU, no game data. Every call is made by slot, as a title built against the console
// headers makes it, and checked against what reached the mock PC device or queue behind the proxy:
//   - device slots 44-82 and 104 are console methods: none of them reaches a PC method (the PC device has
//     ID3D12Device1..14 there), and the public Device1..8 methods are found 39/40 slots later (83-103, 105-112);
//   - each console method the proxy implements does what it should, with what it passed to the PC device;
//   - the command queue: the layout is taken from the first call that proves it, and every slot past
//     ExecuteCommandLists then means what that layout says. The layout is per process, so each one is a run:
//       xs_console_slots_test               device slots, then the 2020 kit's queue (Signal at 14)
//       xs_console_slots_test --later       the later kit's queue (Signal at 16)
//       xs_console_slots_test --present-32  the later kit, decided by a PresentX at 32 before any Signal
// Negative controls: the same counters must see the PC methods that a forwarded console slot does reach.
#include "../shims/d3d12_xs_device.h"

#include <d3d12.h>
#include <windows.h>

#include <io.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <utility>

namespace {

const char* g_label = "xs_console_slots";

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", g_label, __LINE__, #condition); return 1; } \
} while (0)

namespace xs = rt::d3d12_xs;

// ---- Mocks: a PC device, queue and fence that record every call ----

enum class Kind { device, queue, fence, other };

constexpr unsigned kMockSlots = 128;
constexpr LUID kMockLuid{0x89abcdef, 0x01234567};

struct Mock {
    void** vtable;
    Kind kind;
    long refs;
    unsigned calls[kMockSlots];
    std::uint64_t args[kMockSlots][7];
    // What some calls pointed at, copied while the call ran.
    D3D12_COMMAND_QUEUE_DESC queue_desc;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC graphics_desc;
    D3D12_SAMPLER_DESC sampler_desc;
    D3D12_RESOURCE_DESC resource_desc;
    D3D12_HEAP_TYPE heap_type;
    Mock* last_child;
};

Mock g_pool[64];
unsigned g_pool_used = 0;

void** mock_vtable();

Mock* new_mock(Kind kind) {
    if (g_pool_used == std::size(g_pool)) {
        std::fprintf(stderr, "%s: mock pool exhausted\n", g_label);
        std::exit(1);
    }
    Mock* m = &g_pool[g_pool_used++];
    std::memset(m, 0, sizeof(*m));
    m->vtable = mock_vtable();
    m->kind = kind;
    m->refs = 1;
    return m;
}

bool mock_has(const Mock* m, REFIID iid) {
    if (IsEqualIID(iid, __uuidof(IUnknown)) || IsEqualIID(iid, __uuidof(ID3D12Object))) return true;
    switch (m->kind) {
        case Kind::device: {
            // Up to ID3D12Device9 (76 PC slots): every Device1..8 method has a console slot.
            const IID levels[] = {__uuidof(ID3D12Device), __uuidof(ID3D12Device1), __uuidof(ID3D12Device2),
                                  __uuidof(ID3D12Device3), __uuidof(ID3D12Device4), __uuidof(ID3D12Device5),
                                  __uuidof(ID3D12Device6), __uuidof(ID3D12Device7), __uuidof(ID3D12Device8),
                                  __uuidof(ID3D12Device9)};
            for (const IID& level : levels)
                if (IsEqualIID(iid, level)) return true;
            return false;
        }
        case Kind::queue:
            return IsEqualIID(iid, __uuidof(ID3D12DeviceChild)) || IsEqualIID(iid, __uuidof(ID3D12Pageable)) ||
                   IsEqualIID(iid, __uuidof(ID3D12CommandQueue));
        case Kind::fence:
            return IsEqualIID(iid, __uuidof(ID3D12DeviceChild)) || IsEqualIID(iid, __uuidof(ID3D12Pageable)) ||
                   IsEqualIID(iid, __uuidof(ID3D12Fence));
        default: return false;
    }
}

HRESULT mock_query(Mock* m, const IID* riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!riid || !mock_has(m, *riid)) return E_NOINTERFACE;
    *out = m;
    ++m->refs;
    return S_OK;
}

HRESULT mock_create(Mock* device, Kind kind, std::uint64_t riid, std::uint64_t out) {
    if (!out) return S_FALSE;
    Mock* child = new_mock(kind);
    child->refs = 0;
    device->last_child = child;
    return mock_query(child, reinterpret_cast<const IID*>(riid), reinterpret_cast<void**>(out));
}

std::uint64_t mock_call(Mock* m, unsigned slot, const std::uint64_t* w) {
    if (m->kind != Kind::device) return S_OK;
    switch (slot) {
        case 8:  // CreateCommandQueue(desc, riid, out)
            if (w[0]) m->queue_desc = *reinterpret_cast<const D3D12_COMMAND_QUEUE_DESC*>(w[0]);
            return static_cast<std::uint32_t>(mock_create(m, Kind::queue, w[1], w[2]));
        case 10:  // CreateGraphicsPipelineState(desc, riid, out)
            m->graphics_desc = *reinterpret_cast<const D3D12_GRAPHICS_PIPELINE_STATE_DESC*>(w[0]);
            return static_cast<std::uint32_t>(mock_create(m, Kind::other, 0, 0));
        case 22:  // CreateSampler(desc, handle)
            m->sampler_desc = *reinterpret_cast<const D3D12_SAMPLER_DESC*>(w[0]);
            return 0;
        case 27:  // CreateCommittedResource(heap, flags, desc, state, clear, riid, out)
            m->heap_type = reinterpret_cast<const D3D12_HEAP_PROPERTIES*>(w[0])->Type;
            m->resource_desc = *reinterpret_cast<const D3D12_RESOURCE_DESC*>(w[2]);
            return S_OK;
        case 36:  // CreateFence(value, flags, riid, out)
            return static_cast<std::uint32_t>(mock_create(m, Kind::fence, w[2], w[3]));
        case 43:  // GetAdapterLuid through the hidden pointer, as the PC method returns it
            *reinterpret_cast<LUID*>(w[0]) = kMockLuid;
            return w[0];
        default: return S_OK;
    }
}

template <unsigned Slot>
std::uint64_t WINAPI mock_slot(Mock* self, std::uint64_t a1, std::uint64_t a2, std::uint64_t a3, std::uint64_t a4,
                               std::uint64_t a5, std::uint64_t a6, std::uint64_t a7) {
    if constexpr (Slot == 0) {
        return static_cast<std::uint32_t>(mock_query(self, reinterpret_cast<const IID*>(a1),
                                                     reinterpret_cast<void**>(a2)));
    } else if constexpr (Slot == 1) {
        return static_cast<std::uint64_t>(++self->refs);
    } else if constexpr (Slot == 2) {
        return static_cast<std::uint64_t>(--self->refs);
    } else {
        // The words are copied into this frame and passed by pointer, so no tail call writes the caller's frame
        // (xs_proxy_mock_test's mock, same reason).
        ++self->calls[Slot];
        const std::uint64_t words[] = {a1, a2, a3, a4, a5, a6, a7};
        std::memcpy(self->args[Slot], words, sizeof(words));
        return mock_call(self, Slot, words);
    }
}

template <std::size_t... I>
void fill_mock_vtable(void** vtable, std::index_sequence<I...>) {
    ((vtable[I] = reinterpret_cast<void*>(&mock_slot<static_cast<unsigned>(I)>)), ...);
}

void** mock_vtable() {
    static void* table[kMockSlots];
    static const bool filled = [] {
        fill_mock_vtable(table, std::make_index_sequence<kMockSlots>{});
        return true;
    }();
    (void)filled;
    return table;
}

// ---- Calls by slot ----

using SlotFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);

std::uint64_t raw(const void* object, unsigned slot, std::uint64_t a1 = 0, std::uint64_t a2 = 0,
                  std::uint64_t a3 = 0, std::uint64_t a4 = 0, std::uint64_t a5 = 0, std::uint64_t a6 = 0) {
    void* const* vtable = *reinterpret_cast<void* const* const*>(object);
    return reinterpret_cast<SlotFn>(vtable[slot])(const_cast<void*>(object), a1, a2, a3, a4, a5, a6, 0, 0, 0, 0, 0,
                                                  0, 0, 0);
}

HRESULT hr_of(std::uint64_t value) { return static_cast<HRESULT>(static_cast<std::uint32_t>(value)); }

template <class T>
std::uint64_t word(const T* p) { return reinterpret_cast<std::uint64_t>(p); }

const IID kQueueIid = __uuidof(ID3D12CommandQueue);
const IID kFenceIid = __uuidof(ID3D12Fence);
const IID kPipelineIid = __uuidof(ID3D12PipelineState);
const IID kResourceIid = __uuidof(ID3D12Resource);

void* make_device(Mock*& mock) {
    mock = new_mock(Kind::device);
    void* device = nullptr;
    if (FAILED(xs::create_device_proxy(reinterpret_cast<ID3D12Device*>(mock), __uuidof(ID3D12Device), &device)))
        return nullptr;
    return device;
}

// Calls PC slots 44-82 received: the PC ID3D12Device1..14 methods a console slot must never reach.
unsigned pc_device1_calls(const Mock* mock) {
    unsigned calls = 0;
    for (unsigned slot = 44; slot <= 82; ++slot) calls += mock->calls[slot];
    return calls;
}

// stderr of the calls made by run, through a file (the proxy reports on stderr).
template <class Run>
std::string captured_stderr(Run run) {
    char dir[MAX_PATH] = {}, path[MAX_PATH] = {};
    if (!GetTempPathA(MAX_PATH, dir) || !GetTempFileNameA(dir, "xcs", 0, path)) return {};
    std::fflush(stderr);
    const int saved = _dup(2);
    FILE* sink = nullptr;
    if (fopen_s(&sink, path, "w") != 0 || !sink) return {};
    _dup2(_fileno(sink), 2);
    run();
    std::fflush(stderr);
    _dup2(saved, 2);
    _close(saved);
    std::fclose(sink);
    std::string text;
    FILE* in = nullptr;
    if (fopen_s(&in, path, "r") == 0 && in) {
        char buffer[512];
        std::size_t n = 0;
        while ((n = std::fread(buffer, 1, sizeof(buffer), in)) > 0) text.append(buffer, n);
        std::fclose(in);
    }
    DeleteFileA(path);
    return text;
}

// ---- 1. The console block 44-82 never reaches a PC method; Device1..8 are found after it ----

int check_device_order() {
    Mock* mock = nullptr;
    void* device = make_device(mock);
    EXPECT(device);
    // Null arguments throughout: every console handler refuses them before it would use them.
    // 57 is left for check_device_methods, which reads its one report.
    for (unsigned slot = 44; slot <= 82; ++slot)
        if (slot != 57) raw(device, slot);
    raw(device, 104);
    for (unsigned slot = 113; slot < 128; ++slot) EXPECT(hr_of(raw(device, slot)) == E_NOTIMPL);
    EXPECT(pc_device1_calls(mock) == 0);
    for (unsigned slot = 83; slot < kMockSlots; ++slot) EXPECT(mock->calls[slot] == 0);

    // Negative control: the same counters see a PC method as soon as a console slot forwards to one.
    for (unsigned slot = 83; slot <= 103; ++slot) {
        if (slot == 84 || slot == 85 || slot == 89) continue;  // object arrays, see xs_proxy_mock_test
        raw(device, slot);
        EXPECT(mock->calls[slot - 39] == 1);
    }
    for (unsigned slot = 105; slot <= 112; ++slot) {
        raw(device, slot);
        EXPECT(mock->calls[slot - 40] == 1);
    }
    EXPECT(pc_device1_calls(mock) == 21 - 3 + 8);
    EXPECT(xs::device_slot_overridden(78) && !xs::device_slot_overridden(86));
    std::printf("%s: device slots 44-82, 104 and 113-127 reach no PC method; 83-103 and 105-112 reach PC 44-72\n",
                g_label);
    EXPECT(raw(device, 2) == 0);
    return 0;
}

// ---- 2. The console methods the proxy implements ----

int check_device_methods() {
    Mock* mock = nullptr;
    void* device = make_device(mock);
    EXPECT(device);

    // 43 GetAdapterLuid: the LUID comes back in RAX; the PC method was given a buffer of the proxy's.
    const std::uint64_t luid = raw(device, 43);
    EXPECT(luid == (static_cast<std::uint64_t>(kMockLuid.HighPart) << 32 | kMockLuid.LowPart));
    EXPECT(mock->calls[43] == 1 && mock->args[43][0] != 0);

    // 60 CreateGraphicsPipelineStateX: the public description, the console's flag bits removed.
    D3D12_GRAPHICS_PIPELINE_STATE_DESC graphics{};
    graphics.NumRenderTargets = 3;
    graphics.SampleMask = 0xf0f0;
    graphics.Flags = static_cast<D3D12_PIPELINE_STATE_FLAGS>(D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG | 0x10000);
    void* pipeline = nullptr;
    EXPECT(SUCCEEDED(hr_of(raw(device, 60, word(&graphics), 7, 0, word(&kPipelineIid), word(&pipeline)))));
    EXPECT(mock->calls[10] == 1 && mock->graphics_desc.NumRenderTargets == 3 &&
           mock->graphics_desc.SampleMask == 0xf0f0 && mock->graphics_desc.Flags == D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG);

    // 61 CreateSamplerX: a per-axis console sampler to the PC filter word.
    struct ConsoleSampler {
        UINT flags, mag, min, mip, z, reduction;
        D3D12_TEXTURE_ADDRESS_MODE u, v, w;
        FLOAT min_lod, max_lod, bias, bias2;
        UINT anisotropy;
        FLOAT anisotropy_bias;
        UINT threshold;
        D3D12_COMPARISON_FUNC comparison;
        FLOAT border[4];
    };
    const D3D12_CPU_DESCRIPTOR_HANDLE handle{0x5000};
    ConsoleSampler sampler{0, 1, 1, 2, 0, 0, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
                           static_cast<D3D12_TEXTURE_ADDRESS_MODE>(0), 1.0f, 8.0f, 0.5f, 0.0f, 1, 0.0f, 0,
                           D3D12_COMPARISON_FUNC_LESS, {0.25f, 0.5f, 0.75f, 1.0f}};
    raw(device, 61, word(&sampler), handle.ptr);
    EXPECT(mock->calls[22] == 1 && mock->args[22][1] == handle.ptr);
    EXPECT(mock->sampler_desc.Filter == D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR);
    EXPECT(mock->sampler_desc.AddressU == D3D12_TEXTURE_ADDRESS_MODE_WRAP &&
           mock->sampler_desc.AddressV == D3D12_TEXTURE_ADDRESS_MODE_MIRROR &&
           mock->sampler_desc.AddressW == D3D12_TEXTURE_ADDRESS_MODE_CLAMP);  // 0 is no PC mode
    EXPECT(mock->sampler_desc.ComparisonFunc == D3D12_COMPARISON_FUNC_LESS && mock->sampler_desc.MipLODBias == 0.5f &&
           mock->sampler_desc.MinLOD == 1.0f && mock->sampler_desc.MaxLOD == 8.0f &&
           mock->sampler_desc.BorderColor[2] == 0.75f);
    sampler = ConsoleSampler{0, 0, 1, 0, 0, 2, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                             D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 2.0f, 8.0f, 0.0f, 0.0f, 1, 0.0f, 0,
                             D3D12_COMPARISON_FUNC_NEVER, {}};
    raw(device, 61, word(&sampler), handle.ptr);
    EXPECT(mock->sampler_desc.Filter == D3D12_FILTER_MAXIMUM_MIN_LINEAR_MAG_MIP_POINT);
    EXPECT(mock->sampler_desc.MaxLOD == 2.0f);  // no mip filter: the base level only
    sampler.mag = 3;
    sampler.anisotropy = 64;
    raw(device, 61, word(&sampler), handle.ptr);
    EXPECT(mock->sampler_desc.Filter == D3D12_FILTER_MAXIMUM_ANISOTROPIC && mock->sampler_desc.MaxAnisotropy == 16);

    // 68 GetGpuHardwareConfigurationX: two words written, nothing past them.
    std::uint64_t configuration[3] = {0, 0, 0x5a5a};
    raw(device, 68, word(configuration));
    EXPECT(configuration[0] != 0 && configuration[1] == 5 && configuration[2] == 0x5a5a);

    // 74/75 page pools: a handle of the proxy's, which unregisters once.
    std::uint64_t pool = 0;
    EXPECT(hr_of(raw(device, 74, 0x7f0000000000ull, 16, word(&pool))) == S_OK);
    EXPECT(pool != 0 && pool != 0x7f0000000000ull);
    EXPECT(hr_of(raw(device, 75, pool)) == S_OK && hr_of(raw(device, 75, pool)) == E_INVALIDARG);
    EXPECT(hr_of(raw(device, 74, 0x1000, 1, 0)) == E_POINTER);

    // 44 CreatePlacedResourceX and 76 CreateCommittedResourceX: committed PC resources with the layout word replaced.
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = 64;
    texture.Height = 32;
    texture.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
    void* resource = nullptr;
    raw(device, 44, 0x7f0000010000ull, word(&texture), D3D12_RESOURCE_STATE_COMMON, 0, word(&kResourceIid),
        word(&resource));
    EXPECT(mock->calls[27] == 1 && mock->heap_type == D3D12_HEAP_TYPE_DEFAULT && mock->resource_desc.Width == 64 &&
           mock->resource_desc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN);

    // 111 CreateSamplerFeedbackUnorderedAccessViewX: the view description is dropped, the handle moves up.
    raw(device, 111, 0x1111, 0x2222, 0, 0x3333);
    EXPECT(mock->calls[71] == 1 && mock->args[71][0] == 0x1111 && mock->args[71][1] == 0x2222 &&
           mock->args[71][2] == 0x3333);

    // 49 (a driver hint) succeeds with no effect; 57 (named by no source) is refused and reported with its caller.
    EXPECT(hr_of(raw(device, 49, 1, 2)) == S_OK);
    HRESULT unknown = S_OK;
    const std::string report = captured_stderr([&] { unknown = hr_of(raw(device, 57, 0xabc)); });
    EXPECT(unknown == E_NOTIMPL);
    EXPECT(report.find("device slot 57") != std::string::npos && report.find("xs_console_slots_test") != std::string::npos &&
           report.find("0xabc") != std::string::npos);
    EXPECT(pc_device1_calls(mock) == 1);  // only 111's PC 71

    EXPECT(raw(device, 2) == 0);
    std::printf("%s: console device methods 43, 44, 49, 57, 60, 61, 68, 74/75, 111 behave as named\n",
                g_label);
    return 0;
}

// ---- 3. Frame statistics at 81; the public HLE routes at 73 and 78-80 win over the table, and reach no PC method ----

int check_frame_statistics() {
    Mock* mock = nullptr;
    void* device = make_device(mock);
    EXPECT(device);
    std::uint32_t count = 4;
    unsigned char statistics[4 * 32] = {};
    EXPECT(hr_of(raw(device, 81, 1, 0xff, word(&count), word(statistics))) == S_FALSE && count == 0);
    EXPECT(hr_of(raw(device, 81, 1, 0xff, 0, 0)) == E_INVALIDARG);
    // #6935's routes answer before the console table: frame pacing S_OK, a null token or list description refused by
    // the route's own checks. None reaches a PC method (PC 73 and 78-80 are Device1+ methods; 12 is CreateCommandList).
    EXPECT(hr_of(raw(device, 78, 0, 16667, 1, 0)) == S_OK);
    EXPECT(hr_of(raw(device, 79)) == S_OK);
    EXPECT(hr_of(raw(device, 80)) == E_POINTER);
    EXPECT(hr_of(raw(device, 73)) == E_INVALIDARG);
    EXPECT(pc_device1_calls(mock) == 0 && mock->calls[8] == 0 && mock->calls[12] == 0);
    EXPECT(raw(device, 2) == 0);
    std::printf("%s: 81 answers no statistics yet; the routes at 73 and 78-80 answer without reaching a PC method\n",
                g_label);
    return 0;
}

// ---- 4. The command queue layouts ----

struct QueueRig {
    Mock* device_mock = nullptr;
    void* device = nullptr;
    void* queue = nullptr;
    void* fence = nullptr;
    Mock* queue_mock = nullptr;
    std::uint64_t real_fence = 0;
};

int make_rig(QueueRig& rig) {
    rig.device = make_device(rig.device_mock);
    EXPECT(rig.device);
    static const D3D12_COMMAND_QUEUE_DESC desc{};
    EXPECT(hr_of(raw(rig.device, 8, word(&desc), word(&kQueueIid), word(&rig.queue))) == S_OK);
    EXPECT(hr_of(raw(rig.device, 36, 0, 0, word(&kFenceIid), word(&rig.fence))) == S_OK);
    rig.queue_mock = reinterpret_cast<Mock*>(xs::queue_proxy_real_queue(rig.queue));
    rig.real_fence = word(xs::fence_proxy_real_fence(rig.fence));
    EXPECT(rig.queue_mock && rig.real_fence);
    return 0;
}

int queue_calls(const QueueRig& rig) {
    int calls = 0;
    for (unsigned slot = 3; slot < kMockSlots; ++slot) calls += static_cast<int>(rig.queue_mock->calls[slot]);
    return calls;
}

int check_queue_kit2020() {
    QueueRig rig;
    if (make_rig(rig)) return 1;
    D3D12_COMMAND_QUEUE_DESC desc{};
    // Before the layout is known, a slot that differs between the layouts does nothing.
    EXPECT(hr_of(raw(rig.queue, 26, word(&desc))) == S_OK && queue_calls(rig) == 0);
    // Signal at 14 with a fence decides it, and reaches the PC Signal with the real fence.
    EXPECT(hr_of(raw(rig.queue, 14, word(rig.fence), 5)) == S_OK);
    EXPECT(rig.queue_mock->calls[14] == 1 && rig.queue_mock->args[14][0] == rig.real_fence &&
           rig.queue_mock->args[14][1] == 5);
    raw(rig.queue, 11, 0, 0, 0);  // SetMarker, public
    EXPECT(rig.queue_mock->calls[11] == 1);
    raw(rig.queue, 15, word(rig.fence), 6);  // Wait
    EXPECT(rig.queue_mock->calls[15] == 1 && rig.queue_mock->args[15][0] == rig.real_fence);
    raw(rig.queue, 23, word(rig.fence), 7, 1);  // SignalX: the public Signal
    EXPECT(rig.queue_mock->calls[14] == 2 && rig.queue_mock->args[14][0] == rig.real_fence &&
           rig.queue_mock->args[14][1] == 7);
    raw(rig.queue, 26, word(&desc));  // GetDesc, through the PC slot 18 with the hidden pointer
    EXPECT(rig.queue_mock->calls[18] == 1 && rig.queue_mock->args[18][0] == word(&desc));
    // PIX, SuspendX/ResumeX and the rest: accepted; PC 16 and 17 (timestamps) are not these.
    for (unsigned slot : {16u, 17u, 18u, 19u, 20u, 21u, 22u, 24u, 25u, 31u}) EXPECT(hr_of(raw(rig.queue, slot)) == S_OK);
    EXPECT(hr_of(raw(rig.queue, 30, 0, 0, 0)) == S_OK);  // PresentX: accepted, not shown yet
    for (unsigned slot : {27u, 28u, 29u}) EXPECT(hr_of(raw(rig.queue, slot)) == E_NOTIMPL);
    EXPECT(rig.queue_mock->calls[16] == 0 && rig.queue_mock->calls[17] == 0 && rig.queue_mock->calls[18] == 1);
    EXPECT(queue_calls(rig) == 5);
    std::printf("%s: queue, 2020 kit: Signal 14 decides; SignalX 23 and GetDesc 26 forwarded, 16-31 never PC\n",
                g_label);
    return 0;
}

int check_queue_later(bool by_present) {
    QueueRig rig;
    if (make_rig(rig)) return 1;
    // Slot 14 with something other than a fence: one of the later kit's two new methods, or a marker.
    EXPECT(hr_of(raw(rig.queue, 14, 0x40, 1)) == S_OK && queue_calls(rig) == 0);
    if (by_present) EXPECT(hr_of(raw(rig.queue, 32, 0, 0, 0)) == S_OK && queue_calls(rig) == 0);
    EXPECT(hr_of(raw(rig.queue, 16, word(rig.fence), 9)) == S_OK);  // Signal
    EXPECT(rig.queue_mock->calls[14] == 1 && rig.queue_mock->args[14][0] == rig.real_fence &&
           rig.queue_mock->args[14][1] == 9);
    raw(rig.queue, 17, word(rig.fence), 9);  // Wait
    EXPECT(rig.queue_mock->calls[15] == 1);
    raw(rig.queue, 25, word(rig.fence), 10, 0);  // SignalX
    EXPECT(rig.queue_mock->calls[14] == 2 && rig.queue_mock->args[14][1] == 10);
    D3D12_COMMAND_QUEUE_DESC desc{};
    raw(rig.queue, 28, word(&desc));  // GetDesc
    EXPECT(rig.queue_mock->calls[18] == 1);
    // After the decision a fence at 14 is not a Signal, and 11-15 reach nothing.
    for (unsigned slot = 11; slot <= 15; ++slot) EXPECT(hr_of(raw(rig.queue, slot, word(rig.fence))) == S_OK);
    EXPECT(hr_of(raw(rig.queue, 32, 0, 0, 0)) == S_OK);  // PresentX
    EXPECT(hr_of(raw(rig.queue, 30)) == E_NOTIMPL);      // the 2020 kit's 28: a page-mapping method
    EXPECT(queue_calls(rig) == 4);
    std::printf("%s: queue, later kit%s: Signal 16, Wait 17, SignalX 25, GetDesc 28 forwarded; 11-15 never PC\n",
                g_label, by_present ? " (decided by PresentX at 32)" : "");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "--later") == 0) return check_queue_later(false);
    if (std::strcmp(mode, "--present-32") == 0) return check_queue_later(true);
    if (*mode) {
        std::fprintf(stderr, "%s: unknown option %s\n", g_label, mode);
        return 2;
    }
    if (check_device_order() || check_device_methods() || check_frame_statistics() || check_queue_kit2020()) return 1;
    std::printf("%s: all passed\n", g_label);
    return 0;
}

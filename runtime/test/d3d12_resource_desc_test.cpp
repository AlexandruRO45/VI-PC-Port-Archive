// Compare ID3D12Resource::GetDesc on the Vulkan-backed device with Windows D3D12 (the layer's own adapter and WARP):
// what a description that leaves fields at their defaults (Alignment 0, MipLevels 0) reads back as. Also the newer
// interfaces of resources and heaps: QueryInterface for ID3D12Resource1/2 and ID3D12Heap1 on committed, placed and
// reserved resources, GetDesc1 against GetDesc, and GetProtectedResourceSession on an object with no session.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Case {
    const char* name;
    D3D12_RESOURCE_DESC desc;
    D3D12_HEAP_TYPE heap;  // of a committed resource
    D3D12_RESOURCE_STATES state;
    bool reserved;
};

D3D12_RESOURCE_DESC buffer(UINT64 bytes, UINT64 alignment = 0, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = alignment;
    desc.Width = bytes;
    desc.Height = desc.SampleDesc.Count = 1;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC texture(D3D12_RESOURCE_DIMENSION dimension, UINT64 width, UINT height, UINT16 depth_or_layers,
                            UINT16 mips, DXGI_FORMAT format, UINT samples = 1,
                            D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, UINT64 alignment = 0) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = dimension;
    desc.Alignment = alignment;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth_or_layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = samples;
    desc.Flags = flags;
    return desc;
}

HRESULT create(ID3D12Device* device, const Case& c, ID3D12Resource** out) {
    if (c.reserved) return device->CreateReservedResource(&c.desc, c.state, nullptr, IID_PPV_ARGS(out));
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = c.heap;
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &c.desc, c.state, nullptr, IID_PPV_ARGS(out));
}

bool same(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b) {
    return a.Dimension == b.Dimension && a.Alignment == b.Alignment && a.Width == b.Width && a.Height == b.Height &&
           a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels == b.MipLevels && a.Format == b.Format &&
           a.SampleDesc.Count == b.SampleDesc.Count && a.SampleDesc.Quality == b.SampleDesc.Quality &&
           a.Layout == b.Layout && a.Flags == b.Flags;
}

void print(const char* who, const D3D12_RESOURCE_DESC& d) {
    std::fprintf(stderr,
                 "  %-6s dimension %d alignment %llu %llux%ux%u mips %u format %d samples %u/%u layout %d flags 0x%x\n",
                 who, int(d.Dimension), static_cast<unsigned long long>(d.Alignment),
                 static_cast<unsigned long long>(d.Width), d.Height, unsigned(d.DepthOrArraySize), unsigned(d.MipLevels),
                 int(d.Format), d.SampleDesc.Count, d.SampleDesc.Quality, int(d.Layout), unsigned(d.Flags));
}

struct Totals {
    unsigned matched = 0, mismatched = 0, host_refused = 0, ours_refused = 0;
};

// Every case the reference creates must read back the same from the layer. A case the reference refuses says
// nothing about GetDesc and is skipped; one only the layer refuses is counted (dimensions it does not create yet).
Totals compare(ID3D12Device* ours, ID3D12Device* host, const char* host_name, const std::vector<Case>& cases) {
    Totals totals;
    for (const Case& c : cases) {
        Ref<ID3D12Resource> expected, actual;
        if (FAILED(create(host, c, &expected.p)) || !expected.p) {
            ++totals.host_refused;
            continue;
        }
        const HRESULT hr = create(ours, c, &actual.p);
        if (FAILED(hr) || !actual.p) {
            std::fprintf(stderr, "%s: not created by the layer (0x%08lx), not compared with %s\n", c.name,
                         static_cast<unsigned long>(hr), host_name);
            ++totals.ours_refused;
            continue;
        }
        const D3D12_RESOURCE_DESC a = actual->GetDesc(), e = expected->GetDesc();
        if (same(a, e)) {
            ++totals.matched;
            continue;
        }
        ++totals.mismatched;
        std::fprintf(stderr, "%s: GetDesc differs from %s\n", c.name, host_name);
        print("asked", c.desc);
        print("ours", a);
        print("host", e);
    }
    return totals;
}

// ---- Resource1, Resource2 and Heap1: QueryInterface, GetDesc1 and GetProtectedResourceSession ----

struct Prot {
    HRESULT valid = 0, other = 0, null_out = 0;
    bool valid_nulled = false, other_nulled = false, null_crashed = false;
};

// What one resource answers. Every value is compared between the layer and the host.
struct ResourceAnswers {
    HRESULT qi1 = 0, qi2 = 0;
    bool identity = false;                  // IUnknown of the resource, Resource1 and Resource2 is one pointer
    long refs_added1 = 0, refs_added2 = 0;  // reference count growth of each successful QueryInterface
    Prot prot;
    bool have_desc1 = false;
    D3D12_RESOURCE_DESC desc{};
    D3D12_RESOURCE_DESC1 desc1{};
};

struct HeapAnswers {
    HRESULT qi1 = 0;
    bool identity = false;
    long refs_added1 = 0;
    Prot prot;
};

ULONG count_of(IUnknown* object) {
    object->AddRef();
    return object->Release();
}

IUnknown* identity_of(IUnknown* object) {
    IUnknown* unknown = nullptr;
    if (FAILED(object->QueryInterface(IID_PPV_ARGS(&unknown)))) return nullptr;
    unknown->Release();
    return unknown;
}

// A null out pointer is in a function of its own: it uses structured exception handling, so nothing with a
// destructor may live in it.
template<class T>
HRESULT null_out_call(T* object, bool* crashed) {
    __try {
        return object->GetProtectedResourceSession(__uuidof(ID3D12ProtectedResourceSession), nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *crashed = true;
        return E_FAIL;
    }
}

template<class T>
Prot probe_session(T* object, bool probe_null) {
    Prot prot;
    void* const sentinel = reinterpret_cast<void*>(static_cast<std::uintptr_t>(1));
    void* out = sentinel;
    prot.valid = object->GetProtectedResourceSession(__uuidof(ID3D12ProtectedResourceSession), &out);
    prot.valid_nulled = !out;
    out = sentinel;
    prot.other = object->GetProtectedResourceSession(__uuidof(ID3D12Device), &out);
    prot.other_nulled = !out;
    if (probe_null) prot.null_out = null_out_call(object, &prot.null_crashed);
    return prot;
}

ResourceAnswers answers_of(ID3D12Resource* resource, bool probe_null) {
    ResourceAnswers a;
    a.desc = resource->GetDesc();
    IUnknown* const unknown = identity_of(resource);
    a.identity = true;
    const ULONG before = count_of(resource);
    ID3D12Resource1* r1 = nullptr;
    a.qi1 = resource->QueryInterface(IID_PPV_ARGS(&r1));
    if (SUCCEEDED(a.qi1) && r1) {
        a.refs_added1 = long(count_of(resource)) - long(before);
        a.identity = a.identity && identity_of(r1) == unknown;
        a.prot = probe_session(r1, probe_null);
        r1->Release();
    }
    const ULONG before2 = count_of(resource);
    ID3D12Resource2* r2 = nullptr;
    a.qi2 = resource->QueryInterface(IID_PPV_ARGS(&r2));
    if (SUCCEEDED(a.qi2) && r2) {
        a.refs_added2 = long(count_of(resource)) - long(before2);
        a.identity = a.identity && identity_of(r2) == unknown;
        a.desc1 = r2->GetDesc1();
        a.have_desc1 = true;
        r2->Release();
    }
    return a;
}

HeapAnswers answers_of(ID3D12Heap* heap, bool probe_null) {
    HeapAnswers a;
    IUnknown* const unknown = identity_of(heap);
    a.identity = true;
    const ULONG before = count_of(heap);
    ID3D12Heap1* h1 = nullptr;
    a.qi1 = heap->QueryInterface(IID_PPV_ARGS(&h1));
    if (SUCCEEDED(a.qi1) && h1) {
        a.refs_added1 = long(count_of(heap)) - long(before);
        a.identity = identity_of(h1) == unknown;
        a.prot = probe_session(h1, probe_null);
        h1->Release();
    }
    return a;
}

// The fields GetDesc1 shares with GetDesc.
template<class A, class B>
bool same_base(const A& a, const B& b) {
    return a.Dimension == b.Dimension && a.Alignment == b.Alignment && a.Width == b.Width && a.Height == b.Height &&
           a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels == b.MipLevels && a.Format == b.Format &&
           a.SampleDesc.Count == b.SampleDesc.Count && a.SampleDesc.Quality == b.SampleDesc.Quality &&
           a.Layout == b.Layout && a.Flags == b.Flags;
}

bool same(const D3D12_RESOURCE_DESC1& a, const D3D12_RESOURCE_DESC1& b) {
    return same_base(a, b) && a.SamplerFeedbackMipRegion.Width == b.SamplerFeedbackMipRegion.Width &&
           a.SamplerFeedbackMipRegion.Height == b.SamplerFeedbackMipRegion.Height &&
           a.SamplerFeedbackMipRegion.Depth == b.SamplerFeedbackMipRegion.Depth;
}

bool same(const Prot& a, const Prot& b) {
    // A null out pointer is not compared: the host dereferences it (access violation); see null_out_ok.
    return a.valid == b.valid && a.other == b.other && a.valid_nulled == b.valid_nulled &&
           a.other_nulled == b.other_nulled;
}

bool same(const ResourceAnswers& a, const ResourceAnswers& b) {
    return a.qi1 == b.qi1 && a.qi2 == b.qi2 && a.identity == b.identity && a.refs_added1 == b.refs_added1 &&
           a.refs_added2 == b.refs_added2 && same(a.prot, b.prot) && a.have_desc1 == b.have_desc1 &&
           (!a.have_desc1 || (same(a.desc1, b.desc1) && same_base(a.desc1, a.desc) == same_base(b.desc1, b.desc)));
}

bool same(const HeapAnswers& a, const HeapAnswers& b) {
    return a.qi1 == b.qi1 && a.identity == b.identity && a.refs_added1 == b.refs_added1 && same(a.prot, b.prot);
}

void print(const char* who, const Prot& p) {
    std::fprintf(stderr,
                 "  %-6s session: valid 0x%08lx nulled %d, other 0x%08lx nulled %d, null out 0x%08lx crashed %d\n", who,
                 static_cast<unsigned long>(p.valid), p.valid_nulled, static_cast<unsigned long>(p.other),
                 p.other_nulled, static_cast<unsigned long>(p.null_out), p.null_crashed);
}

void print(const char* who, const ResourceAnswers& a) {
    std::fprintf(stderr, "  %-6s QI Resource1 0x%08lx (+%ld) Resource2 0x%08lx (+%ld) identity %d; GetDesc1 %s", who,
                 static_cast<unsigned long>(a.qi1), a.refs_added1, static_cast<unsigned long>(a.qi2), a.refs_added2,
                 a.identity, a.have_desc1 ? "present" : "unavailable");
    if (a.have_desc1) {
        std::fprintf(stderr, ", alignment %llu, mip region %u/%u/%u, base fields as GetDesc: %d",
                     static_cast<unsigned long long>(a.desc1.Alignment), a.desc1.SamplerFeedbackMipRegion.Width,
                     a.desc1.SamplerFeedbackMipRegion.Height, a.desc1.SamplerFeedbackMipRegion.Depth,
                     int(same_base(a.desc1, a.desc)));
    }
    std::fprintf(stderr, "\n");
    print(who, a.prot);
}

void print(const char* who, const HeapAnswers& a) {
    std::fprintf(stderr, "  %-6s QI Heap1 0x%08lx (+%ld) identity %d\n", who, static_cast<unsigned long>(a.qi1),
                 a.refs_added1, a.identity);
    print(who, a.prot);
}

D3D12_HEAP_PROPERTIES default_properties() {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    return heap;
}

D3D12_RESOURCE_DESC msaa_target() {
    return texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 256, 256, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 4,
                   D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
}

D3D12_RESOURCE_DESC small_texture() {
    return texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
}

D3D12_RESOURCE_DESC tiled_texture() {
    D3D12_RESOURCE_DESC desc = texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1024, 1024, 1, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    return desc;
}

D3D12_RESOURCE_DESC1 as_desc1(const D3D12_RESOURCE_DESC& desc) {
    D3D12_RESOURCE_DESC1 out{};
    std::memcpy(&out, &desc, sizeof desc);  // the DESC1 layout is the DESC one plus the trailing mip region
    return out;
}

HRESULT make_heap(ID3D12Device* device, UINT64 bytes, UINT64 alignment, D3D12_HEAP_FLAGS flags, ID3D12Heap** out) {
    D3D12_HEAP_DESC desc{};
    desc.SizeInBytes = bytes;
    desc.Properties = default_properties();
    desc.Alignment = alignment;
    desc.Flags = flags;
    return device->CreateHeap(&desc, IID_PPV_ARGS(out));
}

struct Made {
    Ref<ID3D12Heap> heap;  // of a placed resource
    Ref<ID3D12Resource> resource;
};

struct Kind {
    const char* name;
    HRESULT (*make)(ID3D12Device* device, Made* made);
};

const Kind kKinds[] = {
    {"committed buffer",
     [](ID3D12Device* device, Made* made) {
         const auto heap = default_properties();
         const auto desc = buffer(256);
         return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&made->resource.p));
     }},
    {"committed 2D texture",
     [](ID3D12Device* device, Made* made) {
         const auto heap = default_properties();
         const auto desc = small_texture();
         return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&made->resource.p));
     }},
    {"committed MSAA texture, Alignment 0",
     [](ID3D12Device* device, Made* made) {
         const auto heap = default_properties();
         const auto desc = msaa_target();
         return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                IID_PPV_ARGS(&made->resource.p));
     }},
    {"CreateCommittedResource2 texture",
     [](ID3D12Device* device, Made* made) {
         Ref<ID3D12Device8> device8;
         if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device8.p)))) return E_NOINTERFACE;
         const auto heap = default_properties();
         const auto desc = as_desc1(small_texture());
         return device8->CreateCommittedResource2(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr, nullptr, IID_PPV_ARGS(&made->resource.p));
     }},
    {"CreateCommittedResource2 buffer with a mip region",
     [](ID3D12Device* device, Made* made) {
         Ref<ID3D12Device8> device8;
         if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device8.p)))) return E_NOINTERFACE;
         const auto heap = default_properties();
         auto desc = as_desc1(buffer(256));
         desc.SamplerFeedbackMipRegion = {4, 4, 1};
         return device8->CreateCommittedResource2(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr, nullptr, IID_PPV_ARGS(&made->resource.p));
     }},
    {"CreateCommittedResource3 buffer with a mip region",
     [](ID3D12Device* device, Made* made) {
         Ref<ID3D12Device10> device10;
         if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device10.p)))) return E_NOINTERFACE;
         const auto heap = default_properties();
         auto desc = as_desc1(buffer(256));
         desc.SamplerFeedbackMipRegion = {2, 2, 1};
         return device10->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_BARRIER_LAYOUT_UNDEFINED,
                                                   nullptr, nullptr, 0, nullptr, IID_PPV_ARGS(&made->resource.p));
     }},
    {"CreateCommittedResource3 MSAA texture",
     [](ID3D12Device* device, Made* made) {
         Ref<ID3D12Device10> device10;
         if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device10.p)))) return E_NOINTERFACE;
         const auto heap = default_properties();
         const auto desc = as_desc1(msaa_target());
         return device10->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_BARRIER_LAYOUT_COMMON,
                                                   nullptr, nullptr, 0, nullptr, IID_PPV_ARGS(&made->resource.p));
     }},
    {"placed buffer",
     [](ID3D12Device* device, Made* made) {
         const HRESULT hr = make_heap(device, 1 << 20, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, &made->heap.p);
         if (FAILED(hr)) return hr;
         const auto desc = buffer(256);
         return device->CreatePlacedResource(made->heap.p, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             IID_PPV_ARGS(&made->resource.p));
     }},
    {"placed 2D texture",
     [](ID3D12Device* device, Made* made) {
         const HRESULT hr =
             make_heap(device, 1 << 20, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES, &made->heap.p);
         if (FAILED(hr)) return hr;
         const auto desc = small_texture();
         return device->CreatePlacedResource(made->heap.p, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             IID_PPV_ARGS(&made->resource.p));
     }},
    {"placed MSAA texture, Alignment 0",
     [](ID3D12Device* device, Made* made) {
         const HRESULT hr = make_heap(device, 8 << 20, D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT,
                                      D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES, &made->heap.p);
         if (FAILED(hr)) return hr;
         const auto desc = msaa_target();
         return device->CreatePlacedResource(made->heap.p, 0, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                             IID_PPV_ARGS(&made->resource.p));
     }},
    {"CreatePlacedResource1 texture",
     [](ID3D12Device* device, Made* made) {
         Ref<ID3D12Device8> device8;
         if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device8.p)))) return E_NOINTERFACE;
         const HRESULT hr =
             make_heap(device, 1 << 20, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES, &made->heap.p);
         if (FAILED(hr)) return hr;
         const auto desc = as_desc1(small_texture());
         return device8->CreatePlacedResource1(made->heap.p, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&made->resource.p));
     }},
    {"CreatePlacedResource1 buffer with a mip region",
     [](ID3D12Device* device, Made* made) {
         Ref<ID3D12Device8> device8;
         if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device8.p)))) return E_NOINTERFACE;
         const HRESULT hr = make_heap(device, 1 << 20, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, &made->heap.p);
         if (FAILED(hr)) return hr;
         auto desc = as_desc1(buffer(256));
         desc.SamplerFeedbackMipRegion = {3, 3, 1};
         return device8->CreatePlacedResource1(made->heap.p, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&made->resource.p));
     }},
    {"reserved buffer",
     [](ID3D12Device* device, Made* made) {
         const auto desc = buffer(1 << 20);
         return device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&made->resource.p));
     }},
    {"reserved 2D texture",
     [](ID3D12Device* device, Made* made) {
         const auto desc = tiled_texture();
         return device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&made->resource.p));
     }},
    {"CreateReservedResource2 texture",
     [](ID3D12Device* device, Made* made) {
         Ref<ID3D12Device10> device10;
         if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device10.p)))) return E_NOINTERFACE;
         const auto desc = tiled_texture();
         return device10->CreateReservedResource2(&desc, D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr, 0, nullptr,
                                                  IID_PPV_ARGS(&made->resource.p));
     }},
};

struct HeapKind {
    const char* name;
    UINT64 bytes, alignment;
    D3D12_HEAP_FLAGS flags;
};

const HeapKind kHeapKinds[] = {
    {"default heap, buffers", 1 << 20, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS},
    {"default heap, render-target textures, 4 MiB aligned", 8 << 20, D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT,
     D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES},
};

// Each object the host creates is created on the layer too and the two answers are compared; the host's are printed
// (they are the measurement behind the layer's behaviour). One the host refuses says nothing and is skipped; one only
// the layer refuses is counted.
// The layer's null out pointer answer: E_POINTER, and no crash (the host crashes there, so it is not compared).
bool null_out_ok(const Prot& p) {
    return p.null_out == E_POINTER && !p.null_crashed;
}

// An object the layer fails to create is a failure, except the placed resources and heaps on a device without buffer
// device address: they live in the GPU address arena, which needs it.
bool refusal_tolerated(bool arena, bool needs_arena) {
    return needs_arena && !arena;
}

Totals compare_newer(ID3D12Device* ours, ID3D12Device* host, const char* host_name, bool probe_null, bool arena) {
    Totals totals;
    for (const Kind& kind : kKinds) {
        Made expected, actual;
        if (FAILED(kind.make(host, &expected)) || !expected.resource.p) {
            ++totals.host_refused;
            std::fprintf(stderr, "[%s] %s: refused by the host\n", host_name, kind.name);
            continue;
        }
        const ResourceAnswers e = answers_of(expected.resource.p, probe_null);
        std::fprintf(stderr, "[%s] %s\n", host_name, kind.name);
        print("host", e);
        const HRESULT hr = kind.make(ours, &actual);
        if (FAILED(hr) || !actual.resource.p) {
            const bool tolerated = refusal_tolerated(arena, std::strstr(kind.name, "laced") != nullptr);
            std::fprintf(stderr, "%s: not created by the layer (0x%08lx)%s\n", kind.name,
                         static_cast<unsigned long>(hr), tolerated ? ", tolerated: needs the GPU address arena" : "");
            ++(tolerated ? totals.ours_refused : totals.mismatched);
            continue;
        }
        const ResourceAnswers a = answers_of(actual.resource.p, probe_null);
        if (same(a, e) && null_out_ok(a.prot)) {
            ++totals.matched;
            continue;
        }
        ++totals.mismatched;
        std::fprintf(stderr, "%s: Resource1/Resource2 answers differ from %s\n", kind.name, host_name);
        print("ours", a);
    }
    for (const HeapKind& kind : kHeapKinds) {
        Ref<ID3D12Heap> expected, actual;
        if (FAILED(make_heap(host, kind.bytes, kind.alignment, kind.flags, &expected.p)) || !expected.p) {
            ++totals.host_refused;
            std::fprintf(stderr, "[%s] heap, %s: refused by the host\n", host_name, kind.name);
            continue;
        }
        const HeapAnswers e = answers_of(expected.p, probe_null);
        std::fprintf(stderr, "[%s] heap, %s\n", host_name, kind.name);
        print("host", e);
        const HRESULT hr = make_heap(ours, kind.bytes, kind.alignment, kind.flags, &actual.p);
        if (FAILED(hr) || !actual.p) {
            const bool tolerated = refusal_tolerated(arena, true);
            std::fprintf(stderr, "heap, %s: not created by the layer (0x%08lx)%s\n", kind.name,
                         static_cast<unsigned long>(hr), tolerated ? ", tolerated: needs the GPU address arena" : "");
            ++(tolerated ? totals.ours_refused : totals.mismatched);
            continue;
        }
        const HeapAnswers a = answers_of(actual.p, probe_null);
        if (same(a, e) && null_out_ok(a.prot)) {
            ++totals.matched;
            continue;
        }
        ++totals.mismatched;
        std::fprintf(stderr, "heap, %s: Heap1 answers differ from %s\n", kind.name, host_name);
        print("ours", a);
    }
    return totals;
}
}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;  // heaps (and so placed resources) live in the GPU address arena
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    unsigned matched = 0, mismatched = 0, references = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = vk.buffer_device_address;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;

        constexpr auto t1 = D3D12_RESOURCE_DIMENSION_TEXTURE1D, t2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                       t3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        constexpr auto rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
        constexpr auto none = D3D12_RESOURCE_FLAG_NONE, target = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       depth = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        constexpr auto common = D3D12_RESOURCE_STATE_COMMON, render = D3D12_RESOURCE_STATE_RENDER_TARGET;
        constexpr auto default_heap = D3D12_HEAP_TYPE_DEFAULT;
        constexpr UINT64 k4 = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT, k64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                         m4 = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;
        const D3D12_RESOURCE_DESC tiled = tiled_texture();
        const std::vector<Case> cases = {
            // Buffers: Alignment 0 reads back as 64 KiB on every heap type.
            {"buffer, default heap", buffer(100), default_heap, common, false},
            {"buffer, upload heap", buffer(100), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false},
            {"buffer, readback heap", buffer(100), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false},
            {"buffer, 64 KiB asked", buffer(100, k64), default_heap, common, false},
            {"buffer, unordered access", buffer(70000, 0, uav), default_heap, common, false},
            // Single-sampled textures: 64 KiB unless the description asks for 4 KiB; MipLevels 0 is the full chain.
            {"2D 64x64", texture(t2, 64, 64, 1, 1, rgba8), default_heap, common, false},
            {"2D 64x64, all mips", texture(t2, 64, 64, 1, 0, rgba8), default_heap, common, false},
            {"2D 100x60, all mips", texture(t2, 100, 60, 1, 0, rgba8), default_heap, common, false},
            {"2D 64x64, 4 KiB asked", texture(t2, 64, 64, 1, 1, rgba8, 1, none, k4), default_heap, common, false},
            {"2D 64x64, 64 KiB asked", texture(t2, 64, 64, 1, 1, rgba8, 1, none, k64), default_heap, common, false},
            {"2D 1024x1024, all mips", texture(t2, 1024, 1024, 1, 0, rgba8), default_heap, common, false},
            {"2D array of 6, all mips", texture(t2, 256, 256, 6, 0, rgba8), default_heap, common, false},
            {"2D typeless", texture(t2, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_TYPELESS), default_heap, common, false},
            {"2D render target", texture(t2, 256, 256, 1, 1, rgba8, 1, target), default_heap, render, false},
            {"2D render target, all mips", texture(t2, 256, 256, 1, 0, rgba8, 1, target), default_heap, render, false},
            {"2D depth", texture(t2, 256, 256, 1, 1, DXGI_FORMAT_D32_FLOAT, 1, depth), default_heap,
             D3D12_RESOURCE_STATE_DEPTH_WRITE, false},
            {"2D unordered access", texture(t2, 256, 256, 1, 1, rgba8, 1, uav), default_heap, common, false},
            {"3D 32x32x32, all mips", texture(t3, 32, 32, 32, 0, rgba8), default_heap, common, false},
            {"3D 64x16x4, all mips", texture(t3, 64, 16, 4, 0, rgba8), default_heap, common, false},
            {"1D 128, all mips", texture(t1, 128, 1, 1, 0, rgba8), default_heap, common, false},
            // Multisampled textures: 4 MiB unless the description asks for 64 KiB.
            {"2D 4x render target", texture(t2, 256, 256, 1, 1, rgba8, 4, target), default_heap, render, false},
            {"2D 4x render target, 64 KiB asked", texture(t2, 64, 64, 1, 1, rgba8, 4, target, k64), default_heap,
             render, false},
            {"2D 4x render target, 4 MiB asked", texture(t2, 64, 64, 1, 1, rgba8, 4, target, m4), default_heap, render,
             false},
            {"2D 4x depth", texture(t2, 256, 256, 1, 1, DXGI_FORMAT_D32_FLOAT, 4, depth), default_heap,
             D3D12_RESOURCE_STATE_DEPTH_WRITE, false},
            // Reserved resources report their own layout and the same alignment.
            {"reserved 2D 1024x1024, all mips", tiled, default_heap, common, true},
            {"reserved buffer", buffer(1 << 20), default_heap, common, true},
        };

        if (hostwine::host_is_wine()) return 77;  // every case is a comparison with the host
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            std::fprintf(stderr, "no DXGI factory\n");
            return 1;
        }
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string adapter_name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            const char* name = adapter_name.c_str();
            const Totals totals = compare(ours.p, host.p, name, cases);
            std::printf("resource desc against %s: %u of %zu descriptions match, %u differ, %u refused by the host, "
                        "%u not created by the layer\n",
                        name, totals.matched, cases.size(), totals.mismatched, totals.host_refused, totals.ours_refused);
            matched += totals.matched;
            mismatched += totals.mismatched;
            const Totals newer = compare_newer(ours.p, host.p, name, true, vk.buffer_device_address);
            std::printf("resource1/resource2/heap1 against %s: %u of %zu objects match, %u differ, %u refused by the "
                        "host, %u not created by the layer\n",
                        name, newer.matched, std::size(kKinds) + std::size(kHeapKinds), newer.mismatched,
                        newer.host_refused, newer.ours_refused);
            matched += newer.matched;
            mismatched += newer.mismatched;
            ++references;
        }
    }
    if (!references) {
        std::fprintf(stderr, "no D3D12 reference device (hardware adapter or WARP)\n");
        return 1;
    }
    if (mismatched || !matched) return 1;
    if (vk.validation_errors()) return 1;
    return 0;
}

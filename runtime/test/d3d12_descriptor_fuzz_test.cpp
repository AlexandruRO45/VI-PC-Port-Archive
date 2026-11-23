// Differential fuzzer for the descriptor paths of the D3D12 layer, compared with the host PC runtime (or WARP).
// Each case is a random program built from (--seed, case number) alone, run twice: on a host D3D12 device with an
// HLSL compute shader (SM 5.1 resource arrays), and on the layer through the Xbox-ABI proxy with
// test/glsl/cs_descriptor_fuzz.comp, written to the translator's runtime contract. A case has:
//   - resources: three RGBA8/BGRA8 textures (sizes 4 to 16, 1 to 5 mips, one of them an array of 2 to 4 layers),
//     an R32_UINT texture with unordered access, a buffer for shader-resource views, one for unordered-access views
//     and one for constant-buffer views, all filled with bytes that depend on the case only;
//   - CPU-only and shader-visible CBV/SRV/UAV and sampler heaps of random sizes;
//   - rounds. A round first writes descriptors on the CPU, in random order: CreateShaderResourceView (2D textures
//     with mip ranges, min-LOD clamps and component mappings, 2D arrays with layer ranges, typed, raw and structured
//     buffers, null descriptors), CreateUnorderedAccessView (typed and raw buffers, a texture mip, null),
//     CreateConstantBufferView (offsets, sizes, null), CreateSampler (address modes, integer LOD bias and clamps,
//     border colours, anisotropy and comparison functions the filter ignores), CopyDescriptorsSimple and
//     CopyDescriptors with several ranges of different sizes on each side (and the null size array). Copies read
//     CPU-only heaps and write any heap; later writes overwrite earlier ones, null over live and live over null.
//     Then one command list dispatches up to 16 probes, switching between the shader-visible heaps with
//     SetDescriptorHeaps and moving SetComputeRootDescriptorTable bases and the root constants between dispatches.
//     A probe reads one slot of the bound view table the way its contents allow (sample, load, buffer loads, CBV
//     rows, a UAV texture load), 64 invocations of 16 bytes into a root UAV.
// Every output byte of every round is compared. Sampling uses point filters, integer LODs and texel centres from
// two widths left of the image to three right, so every address mode and the border are reached and both runtimes
// must give the same bits; linear and anisotropic filtering are not bit-exact between implementations and are not
// generated. What a slot holds is tracked while generating, so a probe only reads a slot as the type it has: no
// program depends on undefined D3D12 behaviour (--d3d12-debug runs the host with the debug layer to check that).
// Where NVIDIA's driver and WARP give different answers the generator stays out: a border channel on a half step of
// 8 bits, a component mapping on reads that return zeros (out of range, below the min-LOD clamp) or on the border,
// and a min-LOD clamp past the view's last mip. The layer agrees with NVIDIA's driver in each of them.
//
//   d3d12_descriptor_fuzz_test [--seed N] [--cases N] [--only K] [--warp] [--d3d12-debug] [--verbose]
//                              [--null-border] [--cbv-upload]
//
// --only K runs case K of the seed alone (what a failure prints to reproduce it); --warp compares with WARP instead
// of the hardware adapter (LEONIDA_HOST_ADAPTER, as in host_adapter.h, otherwise); --verbose prints each program.
// One known difference is left out unless --null-border asks for it: D3D12 reads 0 from a null SRV whatever the
// sampler, while the layer's null T# (all zeros) misses the hash table and the shader samples the fallback image with
// the S#, so a BORDER sampler outside the image returns its border colour. That is the translator's lookup-miss rule
// (a miss is index 0, docs/d3d12_vulkan.md), which the descriptor code cannot change.
// The constant buffer is a DEFAULT-heap buffer unless --cbv-upload puts it in an UPLOAD heap, as applications often
// do. An upload buffer whose memory lies below the GPU address arena has a VA that does not fit a V#'s 48-bit base,
// so its CBVs read the wrong address and the device is lost (D3D-516, fixed by #5052); --cbv-upload finds that.
// ctest runs a fixed seed with a few cases. No game data. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation (0 errors required).
#include "host_adapter.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_CS_DESCRIPTOR_FUZZ_SPV
#error "LEONIDA_TEST_CS_DESCRIPTOR_FUZZ_SPV must name the compiled test shader"
#endif

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

// ---- the program ----

constexpr UINT kTableSpan = 16;       // descriptors in each table range of the root signature
constexpr UINT kMaxProbes = 16;       // dispatches in one round
constexpr UINT kProbeBytes = 64 * 16;  // 64 invocations of one uint4
constexpr UINT kBufferBytes = 8192;   // each of the three buffers
constexpr UINT kTextures = 4;         // three sampled, the last one R32_UINT with unordered access
constexpr int kNull = -1, kSrvBuffer = 100, kUavBuffer = 101;

// What a probe does with its slot. The numbers are the shaders'.
enum Kind : std::uint32_t {
    sample_2d, load_2d, sample_2d_array, typed_srv, raw_srv, struct4_srv, struct16_srv, cbv, typed_uav, raw_uav,
    tex_uav,
};
const char* const kind_names[] = {"sample_2d", "load_2d", "sample_2d_array", "typed_srv", "raw_srv", "struct4_srv",
                                  "struct16_srv", "cbv", "typed_uav", "raw_uav", "tex_uav"};

// What a heap slot holds, as far as the generator knows.
enum class Holds : std::uint8_t {
    nothing, tex2d, tex2d_array, typed_srv, raw_srv, struct4_srv, struct16_srv, cbv, typed_uav, raw_uav, tex_uav,
    sampler,
};

struct Contents {
    Holds holds = Holds::nothing;
    UINT mips = 1;    // the view's mips (texture SRVs)
    UINT clamped = 0;  // the view's first mip its ResourceMinLODClamp leaves readable (2D texture SRVs)
    UINT layers = 1;  // the view's layers (2D array SRVs)
    UINT rows = 0;    // 16-byte rows (CBVs)
    int op = -1;      // the operation that wrote it, for the report: "round.op"
    int round = -1;
    bool null = false;    // a null view
    bool mapped = false;  // a texture view whose component mapping is not the default
    bool border = false;  // a sampler with a BORDER address mode
};

struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(seed) {}
    std::uint64_t next() {  // splitmix64
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    UINT below(UINT n) { return n ? UINT(next() % n) : 0; }
    UINT range(UINT low, UINT high) { return low + below(high - low + 1); }  // inclusive
    bool chance(UINT percent) { return below(100) < percent; }
};

struct TextureSpec {
    DXGI_FORMAT format;
    UINT width, height;
    UINT16 mips, layers;
    bool uav;
};

struct HeapSpec {
    D3D12_DESCRIPTOR_HEAP_TYPE type;
    bool visible;
    UINT size;
};

struct Place {
    UINT heap, slot;
};

struct Op {
    enum Type { create_srv, create_uav, create_cbv, create_sampler, copy_simple, copy } type{};
    Place at{};  // where a create writes, or the first destination of a simple copy
    int resource = kNull;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    UINT cbv_offset = 0, cbv_size = 0;  // CBV: offset into the constant buffer (size 0: a null description)
    D3D12_SAMPLER_DESC sampler{};
    Place from{};  // simple copy source
    UINT count = 0;
    std::vector<Place> destinations, sources;  // copy ranges
    std::vector<UINT> destination_sizes, source_sizes;
    bool null_sizes = false;  // every range is one descriptor: pass null size arrays
    D3D12_DESCRIPTOR_HEAP_TYPE copy_type{};
};

struct Probe {
    Kind kind{};
    UINT view_heap = 0, view_base = 0, view_index = 0;
    UINT sampler_heap = 0, sampler_base = 0, sampler_index = 0;
    UINT level = 0, layer = 0;
};

struct Round {
    std::vector<Op> ops;
    std::vector<Probe> probes;
};

struct Program {
    std::uint64_t seed = 0;
    std::array<TextureSpec, kTextures> textures{};
    std::vector<HeapSpec> heaps;
    std::vector<Round> rounds;
};

bool sampled_kind(Kind kind) { return kind == sample_2d || kind == sample_2d_array; }

UINT element_bytes(DXGI_FORMAT format) {
    return format == DXGI_FORMAT_R32G32B32A32_UINT ? 16 : format == DXGI_FORMAT_R32G32_UINT ? 8 : 4;
}

UINT random_mapping(Rng& rng) {
    if (rng.chance(50)) return D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    return D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(rng.below(6), rng.below(6), rng.below(6), rng.below(6));
}

// The generator. It keeps what every slot holds so that probes and copies stay within defined behaviour.
class Generator {
public:
    // min_lod: ResourceMinLODClamp can be generated (imageViewMinLod); mirror_once: MIRROR_ONCE can (the layer clamps
    // without samplerMirrorClampToEdge, by design, and logs it).
    // null_border: a null texture SRV may be sampled with a BORDER sampler (--null-border).
    Generator(std::uint64_t seed, bool min_lod, bool mirror_once, bool null_border)
        : rng_(seed), min_lod_(min_lod), mirror_once_(mirror_once), null_border_(null_border) {
        program_.seed = seed;
    }

    Program build() {
        textures();
        heaps();
        contents_.resize(program_.heaps.size());
        for (UINT h = 0; h < program_.heaps.size(); ++h) contents_[h].resize(program_.heaps[h].size);
        const UINT rounds = rng_.range(2, 4);
        for (UINT r = 0; r < rounds; ++r) round(r);
        return std::move(program_);
    }

private:
    Rng rng_;
    bool min_lod_, mirror_once_, null_border_;
    Program program_;
    std::vector<std::vector<Contents>> contents_;

    void textures() {
        const UINT sizes[] = {4, 8, 16};
        for (UINT t = 0; t < kTextures; ++t) {
            TextureSpec& spec = program_.textures[t];
            spec.uav = t == kTextures - 1;
            spec.format = spec.uav ? DXGI_FORMAT_R32_UINT
                          : rng_.chance(50) ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM;
            spec.width = sizes[rng_.below(3)];
            spec.height = sizes[rng_.below(3)];
            UINT full = 1;
            while ((std::max(spec.width, spec.height) >> full) > 0) ++full;
            spec.mips = UINT16(rng_.range(1, spec.uav ? std::min(full, 3u) : full));
            // Texture 0 is always one layer (a TEXTURE2D view needs that), texture 2 always an array.
            spec.layers = UINT16(t == 2 ? rng_.range(2, 4) : t == 1 && rng_.chance(50) ? rng_.range(2, 3) : 1);
        }
    }

    void heaps() {
        const auto add = [&](D3D12_DESCRIPTOR_HEAP_TYPE type, bool visible, UINT low, UINT high) {
            program_.heaps.push_back({type, visible, rng_.range(low, high)});
        };
        // Shader-visible heaps hold at least one table span; the first of each type is the one bound first.
        add(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, kTableSpan, 64);
        add(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true, kTableSpan, 32);
        add(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, false, 4, 40);
        add(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, false, 4, 24);
        if (rng_.chance(60)) add(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, kTableSpan, 48);
        if (rng_.chance(40)) add(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true, kTableSpan, 32);
        if (rng_.chance(60)) add(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, false, 4, 32);
        if (rng_.chance(40)) add(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, false, 4, 16);
    }

    std::vector<UINT> heaps_of(D3D12_DESCRIPTOR_HEAP_TYPE type, int visible) const {  // visible: 0, 1, -1 either
        std::vector<UINT> out;
        for (UINT h = 0; h < program_.heaps.size(); ++h)
            if (program_.heaps[h].type == type && (visible < 0 || int(program_.heaps[h].visible) == visible))
                out.push_back(h);
        return out;
    }

    UINT pick(const std::vector<UINT>& from) { return from[rng_.below(UINT(from.size()))]; }

    // A place to write a view: any heap of the type, biased to the shader-visible ones' first table spans.
    Place target(D3D12_DESCRIPTOR_HEAP_TYPE type) {
        const UINT heap = rng_.chance(70) ? pick(heaps_of(type, 1)) : pick(heaps_of(type, -1));
        const UINT size = program_.heaps[heap].size;
        const UINT slot = rng_.chance(60) ? rng_.below(std::min(size, kTableSpan + 8)) : rng_.below(size);
        return {heap, slot};
    }

    void round(UINT r) {
        Round round;
        const UINT ops = rng_.range(8, 36);
        for (UINT i = 0; i < ops; ++i) {
            Op op;
            const UINT what = rng_.below(100);
            bool ok;
            if (what < 22) ok = make_texture_srv(op);
            else if (what < 40) ok = make_buffer_srv(op);
            else if (what < 50) ok = make_uav(op);
            else if (what < 58) ok = make_cbv(op);
            else if (what < 72) ok = make_sampler(op);
            else if (what < 86) ok = make_copy_simple(op);
            else ok = make_copy(op);
            if (!ok) continue;
            apply(op, r, int(round.ops.size()));
            round.ops.push_back(std::move(op));
        }
        const UINT probes = rng_.range(4, kMaxProbes);
        for (UINT i = 0; i < probes * 4 && round.probes.size() < probes; ++i) {
            Probe probe;
            if (make_probe(probe)) round.probes.push_back(probe);
        }
        program_.rounds.push_back(std::move(round));
    }

    void record(Place at, Contents c, UINT r, int op) {
        c.round = int(r);
        c.op = op;
        contents_[at.heap][at.slot] = c;
    }

    // What a create or a copy does to the generator's view of the heaps, in the order D3D12 applies it.
    void apply(const Op& op, UINT r, int index) {
        if (op.type == Op::copy_simple) {
            for (UINT k = 0; k < op.count; ++k)
                contents_[op.at.heap][op.at.slot + k] = contents_[op.from.heap][op.from.slot + k];
            return;
        }
        if (op.type == Op::copy) {
            std::vector<Contents> moved;
            for (size_t s = 0; s < op.sources.size(); ++s)
                for (UINT k = 0; k < op.source_sizes[s]; ++k)
                    moved.push_back(contents_[op.sources[s].heap][op.sources[s].slot + k]);
            size_t at = 0;
            for (size_t d = 0; d < op.destinations.size(); ++d)
                for (UINT k = 0; k < op.destination_sizes[d]; ++k)
                    contents_[op.destinations[d].heap][op.destinations[d].slot + k] = moved[at++];
            return;
        }
        Contents c;
        if (op.type == Op::create_sampler) {
            c.holds = Holds::sampler;
            const auto& s = op.sampler;
            c.border = s.AddressU == D3D12_TEXTURE_ADDRESS_MODE_BORDER || s.AddressV == D3D12_TEXTURE_ADDRESS_MODE_BORDER ||
                       s.AddressW == D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        } else if (op.type == Op::create_cbv) {
            c.holds = Holds::cbv;
            c.rows = op.cbv_size / 16;
        } else if (op.type == Op::create_srv) {
            const auto& d = op.srv;
            if (d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D) {
                c.holds = Holds::tex2d;
                c.mips = op.resource == kNull ? 1 : d.Texture2D.MipLevels == UINT(-1)
                    ? program_.textures[op.resource].mips - d.Texture2D.MostDetailedMip : d.Texture2D.MipLevels;
                const UINT clamp = UINT(d.Texture2D.ResourceMinLODClamp);  // a whole resource mip
                c.clamped = clamp > d.Texture2D.MostDetailedMip ? clamp - d.Texture2D.MostDetailedMip : 0;
            } else if (d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY) {
                c.holds = Holds::tex2d_array;
                c.mips = op.resource == kNull ? 1 : d.Texture2DArray.MipLevels == UINT(-1)
                    ? program_.textures[op.resource].mips - d.Texture2DArray.MostDetailedMip
                    : d.Texture2DArray.MipLevels;
                c.layers = op.resource == kNull ? 1 : d.Texture2DArray.ArraySize;
            } else if (d.Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW) {
                c.holds = Holds::raw_srv;
            } else if (d.Buffer.StructureByteStride == 4) {
                c.holds = Holds::struct4_srv;
            } else if (d.Buffer.StructureByteStride == 16) {
                c.holds = Holds::struct16_srv;
            } else {
                c.holds = Holds::typed_srv;
            }
        } else {
            const auto& d = op.uav;
            c.holds = d.ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2D ? Holds::tex_uav
                      : d.Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW ? Holds::raw_uav : Holds::typed_uav;
        }
        c.null = (op.type == Op::create_srv || op.type == Op::create_uav) && op.resource == kNull;
        c.mapped = (c.holds == Holds::tex2d || c.holds == Holds::tex2d_array) &&
                   op.srv.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        record(op.at, c, r, index);
    }

    bool make_texture_srv(Op& op) {
        op.type = Op::create_srv;
        op.at = target(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const bool null = rng_.chance(12);
        const bool array = rng_.chance(35);
        UINT t = array ? rng_.below(3) : (rng_.chance(50) ? 0 : rng_.below(3));
        if (!array && program_.textures[t].layers > 1) t = 0;
        const TextureSpec& spec = program_.textures[t];
        op.resource = null ? kNull : int(t);
        auto& d = op.srv;
        d.Format = null ? DXGI_FORMAT_R8G8B8A8_UNORM : spec.format;
        d.Shader4ComponentMapping = random_mapping(rng_);
        const UINT first = null ? 0 : rng_.below(spec.mips);
        const UINT mips = null ? UINT(-1) : rng_.chance(40) ? UINT(-1) : rng_.range(1, spec.mips - first);
        // A clamp past the view's last mip is left out: NVIDIA's driver then reads 0 and WARP the last mip.
        const UINT last = null ? 0 : first + (mips == UINT(-1) ? spec.mips - first : mips) - 1;
        const float clamp = !null && min_lod_ && rng_.chance(25) ? float(rng_.range(0, last)) : 0.0f;
        if (array) {
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            const UINT first_layer = null ? 0 : rng_.below(spec.layers);
            d.Texture2DArray = {first, mips, first_layer, null ? 1 : rng_.range(1, spec.layers - first_layer), 0,
                                clamp};
        } else {
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            d.Texture2D = {first, mips, 0, clamp};
        }
        return true;
    }

    bool make_buffer_srv(Op& op) {
        op.type = Op::create_srv;
        op.at = target(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        op.resource = rng_.chance(12) ? kNull : kSrvBuffer;
        auto& d = op.srv;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        const UINT form = rng_.below(4);
        UINT element = 4;
        if (form == 0) {
            const DXGI_FORMAT formats[] = {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_UINT,
                                           DXGI_FORMAT_R32G32B32A32_UINT};
            d.Format = formats[rng_.below(3)];
            d.Shader4ComponentMapping = random_mapping(rng_);
            element = element_bytes(d.Format);
        } else if (form == 1) {
            d.Format = DXGI_FORMAT_R32_TYPELESS;
            d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        } else {
            d.Format = DXGI_FORMAT_UNKNOWN;
            element = form == 2 ? 4 : 16;
            d.Buffer.StructureByteStride = element;
        }
        const UINT elements = kBufferBytes / element;
        // A raw view starts on 16 bytes; the others anywhere.
        const UINT step = form == 1 ? 4 : 1;
        d.Buffer.FirstElement = UINT64(rng_.below((elements - 1) / step) * step);
        d.Buffer.NumElements = rng_.range(1, std::min<UINT>(80, elements - UINT(d.Buffer.FirstElement)));
        // Out-of-range reads through a component mapping that is not the default are left out: NVIDIA's driver
        // applies the mapping to the zeros, WARP returns the format's defaults. Such a view covers all 64 reads.
        if (d.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING) {
            d.Buffer.FirstElement = rng_.below(elements - 64 + 1);
            d.Buffer.NumElements = rng_.range(64, std::min<UINT>(80, elements - UINT(d.Buffer.FirstElement)));
        }
        return true;
    }

    bool make_uav(Op& op) {
        op.type = Op::create_uav;
        op.at = target(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const bool null = rng_.chance(12);
        auto& d = op.uav;
        const UINT form = rng_.below(3);
        if (form == 2) {
            const TextureSpec& spec = program_.textures[kTextures - 1];
            op.resource = null ? kNull : int(kTextures - 1);
            d.Format = DXGI_FORMAT_R32_UINT;
            d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            d.Texture2D.MipSlice = null ? 0 : rng_.below(spec.mips);
            return true;
        }
        op.resource = null ? kNull : kUavBuffer;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        if (form == 0) {
            d.Format = DXGI_FORMAT_R32_UINT;
            d.Buffer.FirstElement = rng_.below(kBufferBytes / 4 - 1);
        } else {
            d.Format = DXGI_FORMAT_R32_TYPELESS;
            d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            d.Buffer.FirstElement = UINT64(rng_.below(kBufferBytes / 16 - 1) * 4);
        }
        d.Buffer.NumElements = rng_.range(1, std::min<UINT>(80, kBufferBytes / 4 - UINT(d.Buffer.FirstElement)));
        return true;
    }

    bool make_cbv(Op& op) {
        op.type = Op::create_cbv;
        op.at = target(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (rng_.chance(12)) return true;  // null: no buffer
        op.cbv_size = 256 * rng_.range(1, 4);
        op.cbv_offset = 256 * rng_.below((kBufferBytes - op.cbv_size) / 256 + 1);
        return true;
    }

    bool make_sampler(Op& op) {
        op.type = Op::create_sampler;
        op.at = target(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        auto& s = op.sampler;
        s.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        s.AddressU = D3D12_TEXTURE_ADDRESS_MODE(rng_.range(1, mirror_once_ ? 5 : 4));
        s.AddressV = D3D12_TEXTURE_ADDRESS_MODE(rng_.range(1, mirror_once_ ? 5 : 4));
        s.AddressW = D3D12_TEXTURE_ADDRESS_MODE(rng_.range(1, mirror_once_ ? 5 : 4));
        s.MipLODBias = rng_.chance(50) ? 0.0f : float(int(rng_.range(0, 4)) - 2);
        s.MaxAnisotropy = rng_.chance(70) ? 1 : rng_.range(1, 16);  // ignored by a point filter
        s.ComparisonFunc = D3D12_COMPARISON_FUNC(rng_.range(1, 8));  // ignored by a non-comparison filter
        // No channel lands on a half step of the 8-bit textures (0.5 is 127.5/255): NVIDIA's driver rounds that tie
        // down and WARP up.
        static const float borders[][4] = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}, {0.25f, 0.6f, 0.75f, 1},
                                           {1, 0, 0.4f, 0.125f}};
        std::memcpy(s.BorderColor, borders[rng_.below(5)], sizeof(s.BorderColor));
        s.MinLOD = rng_.chance(70) ? 0.0f : float(rng_.range(0, 3));
        s.MaxLOD = rng_.chance(60) ? D3D12_FLOAT32_MAX : s.MinLOD + float(rng_.range(0, 3));
        return true;
    }

    // A source range in a CPU-only heap of the type, and a destination in a different heap.
    bool make_copy_simple(Op& op) {
        op.type = Op::copy_simple;
        const auto type = rng_.chance(70) ? D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV : D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        op.copy_type = type;
        const UINT src = pick(heaps_of(type, 0));
        std::vector<UINT> others;
        for (UINT h : heaps_of(type, -1))
            if (h != src) others.push_back(h);
        if (others.empty()) return false;
        const UINT dst = pick(others);
        op.count = rng_.range(1, std::min({8u, program_.heaps[src].size, program_.heaps[dst].size}));
        op.from = {src, rng_.below(program_.heaps[src].size - op.count + 1)};
        const UINT dst_size = program_.heaps[dst].size;
        const UINT close_end = std::min(dst_size, kTableSpan + 8);
        op.at = {dst, rng_.chance(60) && close_end >= op.count ? rng_.below(close_end - op.count + 1)
                                                          : rng_.below(dst_size - op.count + 1)};
        return true;
    }

    // Several source ranges (CPU-only heaps) and several destination ranges (any other heap), the same total. Ranges
    // on one side never overlap each other, and a destination heap is never a source heap.
    bool make_copy(Op& op) {
        op.type = Op::copy;
        const auto type = rng_.chance(70) ? D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV : D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        op.copy_type = type;
        const auto sources = heaps_of(type, 0);
        std::vector<UINT> destinations;
        const UINT dst_choice = pick(heaps_of(type, -1));
        for (UINT h : heaps_of(type, -1))
            if (program_.heaps[h].visible || h == dst_choice) destinations.push_back(h);
        std::erase_if(destinations, [&](UINT h) {
            return std::find(sources.begin(), sources.end(), h) != sources.end();
        });
        if (destinations.empty()) return false;
        op.null_sizes = rng_.chance(20);
        const UINT total = op.null_sizes ? rng_.range(1, 4) : rng_.range(1, 12);
        // Split `total` into ranges placed without overlap in the given heaps.
        const auto split = [&](const std::vector<UINT>& heaps, std::vector<Place>& places, std::vector<UINT>& sizes) {
            std::vector<std::vector<bool>> used(program_.heaps.size());
            for (UINT h : heaps) used[h].assign(program_.heaps[h].size, false);
            UINT left = total;
            while (left) {
                const UINT n = op.null_sizes ? 1 : rng_.range(1, std::min(left, 5u));
                bool placed = false;
                for (UINT attempt = 0; attempt < 16 && !placed; ++attempt) {
                    const UINT h = pick(heaps);
                    if (program_.heaps[h].size < n) continue;
                    const UINT close_end = std::min(program_.heaps[h].size, kTableSpan + 8);
                    const UINT at = program_.heaps[h].visible && close_end >= n && rng_.chance(70)
                                        ? rng_.below(close_end - n + 1) : rng_.below(program_.heaps[h].size - n + 1);
                    if (std::any_of(used[h].begin() + at, used[h].begin() + at + n, [](bool b) { return b; }))
                        continue;
                    std::fill(used[h].begin() + at, used[h].begin() + at + n, true);
                    places.push_back({h, at});
                    sizes.push_back(n);
                    placed = true;
                }
                if (!placed) return false;
                left -= n;
            }
            return true;
        };
        return split(sources, op.sources, op.source_sizes) && split(destinations, op.destinations,
                                                                    op.destination_sizes);
    }

    bool make_probe(Probe& probe) {
        const auto view_heaps = heaps_of(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
        probe.view_heap = pick(view_heaps);
        const UINT size = program_.heaps[probe.view_heap].size;
        probe.view_base = rng_.chance(50) ? 0 : rng_.below(size - kTableSpan + 1);
        std::vector<UINT> live;
        for (UINT k = 0; k < kTableSpan; ++k)
            if (contents_[probe.view_heap][probe.view_base + k].holds != Holds::nothing) live.push_back(k);
        if (live.empty()) return false;
        probe.view_index = pick(live);
        const Contents& c = contents_[probe.view_heap][probe.view_base + probe.view_index];
        switch (c.holds) {
        case Holds::tex2d:
            probe.kind = rng_.chance(60) ? sample_2d : load_2d;
            // A load below the min-LOD clamp reads zeros, which NVIDIA's driver maps through the view's component
            // mapping and WARP does not: loads stay at or above it.
            probe.level = probe.kind == sample_2d ? rng_.below(c.mips + 1) : rng_.range(c.clamped, c.mips - 1);
            break;
        case Holds::tex2d_array:
            probe.kind = sample_2d_array;
            probe.level = rng_.below(c.mips + 1);
            probe.layer = rng_.below(c.layers + 1);
            break;
        case Holds::typed_srv: probe.kind = typed_srv; break;
        case Holds::raw_srv: probe.kind = raw_srv; break;
        case Holds::struct4_srv: probe.kind = struct4_srv; break;
        case Holds::struct16_srv: probe.kind = struct16_srv; break;
        case Holds::cbv:
            probe.kind = cbv;
            probe.level = c.rows ? rng_.range(1, std::min(64u, c.rows)) : rng_.range(1, 64);
            break;
        case Holds::typed_uav: probe.kind = typed_uav; break;
        case Holds::raw_uav: probe.kind = raw_uav; break;
        case Holds::tex_uav: probe.kind = tex_uav; break;
        default: return false;
        }
        if (!sampled_kind(probe.kind)) {
            probe.sampler_heap = heaps_of(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1)[0];
            return true;
        }
        const auto sampler_heaps = heaps_of(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1);
        for (UINT attempt = 0; attempt < 8; ++attempt) {
            probe.sampler_heap = pick(sampler_heaps);
            const UINT samplers = program_.heaps[probe.sampler_heap].size;
            probe.sampler_base = rng_.chance(50) ? 0 : rng_.below(samplers - kTableSpan + 1);
            std::vector<UINT> ready;
            for (UINT k = 0; k < kTableSpan; ++k)
                if (const Contents& s = contents_[probe.sampler_heap][probe.sampler_base + k];
                    s.holds == Holds::sampler && (null_border_ || !(c.null && s.border)) &&
                    !(c.mapped && s.border))  // whether a view's mapping applies to the border is the driver's
                                              // choice (VK_EXT_border_color_swizzle): NVIDIA and WARP differ
                    ready.push_back(k);
            if (ready.empty()) continue;
            probe.sampler_index = pick(ready);
            return true;
        }
        return false;
    }
};

// ---- printing ----

const char* heap_type_name(D3D12_DESCRIPTOR_HEAP_TYPE type) {
    return type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ? "sampler" : "view";
}

std::string describe(const Op& op) {
    char text[512];
    const auto place = [](Place p) { return "h" + std::to_string(p.heap) + "[" + std::to_string(p.slot) + "]"; };
    const std::string res = op.resource == kNull ? "null" : op.resource == kSrvBuffer ? "srvbuf"
                            : op.resource == kUavBuffer ? "uavbuf" : "tex" + std::to_string(op.resource);
    switch (op.type) {
    case Op::create_srv: {
        const auto& d = op.srv;
        if (d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D)
            std::snprintf(text, sizeof text, "SRV %s TEXTURE2D %s fmt %d mips %u+%d clamp %g map 0x%x",
                          place(op.at).c_str(), res.c_str(), int(d.Format), d.Texture2D.MostDetailedMip,
                          int(d.Texture2D.MipLevels), d.Texture2D.ResourceMinLODClamp, d.Shader4ComponentMapping);
        else if (d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY)
            std::snprintf(text, sizeof text,
                          "SRV %s TEXTURE2DARRAY %s fmt %d mips %u+%d layers %u+%u clamp %g map 0x%x",
                          place(op.at).c_str(), res.c_str(), int(d.Format), d.Texture2DArray.MostDetailedMip,
                          int(d.Texture2DArray.MipLevels), d.Texture2DArray.FirstArraySlice,
                          d.Texture2DArray.ArraySize, d.Texture2DArray.ResourceMinLODClamp,
                          d.Shader4ComponentMapping);
        else
            std::snprintf(text, sizeof text, "SRV %s BUFFER %s fmt %d first %llu num %u stride %u flags %u map 0x%x",
                          place(op.at).c_str(), res.c_str(), int(d.Format),
                          static_cast<unsigned long long>(d.Buffer.FirstElement), d.Buffer.NumElements,
                          d.Buffer.StructureByteStride, unsigned(d.Buffer.Flags), d.Shader4ComponentMapping);
        break;
    }
    case Op::create_uav:
        if (op.uav.ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2D)
            std::snprintf(text, sizeof text, "UAV %s TEXTURE2D %s mip %u", place(op.at).c_str(), res.c_str(),
                          op.uav.Texture2D.MipSlice);
        else
            std::snprintf(text, sizeof text, "UAV %s BUFFER %s fmt %d first %llu num %u flags %u",
                          place(op.at).c_str(), res.c_str(), int(op.uav.Format),
                          static_cast<unsigned long long>(op.uav.Buffer.FirstElement), op.uav.Buffer.NumElements,
                          unsigned(op.uav.Buffer.Flags));
        break;
    case Op::create_cbv:
        std::snprintf(text, sizeof text, "CBV %s offset %u size %u%s", place(op.at).c_str(), op.cbv_offset,
                      op.cbv_size, op.cbv_size ? "" : " (null)");
        break;
    case Op::create_sampler: {
        const auto& s = op.sampler;
        std::snprintf(text, sizeof text,
                      "Sampler %s addr %d/%d/%d bias %g aniso %u cmp %d border %g,%g,%g,%g lod [%g, %g]",
                      place(op.at).c_str(), int(s.AddressU), int(s.AddressV), int(s.AddressW), s.MipLODBias,
                      s.MaxAnisotropy, int(s.ComparisonFunc), s.BorderColor[0], s.BorderColor[1], s.BorderColor[2],
                      s.BorderColor[3], s.MinLOD, s.MaxLOD);
        break;
    }
    case Op::copy_simple:
        std::snprintf(text, sizeof text, "CopyDescriptorsSimple(%u, %s <- %s, %s)", op.count, place(op.at).c_str(),
                      place(op.from).c_str(), heap_type_name(op.copy_type));
        break;
    case Op::copy: {
        std::string d, s;
        for (size_t i = 0; i < op.destinations.size(); ++i)
            d += (i ? " " : "") + place(op.destinations[i]) + "x" + std::to_string(op.destination_sizes[i]);
        for (size_t i = 0; i < op.sources.size(); ++i)
            s += (i ? " " : "") + place(op.sources[i]) + "x" + std::to_string(op.source_sizes[i]);
        std::snprintf(text, sizeof text, "CopyDescriptors(%s <- %s, %s%s)", d.c_str(), s.c_str(),
                      heap_type_name(op.copy_type), op.null_sizes ? ", null sizes" : "");
        break;
    }
    }
    return text;
}

std::string describe(const Probe& p) {
    char text[256];
    std::snprintf(text, sizeof text, "%s view h%u base %u index %u, sampler h%u base %u index %u, level %u layer %u",
                  kind_names[p.kind], p.view_heap, p.view_base, p.view_index, p.sampler_heap, p.sampler_base,
                  p.sampler_index, p.level, p.layer);
    return text;
}

void print_program(const Program& program) {
    for (UINT t = 0; t < kTextures; ++t) {
        const auto& s = program.textures[t];
        std::printf("  tex%u: fmt %d %ux%u mips %u layers %u%s\n", t, int(s.format), s.width, s.height, s.mips,
                    s.layers, s.uav ? " uav" : "");
    }
    for (size_t h = 0; h < program.heaps.size(); ++h)
        std::printf("  h%zu: %s heap of %u, %s\n", h, heap_type_name(program.heaps[h].type), program.heaps[h].size,
                    program.heaps[h].visible ? "shader-visible" : "CPU-only");
    for (size_t r = 0; r < program.rounds.size(); ++r) {
        std::printf("  round %zu\n", r);
        for (size_t i = 0; i < program.rounds[r].ops.size(); ++i)
            std::printf("    op %zu: %s\n", i, describe(program.rounds[r].ops[i]).c_str());
        for (size_t i = 0; i < program.rounds[r].probes.size(); ++i)
            std::printf("    probe %zu: %s\n", i, describe(program.rounds[r].probes[i]).c_str());
    }
}

// ---- the bytes both sides upload ----

std::uint8_t content_byte(std::uint64_t seed, std::uint32_t resource, std::uint32_t sub, std::uint32_t offset) {
    std::uint64_t z = seed ^ (std::uint64_t(resource) << 56) ^ (std::uint64_t(sub) << 40) ^ offset;
    z = (z ^ (z >> 33)) * 0xff51afd7ed558ccdull;
    z = (z ^ (z >> 33)) * 0xc4ceb9fe1a85ec53ull;
    return std::uint8_t(z ^ (z >> 33));
}

// ---- running a program on one device ----

const char* const host_shader = R"(
cbuffer Probe : register(b0, space100) { uint kind, view, smp, first, level, layer; };
RWByteAddressBuffer result : register(u0, space100);
Texture2D<float4> tex2d[16] : register(t0, space0);
Texture2DArray<float4> tex2darray[16] : register(t0, space1);
Buffer<uint4> typed[16] : register(t0, space2);
ByteAddressBuffer raw[16] : register(t0, space3);
StructuredBuffer<uint> s4[16] : register(t0, space4);
StructuredBuffer<uint4> s16[16] : register(t0, space5);
RWBuffer<uint> rtyped[16] : register(u0, space0);
RWByteAddressBuffer rraw[16] : register(u0, space1);
RWTexture2D<uint> rtex[16] : register(u0, space2);
struct Rows { uint4 v[64]; };
ConstantBuffer<Rows> cbs[16] : register(b0, space0);
SamplerState samplers[16] : register(s0, space0);

float2 sample_uv(uint i, uint2 size) {
    int2 g = int2(i % 8, i / 8);
    int2 s = max(int2(size) / 2, int2(1, 1));
    return (float2(g * s - 2 * s) + 0.5) / float2(max(int2(size), int2(1, 1)));
}

[numthreads(64, 1, 1)]
void main(uint3 t : SV_GroupThreadID) {
    uint i = t.x;
    uint4 r = uint4(0, 0, 0, 0);
    uint w, h, n;
    if (kind == 0) {
        tex2d[view].GetDimensions(0, w, h, n);
        r = asuint(tex2d[view].SampleLevel(samplers[smp], sample_uv(i, uint2(w, h)), float(level)));
    } else if (kind == 2) {
        tex2darray[view].GetDimensions(0, w, h, n, n);
        r = asuint(tex2darray[view].SampleLevel(samplers[smp], float3(sample_uv(i, uint2(w, h)), float(layer)),
                                                float(level)));
    } else if (kind == 1) {
        tex2d[view].GetDimensions(level, w, h, n);
        int2 xy = int2(i % 8, i / 8) % max(int2(w, h), int2(1, 1));
        r = asuint(tex2d[view].Load(int3(xy, level)));
    } else if (kind == 3) {
        r = typed[view].Load(i);
    } else if (kind == 4) {
        r.x = raw[view].Load(i * 4);
    } else if (kind == 5) {
        r.x = s4[view][i];
    } else if (kind == 6) {
        r = s16[view][i];
    } else if (kind == 7) {
        if (i < level) r = cbs[view].v[i];
    } else if (kind == 8) {
        r.x = rtyped[view][i];
    } else if (kind == 9) {
        r.x = rraw[view].Load(i * 4);
    } else if (kind == 10) {
        rtex[view].GetDimensions(w, h);
        r.x = rtex[view][int2(i % 8, i / 8) % max(int2(w, h), int2(1, 1))];
    }
    result.Store4((first + i) * 16, r);
}
)";

// The root signature both sides use: [0] six constants (the probe), [1] the output as a root UAV, [2] one table of
// every view range type at offset 0 (each in its own register space), [3] a sampler table.
std::vector<std::uint8_t> root_signature_blob() {
    static D3D12_DESCRIPTOR_RANGE views[10];
    const D3D12_DESCRIPTOR_RANGE_TYPE types[10] = {
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        D3D12_DESCRIPTOR_RANGE_TYPE_UAV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        D3D12_DESCRIPTOR_RANGE_TYPE_CBV};
    const UINT spaces[10] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 0};
    for (UINT i = 0; i < 10; ++i) views[i] = {types[i], kTableSpan, 0, spaces[i], 0};
    static D3D12_DESCRIPTOR_RANGE samplers{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, kTableSpan, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[4] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 100, 6};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor = {0, 100};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {10, views};
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[3].DescriptorTable = {1, &samplers};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {4, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    return serialize_root_signature(desc);
}

struct Side {
    const char* name = "";
    ID3D12Device* device = nullptr;
    bool ours = false;  // the layer: heap start handles through the Xbox ABI
    bool cbv_upload = false;  // the constant buffer in an UPLOAD heap (--cbv-upload), else DEFAULT
    Ref<ID3D12RootSignature> root;
    Ref<ID3D12PipelineState> pso;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    UINT64 fence_value = 0;
    UINT view_step = 0, sampler_step = 0;
    std::string error;

    bool fail(const char* what, HRESULT hr = S_OK) {
        char text[160];
        std::snprintf(text, sizeof text, "%s: %s (HRESULT 0x%08x)", name, what, static_cast<unsigned>(hr));
        error = text;
        return false;
    }

    bool init(const std::vector<std::uint8_t>& root_blob, const void* code, size_t code_size) {
        HRESULT hr = device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p));
        if (FAILED(hr)) return fail("CreateRootSignature", hr);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {code, code_size};
        hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p));
        if (FAILED(hr)) return fail("CreateComputePipelineState", hr);
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p));
        if (FAILED(hr)) return fail("CreateCommandQueue", hr);
        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p));
        if (FAILED(hr)) return fail("CreateFence", hr);
        view_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        sampler_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        return true;
    }

    std::uint64_t cpu_start(ID3D12DescriptorHeap* heap) const {
        if (ours) return reinterpret_cast<XboxHeapHandleFn>(slot(heap, 9))(heap);
        return heap->GetCPUDescriptorHandleForHeapStart().ptr;
    }
    std::uint64_t gpu_start(ID3D12DescriptorHeap* heap) const {
        if (ours) return reinterpret_cast<XboxHeapHandleFn>(slot(heap, 10))(heap);
        return heap->GetGPUDescriptorHandleForHeapStart().ptr;
    }

    bool submit(ID3D12GraphicsCommandList* list) {
        if (FAILED(list->Close())) return fail("Close");
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        if (ours && rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success)
            return fail("ExecuteCommandLists replay");
        HRESULT hr = queue->Signal(fence.p, ++fence_value);
        if (FAILED(hr)) return fail("Signal", hr);
        hr = fence->SetEventOnCompletion(fence_value, nullptr);
        if (FAILED(hr)) return fail("SetEventOnCompletion", hr);
        if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed", device->GetDeviceRemovedReason());
        return true;
    }

    // Runs the program; `out` gets each round's output bytes.
    bool run(const Program& program, std::vector<std::vector<std::uint8_t>>& out) {
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        HRESULT hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p));
        if (FAILED(hr)) return fail("CreateCommandAllocator", hr);
        hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(&list.p));
        if (FAILED(hr)) return fail("CreateCommandList", hr);

        // Resources, and the upload of their contents.
        Ref<ID3D12Resource> textures[kTextures], srv_buffer, uav_buffer, cb_buffer, upload, output, readback;
        D3D12_RESOURCE_DESC texture_descs[kTextures]{};
        UINT64 upload_bytes = 3 * kBufferBytes;
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints[kTextures];
        std::vector<UINT> rows[kTextures];
        std::vector<UINT64> row_bytes[kTextures];
        for (UINT t = 0; t < kTextures; ++t) {
            const TextureSpec& spec = program.textures[t];
            D3D12_RESOURCE_DESC& d = texture_descs[t];
            d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            d.Width = spec.width;
            d.Height = spec.height;
            d.DepthOrArraySize = spec.layers;
            d.MipLevels = spec.mips;
            d.Format = spec.format;
            d.SampleDesc.Count = 1;
            d.Flags = spec.uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
            hr = device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d,
                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&textures[t].p));
            if (FAILED(hr)) return fail("CreateCommittedResource(texture)", hr);
            const UINT subresources = UINT(spec.mips) * spec.layers;
            footprints[t].resize(subresources);
            rows[t].resize(subresources);
            row_bytes[t].resize(subresources);
            UINT64 total = 0;
            device->GetCopyableFootprints(&d, 0, subresources, upload_bytes, footprints[t].data(), rows[t].data(),
                                          row_bytes[t].data(), &total);
            upload_bytes += (total + 511) & ~UINT64(511);
        }
        const auto buffer = [&](const D3D12_HEAP_PROPERTIES& heap, UINT64 bytes, D3D12_RESOURCE_FLAGS flags,
                                D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& resource) {
            const auto d = buffer_desc(bytes, flags);
            return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                                             IID_PPV_ARGS(&resource.p)));
        };
        const UINT64 output_bytes = UINT64(kMaxProbes) * kProbeBytes;
        if (!buffer(default_heap, kBufferBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, srv_buffer) ||
            !buffer(default_heap, kBufferBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST, uav_buffer) ||
            !buffer(cbv_upload ? upload_heap : default_heap, kBufferBytes, D3D12_RESOURCE_FLAG_NONE,
                    cbv_upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST, cb_buffer) ||
            !buffer(upload_heap, upload_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, upload) ||
            !buffer(default_heap, output_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, output) ||
            !buffer(readback_heap, output_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, readback))
            return fail("CreateCommittedResource(buffer)");
        std::uint8_t* up = nullptr;
        if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up)))) return fail("Map(upload)");
        for (UINT b = 0; b < 3; ++b)  // the SRV buffer, the UAV buffer and the constants
            for (UINT i = 0; i < kBufferBytes; ++i) up[b * kBufferBytes + i] = content_byte(program.seed, 100 + b, 0, i);
        for (UINT t = 0; t < kTextures; ++t)
            for (size_t s = 0; s < footprints[t].size(); ++s) {
                const auto& f = footprints[t][s];
                for (UINT y = 0; y < rows[t][s]; ++y)
                    for (UINT64 x = 0; x < row_bytes[t][s]; ++x)
                        up[f.Offset + UINT64(y) * f.Footprint.RowPitch + x] =
                            content_byte(program.seed, t, UINT(s), UINT(y * row_bytes[t][s] + x));
            }
        upload->Unmap(0, nullptr);
        std::vector<D3D12_RESOURCE_BARRIER> ready;
        if (cbv_upload) {
            std::uint8_t* cb = nullptr;
            if (FAILED(cb_buffer->Map(0, nullptr, reinterpret_cast<void**>(&cb)))) return fail("Map(constants)");
            for (UINT i = 0; i < kBufferBytes; ++i) cb[i] = content_byte(program.seed, 102, 0, i);
            cb_buffer->Unmap(0, nullptr);
        } else {
            list->CopyBufferRegion(cb_buffer.p, 0, upload.p, 2 * kBufferBytes, kBufferBytes);
            ready.push_back(transition(cb_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                       D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));
        }

        list->CopyBufferRegion(srv_buffer.p, 0, upload.p, 0, kBufferBytes);
        list->CopyBufferRegion(uav_buffer.p, 0, upload.p, kBufferBytes, kBufferBytes);
        for (UINT t = 0; t < kTextures; ++t) {
            for (size_t s = 0; s < footprints[t].size(); ++s) {
                D3D12_TEXTURE_COPY_LOCATION dst{textures[t].p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
                dst.SubresourceIndex = UINT(s);
                D3D12_TEXTURE_COPY_LOCATION src{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                src.PlacedFootprint = footprints[t][s];
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            ready.push_back(transition(textures[t].p, D3D12_RESOURCE_STATE_COPY_DEST,
                                       program.textures[t].uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                               : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
        }
        ready.push_back(transition(srv_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
        ready.push_back(transition(uav_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        list->ResourceBarrier(UINT(ready.size()), ready.data());
        if (!submit(list.p)) return false;

        // Heaps.
        std::vector<Ref<ID3D12DescriptorHeap>> heaps(program.heaps.size());
        std::vector<std::uint64_t> cpu(program.heaps.size()), gpu(program.heaps.size());
        for (size_t h = 0; h < program.heaps.size(); ++h) {
            const auto& spec = program.heaps[h];
            const D3D12_DESCRIPTOR_HEAP_DESC desc{spec.type, spec.size,
                                                  spec.visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
                                                               : D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heaps[h].p));
            if (FAILED(hr)) return fail("CreateDescriptorHeap", hr);
            cpu[h] = cpu_start(heaps[h].p);
            gpu[h] = spec.visible ? gpu_start(heaps[h].p) : 0;
        }
        const auto step = [&](UINT h) {
            return program.heaps[h].type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ? sampler_step : view_step;
        };
        const auto handle = [&](Place p) {
            return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu[p.heap] + std::uint64_t(p.slot) * step(p.heap))};
        };
        const auto resource = [&](int r) -> ID3D12Resource* {
            return r == kNull ? nullptr : r == kSrvBuffer ? srv_buffer.p : r == kUavBuffer ? uav_buffer.p
                                                                                           : textures[r].p;
        };

        out.clear();
        const D3D12_GPU_VIRTUAL_ADDRESS output_va = output->GetGPUVirtualAddress();
        for (size_t r = 0; r < program.rounds.size(); ++r) {
            const Round& round = program.rounds[r];
            for (const Op& op : round.ops) {
                switch (op.type) {
                case Op::create_srv:
                    device->CreateShaderResourceView(resource(op.resource), &op.srv, handle(op.at));
                    break;
                case Op::create_uav:
                    device->CreateUnorderedAccessView(resource(op.resource), nullptr, &op.uav, handle(op.at));
                    break;
                case Op::create_cbv: {
                    D3D12_CONSTANT_BUFFER_VIEW_DESC d{0, 256};
                    if (op.cbv_size) d = {cb_buffer->GetGPUVirtualAddress() + op.cbv_offset, op.cbv_size};
                    device->CreateConstantBufferView(&d, handle(op.at));
                    break;
                }
                case Op::create_sampler:
                    device->CreateSampler(&op.sampler, handle(op.at));
                    break;
                case Op::copy_simple:
                    device->CopyDescriptorsSimple(op.count, handle(op.at), handle(op.from), op.copy_type);
                    break;
                case Op::copy: {
                    std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> d, s;
                    for (Place p : op.destinations) d.push_back(handle(p));
                    for (Place p : op.sources) s.push_back(handle(p));
                    device->CopyDescriptors(UINT(d.size()), d.data(),
                                            op.null_sizes ? nullptr : op.destination_sizes.data(), UINT(s.size()),
                                            s.data(), op.null_sizes ? nullptr : op.source_sizes.data(), op.copy_type);
                    break;
                }
                }
            }

            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr)))
                return fail("Reset");
            list->SetComputeRootSignature(root.p);
            list->SetPipelineState(pso.p);
            list->SetComputeRootUnorderedAccessView(1, output_va);
            UINT bound_views = UINT(-1), bound_samplers = UINT(-1);
            std::uint64_t view_table = 0, sampler_table = 0;
            for (size_t i = 0; i < round.probes.size(); ++i) {
                const Probe& p = round.probes[i];
                if (p.view_heap != bound_views || p.sampler_heap != bound_samplers) {
                    ID3D12DescriptorHeap* bind[] = {heaps[p.view_heap].p, heaps[p.sampler_heap].p};
                    list->SetDescriptorHeaps(2, bind);
                    bound_views = p.view_heap;
                    bound_samplers = p.sampler_heap;
                    view_table = sampler_table = 0;  // tables are set again after a heap change
                }
                const std::uint64_t views = gpu[p.view_heap] + std::uint64_t(p.view_base) * view_step;
                const std::uint64_t samplers = gpu[p.sampler_heap] + std::uint64_t(p.sampler_base) * sampler_step;
                if (views != view_table) list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{views});
                if (samplers != sampler_table)
                    list->SetComputeRootDescriptorTable(3, D3D12_GPU_DESCRIPTOR_HANDLE{samplers});
                view_table = views;
                sampler_table = samplers;
                const std::uint32_t constants[6] = {p.kind, p.view_index, p.sampler_index, UINT(i) * 64, p.level,
                                                    p.layer};
                list->SetComputeRoot32BitConstants(0, 6, constants, 0);
                list->Dispatch(1, 1, 1);
            }
            D3D12_RESOURCE_BARRIER to_copy = transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                        D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            const UINT64 bytes = UINT64(round.probes.size()) * kProbeBytes;
            list->CopyBufferRegion(readback.p, 0, output.p, 0, bytes);
            D3D12_RESOURCE_BARRIER back = transition(output.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list->ResourceBarrier(1, &back);
            if (!submit(list.p)) return false;
            void* mapped = nullptr;
            const D3D12_RANGE read{0, SIZE_T(bytes)};
            if (FAILED(readback->Map(0, &read, &mapped))) return fail("Map(readback)");
            const auto* begin = static_cast<const std::uint8_t*>(mapped);
            out.emplace_back(begin, begin + bytes);
            const D3D12_RANGE none{0, 0};
            readback->Unmap(0, &none);
        }
        return true;
    }
};

// ---- the comparison ----

// The operation that last wrote a slot, found by replaying the program's writes, for the report.
std::string last_writer(const Program& program, size_t round, UINT heap, UINT slot) {
    std::string who = "never written";
    for (size_t r = 0; r <= round; ++r)
        for (size_t i = 0; i < program.rounds[r].ops.size(); ++i) {
            const Op& op = program.rounds[r].ops[i];
            const std::string here = "round " + std::to_string(r) + " op " + std::to_string(i);
            if (op.type == Op::copy_simple) {
                if (op.at.heap == heap && slot >= op.at.slot && slot < op.at.slot + op.count)
                    who = here + " (copied from h" + std::to_string(op.from.heap) + "[" +
                          std::to_string(op.from.slot + slot - op.at.slot) + "])";
            } else if (op.type == Op::copy) {
                for (size_t d = 0; d < op.destinations.size(); ++d)
                    if (op.destinations[d].heap == heap && slot >= op.destinations[d].slot &&
                        slot < op.destinations[d].slot + op.destination_sizes[d])
                        who = here + " (a CopyDescriptors range)";
            } else if (op.at.heap == heap && op.at.slot == slot) {
                who = here + ": " + describe(op);
            }
        }
    return who;
}

struct Options {
    std::uint64_t seed = 1;
    UINT cases = 12;
    int only = -1;
    bool warp = false, d3d12_debug = false, verbose = false, null_border = false, cbv_upload = false;
};

std::uint64_t case_seed(std::uint64_t seed, UINT index) {
    Rng rng(seed * 0x2545f4914f6cdd1dull + index);
    return rng.next();
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool value = i + 1 < argc;
        if (a == "--seed" && value) o.seed = std::strtoull(argv[++i], nullptr, 0);
        else if (a == "--cases" && value) o.cases = UINT(std::strtoul(argv[++i], nullptr, 0));
        else if (a == "--only" && value) o.only = int(std::strtol(argv[++i], nullptr, 0));
        else if (a == "--warp") o.warp = true;
        else if (a == "--d3d12-debug") o.d3d12_debug = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--null-border") o.null_border = true;
        else if (a == "--cbv-upload") o.cbv_upload = true;
        else {
            std::fprintf(stderr, "usage: %s [--seed N] [--cases N] [--only K] [--warp] [--d3d12-debug] [--verbose] "
                                 "[--null-border] [--cbv-upload]\n",
                         argv[0]);
            return false;
        }
    }
    return true;
}

// The host device: WARP with --warp, else hardware as host_adapter.h chooses it (the layer's own GPU when unset).
bool create_host(const Options& o, const vktest::Device& vk, Ref<ID3D12Device>& host, std::string& name) {
    if (o.d3d12_debug) {
        Ref<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug.p)))) debug->EnableDebugLayer();
        else std::printf("--d3d12-debug: the D3D12 debug layer is not installed (Graphics Tools)\n");
    }
    Ref<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
    Ref<IDXGIAdapter> adapter;
    const bool warp = o.warp || hostadapter::wanted() == "warp";
    if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, name)) return false;
    return SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)));
}

// Host debug-layer errors since the last call (with --d3d12-debug), printed; their count.
UINT host_debug_errors(ID3D12Device* host) {
    Ref<ID3D12InfoQueue> queue;
    if (FAILED(host->QueryInterface(IID_PPV_ARGS(&queue.p)))) return 0;
    UINT errors = 0;
    const UINT64 count = queue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T size = 0;
        queue->GetMessage(i, nullptr, &size);
        std::vector<std::uint8_t> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(queue->GetMessage(i, message, &size))) continue;
        if (message->Severity > D3D12_MESSAGE_SEVERITY_ERROR) continue;
        ++errors;
        std::fprintf(stderr, "D3D12 debug layer: %.*s\n", int(message->DescriptionByteLength), message->pDescription);
    }
    queue->ClearStoredMessages();
    return errors;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) return 2;

    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.bindless || !vk.buffer_device_address) {
        std::printf("SKIP: the device lacks the bindless set 0 or buffer device addresses\n");
        return 77;
    }

    // The host: native D3D12 with the HLSL half. Without one (Wine's vkd3d is not Microsoft's runtime) there is
    // nothing to compare with.
    if (GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version")) {
        std::printf("SKIP: the host D3D12 is Wine's, not Microsoft's\n");
        return 77;
    }
    using Compile = decltype(&D3DCompile);
    const HMODULE compiler = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<Compile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    if (!compile) return 77;
    Ref<ID3DBlob> host_code, errors;
    if (FAILED(compile(host_shader, std::strlen(host_shader), nullptr, nullptr, nullptr, "main", "cs_5_1",
                       D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &host_code.p, &errors.p))) {
        std::fprintf(stderr, "HLSL: %s\n", errors.p ? static_cast<const char*>(errors->GetBufferPointer()) : "?");
        return 1;
    }
    Ref<ID3D12Device> host;
    std::string host_name;
    if (!create_host(options, vk, host, host_name)) {
        std::printf("SKIP: no host D3D12 device\n");
        return 77;
    }

    // The layer: the program the PSO carries and a translator index that maps its hash to the GLSL module.
    const auto program_words = fssm_program({0x44465a01, 0xbf810000});
    const auto container = shader_container(program_words);
    const testfiles::ScratchDir scratch("leonida_descriptor_fuzz_test");
    const auto& library = scratch.path();
    if (!testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_DESCRIPTOR_FUZZ_SPV),
                              library / "cs_descriptor_fuzz.spv"))
        return 1;
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_descriptor_fuzz\": {\"stage\": \"cs\", \"fssm_sha256\": \""
              << leonida::sha256_hex(program_words)
              << "\", \"bindings\": [0, 1, 2, 3, 4], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": false, \"wave64\": false}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = true;
    info.bindless = vk.bindless;
    info.bindless_partially_bound = vk.partially_bound;
    info.image_view_min_lod = vk.image_view_min_lod;
    info.sampler_mirror_clamp_to_edge = vk.sampler_mirror_clamp_to_edge;
    info.custom_border_color = vk.custom_border_color;
    info.custom_border_color_without_format = vk.custom_border_color_without_format;
    info.shader_dir = library.string();
    ID3D12Device* inner = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner)))) return 1;
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device), reinterpret_cast<void**>(&layer.p))))
        return 1;
    rt::d3d12::set_device_outer(inner, layer.p);
    if (rt::d3d12::device_push_constants_need_fallback(inner)) {
        std::printf("SKIP: the PSO needs 144 push-constant bytes and this GPU has %u\n",
                    rt::d3d12::device_max_push_constants_size(inner));
        return 77;
    }

    const auto root_blob = root_signature_blob();
    if (root_blob.empty()) {
        std::fprintf(stderr, "the host could not serialize the root signature\n");
        return 1;
    }
    int exit_code = 0;
    unsigned cases = 0, probes = 0, nonzero = 0, mismatched_cases = 0;  // nonzero: probes the host read data from
    const auto started = std::chrono::steady_clock::now();
    {
        Side native, ours;
        native.name = "host";
        native.device = host.p;
        ours.name = "layer";
        ours.device = layer.p;
        ours.ours = true;
        native.cbv_upload = ours.cbv_upload = options.cbv_upload;
        if (!native.init(root_blob, host_code->GetBufferPointer(), host_code->GetBufferSize()) ||
            !ours.init(root_blob, container.data(), container.size())) {
            std::fprintf(stderr, "%s%s\n", native.error.c_str(), ours.error.c_str());
            return 1;
        }
        // The GLSL half indexes the tables with the console's strides.
        if (ours.view_step != 32 || ours.sampler_step != 32) {
            std::fprintf(stderr, "layer descriptor sizes %u/%u, the shader assumes 32/32\n", ours.view_step,
                         ours.sampler_step);
            return 1;
        }
        std::printf("descriptor fuzz: seed %llu, host %s\n", static_cast<unsigned long long>(options.seed),
                    host_name.c_str());

        const UINT first = options.only >= 0 ? UINT(options.only) : 0;
        const UINT last = options.only >= 0 ? UINT(options.only) + 1 : options.cases;
        for (UINT c = first; c < last; ++c) {
            const Program program = Generator(case_seed(options.seed, c), vk.image_view_min_lod,
                                              vk.sampler_mirror_clamp_to_edge, options.null_border).build();
            if (options.verbose) {
                std::printf("case %u:\n", c);
                print_program(program);
            }
            std::vector<std::vector<std::uint8_t>> want, got;
            if (!native.run(program, want)) {
                std::fprintf(stderr, "case %u: %s\n", c, native.error.c_str());
                return 1;
            }
            if (options.d3d12_debug && host_debug_errors(host.p)) {
                std::fprintf(stderr, "case %u: the host debug layer reported errors: the program is not valid\n", c);
                return 1;
            }
            if (!ours.run(program, got)) {
                std::fprintf(stderr, "case %u: %s\n", c, ours.error.c_str());
                exit_code = 1;
                ++mismatched_cases;
                continue;
            }
            ++cases;
            unsigned reported = 0;
            bool mismatch = false;
            for (size_t r = 0; r < program.rounds.size(); ++r) {
                const Round& round = program.rounds[r];
                probes += unsigned(round.probes.size());
                for (size_t p = 0; p < round.probes.size(); ++p) {
                    const std::uint8_t* w = want[r].data() + p * kProbeBytes;
                    const std::uint8_t* g = got[r].data() + p * kProbeBytes;
                    if (std::any_of(w, w + kProbeBytes, [](std::uint8_t b) { return b != 0; })) ++nonzero;
                    if (std::memcmp(w, g, kProbeBytes) == 0) continue;
                    mismatch = true;
                    if (reported++ >= 4) continue;
                    const Probe& probe = round.probes[p];
                    const UINT slot = probe.view_base + probe.view_index;
                    std::fprintf(stderr, "case %u round %zu probe %zu: %s\n  view slot h%u[%u]: %s\n", c, r, p,
                                 describe(probe).c_str(), probe.view_heap, slot,
                                 last_writer(program, r, probe.view_heap, slot).c_str());
                    if (sampled_kind(probe.kind)) {
                        const UINT s = probe.sampler_base + probe.sampler_index;
                        std::fprintf(stderr, "  sampler slot h%u[%u]: %s\n", probe.sampler_heap, s,
                                     last_writer(program, r, probe.sampler_heap, s).c_str());
                    }
                    unsigned shown = 0;
                    for (UINT i = 0; i < 64 && shown < 4; ++i) {
                        std::uint32_t a[4], b[4];
                        std::memcpy(a, w + i * 16, 16);
                        std::memcpy(b, g + i * 16, 16);
                        if (std::memcmp(a, b, 16) == 0) continue;
                        ++shown;
                        std::fprintf(stderr, "  invocation %2u: host %08x %08x %08x %08x  layer %08x %08x %08x %08x\n",
                                     i, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
                    }
                }
            }
            if (mismatch) {
                std::fprintf(stderr, "case %u: %u probe(s) differ; repro: d3d12_descriptor_fuzz_test --seed %llu "
                                     "--only %u --verbose\n",
                             c, reported, static_cast<unsigned long long>(options.seed), c);
                exit_code = 1;
                ++mismatched_cases;
            }
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        exit_code = 1;
    }
    std::printf("descriptor fuzz: %u case(s), %u probes of %u bytes compared (%u read data, the rest zeros), %u case(s) "
                "differ, %u validation errors, %.1f s\n",
                cases, probes, kProbeBytes, nonzero, mismatched_cases, vk.validation_errors(), seconds);
    return exit_code;
}

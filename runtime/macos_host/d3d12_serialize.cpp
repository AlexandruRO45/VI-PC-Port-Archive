// d3d12.dll!D3D12SerializeVersionedRootSignature for Leonida's tests (test/d3d12_test_shaders.h), which take it from
// the system on Windows and from vkd3d under Wine. Writes a DXBC container with one RTS0 chunk in the layout
// Leonida's decoder reads (src/root_signature.cpp, parse_rts0_payload) and returns it in an ID3DBlob.
// ponytail: the DXBC hash is left zero; Leonida's decoder does not check it.
#include "host.h"

#include <atomic>
#include <cstring>

namespace host {
namespace {

// ---- ID3DBlob ----------------------------------------------------------------------------------------------------
struct Blob;
struct BlobVtbl {
    void* query_interface;
    void* add_ref;
    void* release;
    void* get_buffer_pointer;
    void* get_buffer_size;
};
struct Blob {
    const BlobVtbl* vtbl;
    std::atomic<uint32_t> refs;
    std::vector<uint8_t> data;
};
int32_t QueryInterface(Blob* b, const void*, void** out) {  // ponytail: IUnknown and ID3DBlob both answer
    b->refs.fetch_add(1);
    *out = b;
    return 0;
}
uint32_t AddRef(Blob* b) { return b->refs.fetch_add(1) + 1; }
uint32_t Release(Blob* b) {
    uint32_t left = b->refs.fetch_sub(1) - 1;
    if (!left) delete b;
    return left;
}
void* GetBufferPointer(Blob* b) { return b->data.data(); }
uint64_t GetBufferSize(Blob* b) { return b->data.size(); }

#define METHOD(f) reinterpret_cast<void*>(&Thunk<decltype(f), &f>::call)
const BlobVtbl kBlobVtbl = {METHOD(QueryInterface), METHOD(AddRef), METHOD(Release), METHOD(GetBufferPointer),
                            METHOD(GetBufferSize)};

// ---- the RTS0 writer ---------------------------------------------------------------------------------------------
template <typename T> T rd(const uint8_t* p, size_t off) {
    T v;
    std::memcpy(&v, p + off, sizeof v);
    return v;
}
struct Writer {
    std::vector<uint8_t> b;
    size_t u32(uint32_t v) {
        size_t at = b.size();
        b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 4);
        return at;
    }
    void set(size_t at, uint32_t v) { std::memcpy(b.data() + at, &v, 4); }
    void raw(const uint8_t* p, size_t n) { b.insert(b.end(), p, p + n); }
};

// `desc` is D3D12_VERSIONED_ROOT_SIGNATURE_DESC in guest memory: u32 Version, then the matching
// D3D12_ROOT_SIGNATURE_DESC{,1,2} at +8 (NumParameters, pParameters@8, NumStaticSamplers@16, pStaticSamplers@24,
// Flags@32). Parameters are 32 bytes (type, union@8, visibility@24).
std::vector<uint8_t> rts0(const uint8_t* desc) {
    uint32_t version = rd<uint32_t>(desc, 0);
    const uint8_t* d = desc + 8;
    bool v11 = version >= 2, v12 = version == 3;
    uint32_t nparams = rd<uint32_t>(d, 0), nsamplers = rd<uint32_t>(d, 16);
    auto* params = rd<const uint8_t*>(d, 8);
    auto* samplers = rd<const uint8_t*>(d, 24);

    Writer w;
    w.u32(version);
    w.u32(nparams);
    size_t params_off = w.u32(0);
    w.u32(nsamplers);
    size_t samplers_off = w.u32(0);
    w.u32(rd<uint32_t>(d, 32));
    w.set(params_off, static_cast<uint32_t>(w.b.size()));
    std::vector<size_t> payload_at(nparams);
    for (uint32_t i = 0; i < nparams; ++i) {
        const uint8_t* p = params + 32 * i;
        w.u32(rd<uint32_t>(p, 0));
        w.u32(rd<uint32_t>(p, 24));
        payload_at[i] = w.u32(0);
    }
    for (uint32_t i = 0; i < nparams; ++i) {
        const uint8_t* p = params + 32 * i;
        uint32_t type = rd<uint32_t>(p, 0);
        w.set(payload_at[i], static_cast<uint32_t>(w.b.size()));
        if (type == 0) {  // descriptor table: {count, offset}, then the ranges
            uint32_t n = rd<uint32_t>(p, 8);
            auto* ranges = rd<const uint8_t*>(p, 16);
            w.u32(n);
            size_t ranges_off = w.u32(0);
            w.set(ranges_off, static_cast<uint32_t>(w.b.size()));
            w.raw(ranges, n * (v11 ? 24 : 20));  // D3D12_DESCRIPTOR_RANGE(1): already the RTS0 field order
        } else if (type == 1) {               // 32-bit constants
            w.raw(p + 8, 12);
        } else {                              // root CBV/SRV/UAV: register, space (+ flags from 1.1)
            w.raw(p + 8, v11 ? 12 : 8);
        }
    }
    w.set(samplers_off, static_cast<uint32_t>(w.b.size()));
    w.raw(samplers, nsamplers * (v12 ? 56 : 52));  // D3D12_STATIC_SAMPLER_DESC(1) field order

    Writer c;  // DXBC container: magic, hash, version 1, total size, 1 chunk at 36
    c.raw(reinterpret_cast<const uint8_t*>("DXBC"), 4);
    c.b.resize(20, 0);
    c.u32(1);
    size_t total = c.u32(0);
    c.u32(1);
    c.u32(36);
    c.raw(reinterpret_cast<const uint8_t*>("RTS0"), 4);
    c.u32(static_cast<uint32_t>(w.b.size()));
    c.raw(w.b.data(), w.b.size());
    c.set(total, static_cast<uint32_t>(c.b.size()));
    return c.b;
}

int32_t D3D12SerializeVersionedRootSignature(const uint8_t* desc, Blob** out, Blob** error) {
    if (error) *error = nullptr;
    if (!desc || !out) return static_cast<int32_t>(0x80070057);  // E_INVALIDARG
    uint32_t version = rd<uint32_t>(desc, 0);
    if (version < 1 || version > 3) return static_cast<int32_t>(0x80070057);
    *out = new Blob{&kBlobVtbl, {1}, rts0(desc)};
    return 0;
}

}  // namespace

Exports d3d12_serialize_exports() { return {EXPORT("d3d12.dll", D3D12SerializeVersionedRootSignature)}; }

}  // namespace host

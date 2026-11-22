// CreateCommandSignature with a root signature on the Vulkan-backed device, compared with the host PC runtime.
// compare_invalid_arguments (d3d12_vulkan_device_test) covers signatures without one; this covers the root arguments
// a signature can only have with one: root constants (parameter index, DestOffsetIn32BitValues, Num32BitValuesToSet),
// root CBV/SRV/UAV addresses (index and parameter type), vertex buffer slots and the index buffer view, alone and
// combined, with draws and dispatches, and a stride too small for them. Both devices get the same root signature, made
// from one blob serialized by the host runtime. Every HRESULT is recorded, and whether a refused call wrote its
// output. The two records must be the same, step by step. A step the host runtime faults on is reported and not
// compared. No shaders or game data. Skips (77) without a Vulkan GPU, and without a host D3D12 runtime (it serializes
// the root signature). Set LEONIDA_D3D49_VERBOSE=1 to print every step.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr std::uint64_t kFaulted = 0xFA017EDull;  // the call raised an exception
void* const kSentinel = reinterpret_cast<void*>(std::uintptr_t(0x5E17E1));

// The root signature both devices get: parameter
//   0  4 root constants (b0)       3  root UAV (u0)
//   1  root CBV (b1)               4  descriptor table, one SRV range (t1)
//   2  root SRV (t0)               5  1 root constant (b2)
// with the input assembler allowed, so it serves draws and dispatches alike.
std::vector<std::uint8_t> serialize_root_signature(HMODULE d3d12) {
    const auto serialize = reinterpret_cast<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(
        reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeRootSignature")));
    if (!serialize) return {};
    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, 0};
    D3D12_ROOT_PARAMETER params[6]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 4};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[1].Descriptor = {1, 0};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor = {0, 0};
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].Descriptor = {0, 0};
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[4].DescriptorTable = {1, &range};
    params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[5].Constants = {2, 0, 1};
    for (auto& p : params) p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    const D3D12_ROOT_SIGNATURE_DESC desc{6, params, 0, nullptr,
                                         D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    ID3DBlob* blob = nullptr;
    ID3DBlob* error = nullptr;
    std::vector<std::uint8_t> out;
    if (SUCCEEDED(serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error)) && blob) {
        const auto* bytes = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(bytes, bytes + blob->GetBufferSize());
    }
    if (blob) blob->Release();
    if (error) error->Release();
    return out;
}

D3D12_INDIRECT_ARGUMENT_DESC arg(D3D12_INDIRECT_ARGUMENT_TYPE type) {
    D3D12_INDIRECT_ARGUMENT_DESC a{};
    a.Type = type;
    return a;
}
D3D12_INDIRECT_ARGUMENT_DESC constant(UINT index, UINT offset, UINT count) {
    D3D12_INDIRECT_ARGUMENT_DESC a = arg(D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT);
    a.Constant = {index, offset, count};
    return a;
}
D3D12_INDIRECT_ARGUMENT_DESC view(D3D12_INDIRECT_ARGUMENT_TYPE type, UINT index) {
    D3D12_INDIRECT_ARGUMENT_DESC a = arg(type);
    a.ConstantBufferView.RootParameterIndex = index;  // the same field for CBV, SRV and UAV
    return a;
}
D3D12_INDIRECT_ARGUMENT_DESC cbv(UINT index) { return view(D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW, index); }
D3D12_INDIRECT_ARGUMENT_DESC srv(UINT index) { return view(D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW, index); }
D3D12_INDIRECT_ARGUMENT_DESC uav(UINT index) { return view(D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW, index); }
D3D12_INDIRECT_ARGUMENT_DESC vb(UINT slot) {
    D3D12_INDIRECT_ARGUMENT_DESC a = arg(D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW);
    a.VertexBuffer.Slot = slot;
    return a;
}
const D3D12_INDIRECT_ARGUMENT_DESC kDraw = arg(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW);                  // 16 bytes
const D3D12_INDIRECT_ARGUMENT_DESC kDrawIndexed = arg(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED);   // 20 bytes
const D3D12_INDIRECT_ARGUMENT_DESC kDispatch = arg(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH);          // 12 bytes
const D3D12_INDIRECT_ARGUMENT_DESC kIndexBuffer = arg(D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW);  // 16 bytes
const D3D12_INDIRECT_ARGUMENT_DESC kDispatchMesh = arg(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH);  // 12 bytes
D3D12_INDIRECT_ARGUMENT_DESC incrementing(UINT index, UINT offset) {
    D3D12_INDIRECT_ARGUMENT_DESC a = arg(D3D12_INDIRECT_ARGUMENT_TYPE_INCREMENTING_CONSTANT);
    a.IncrementingConstant = {index, offset};
    return a;
}

struct Step {
    std::string name;
    UINT stride;
    std::vector<D3D12_INDIRECT_ARGUMENT_DESC> args;
    bool with_root_signature = true;
};

std::vector<Step> steps() {
    std::vector<Step> s;
    const auto add = [&](std::string name, UINT stride, std::initializer_list<D3D12_INDIRECT_ARGUMENT_DESC> args,
                         bool with_root_signature = true) {
        s.push_back({std::move(name), stride, args, with_root_signature});
    };
    add("draw only", 16, {kDraw});
    add("dispatch only", 12, {kDispatch});
    // Root constants into parameter 0 (4 values) and 5 (1 value).
    add("constant p0 [0, 1), draw", 20, {constant(0, 0, 1), kDraw});
    add("constant p0 [0, 4), draw", 32, {constant(0, 0, 4), kDraw});
    add("constant p0 [3, 4), draw", 20, {constant(0, 3, 1), kDraw});
    add("constant p0 [3, 5): past its 4 values", 24, {constant(0, 3, 2), kDraw});
    add("constant p0 [4, 5): offset at the end", 20, {constant(0, 4, 1), kDraw});
    add("constant p0 [0, 5): one value too many", 36, {constant(0, 0, 5), kDraw});
    add("constant p0, no values", 16, {constant(0, 0, 0), kDraw});
    add("constant p0 offset 0xffffffff", 20, {constant(0, 0xffffffffu, 1), kDraw});
    add("constant p5 [0, 1), draw", 20, {constant(5, 0, 1), kDraw});
    add("constant p5 [0, 2): past its 1 value", 24, {constant(5, 0, 2), kDraw});
    add("constant p6: no such parameter", 20, {constant(6, 0, 1), kDraw});
    add("constant p0xffffffff", 20, {constant(0xffffffffu, 0, 1), kDraw});
    add("constant p1: a root CBV", 20, {constant(1, 0, 1), kDraw});
    add("constant p4: a descriptor table", 20, {constant(4, 0, 1), kDraw});
    add("constant p0, dispatch", 16, {constant(0, 0, 1), kDispatch});
    add("constants p0 [0, 2) and [2, 4)", 32, {constant(0, 0, 2), constant(0, 2, 2), kDraw});
    add("constants p0 [0, 2) and [1, 3): overlapping", 32, {constant(0, 0, 2), constant(0, 1, 2), kDraw});
    add("constants p0 [0, 1) twice", 24, {constant(0, 0, 1), constant(0, 0, 1), kDraw});
    add("constants p0 and p5", 24, {constant(0, 0, 1), constant(5, 0, 1), kDraw});
    // Root descriptor addresses.
    add("CBV p1, draw", 24, {cbv(1), kDraw});
    add("CBV p0: root constants", 24, {cbv(0), kDraw});
    add("CBV p2: a root SRV", 24, {cbv(2), kDraw});
    add("CBV p3: a root UAV", 24, {cbv(3), kDraw});
    add("CBV p4: a descriptor table", 24, {cbv(4), kDraw});
    add("CBV p6: no such parameter", 24, {cbv(6), kDraw});
    add("SRV p2, draw", 24, {srv(2), kDraw});
    add("SRV p1: a root CBV", 24, {srv(1), kDraw});
    add("SRV p3: a root UAV", 24, {srv(3), kDraw});
    add("UAV p3, dispatch", 20, {uav(3), kDispatch});
    add("UAV p2: a root SRV", 20, {uav(2), kDispatch});
    add("UAV p4: a descriptor table", 20, {uav(4), kDispatch});
    add("CBV p1 twice", 32, {cbv(1), cbv(1), kDraw});
    add("UAV p3, draw", 24, {uav(3), kDraw});
    add("SRV p2, dispatch", 20, {srv(2), kDispatch});
    add("CBV p1, dispatch", 20, {cbv(1), kDispatch});
    add("CBV p1, SRV p2 and constants p0, draw", 48, {cbv(1), srv(2), constant(0, 0, 4), kDraw});
    add("CBV p1, SRV p2, UAV p3 and constants p0, dispatch", 52, {cbv(1), srv(2), uav(3), constant(0, 0, 4), kDispatch});
    add("constants p0 then CBV p1, draw", 28, {constant(0, 0, 1), cbv(1), kDraw});
    add("CBV p1 then constants p0, draw", 28, {cbv(1), constant(0, 0, 1), kDraw});
    add("constant p5 then CBV p1, draw", 28, {constant(5, 0, 1), cbv(1), kDraw});
    add("constant p5 then constant p0, draw", 24, {constant(5, 0, 1), constant(0, 0, 1), kDraw});
    add("SRV p2 then CBV p1, draw", 32, {srv(2), cbv(1), kDraw});
    add("CBV p1 then SRV p2, draw", 32, {cbv(1), srv(2), kDraw});
    add("constants p0 [2, 4) then [0, 2), draw", 32, {constant(0, 2, 2), constant(0, 0, 2), kDraw});
    add("CBV p1, SRV p2, UAV p3, draw", 40, {cbv(1), srv(2), uav(3), kDraw});
    add("CBV p1, SRV p2, UAV p3 and constants p0", 56, {cbv(1), srv(2), uav(3), constant(0, 0, 4), kDraw});
    add("constant p0 [0, 1) and CBV p1, dispatch", 24, {constant(0, 0, 1), cbv(1), kDispatch});
    // Vertex and index buffer views. A root signature with no root argument is a case of its own, so each of these
    // also sets a root constant (4 bytes, first) unless its name says "only".
    add("VB slot 0 only", 32, {vb(0), kDraw});
    add("IB only, draw indexed", 36, {kIndexBuffer, kDrawIndexed});
    add("VB slot 0, draw", 36, {constant(5, 0, 1), vb(0), kDraw});
    add("VB slot 31, draw", 36, {constant(5, 0, 1), vb(31), kDraw});
    add("VB slot 32", 36, {constant(5, 0, 1), vb(32), kDraw});
    add("VB slot 0xffffffff", 36, {constant(5, 0, 1), vb(0xffffffffu), kDraw});
    add("VB slots 0 and 1, draw", 52, {constant(5, 0, 1), vb(0), vb(1), kDraw});
    add("VB slot 0 twice", 52, {constant(5, 0, 1), vb(0), vb(0), kDraw});
    add("VB slot 1 then slot 0", 52, {constant(5, 0, 1), vb(1), vb(0), kDraw});
    add("VB slot 0 before the root constant", 36, {vb(0), constant(5, 0, 1), kDraw});
    add("VB slot 0, dispatch", 32, {constant(5, 0, 1), vb(0), kDispatch});
    add("IB, draw indexed", 40, {constant(5, 0, 1), kIndexBuffer, kDrawIndexed});
    add("IB, draw (not indexed)", 36, {constant(5, 0, 1), kIndexBuffer, kDraw});
    add("IB twice, draw indexed", 56, {constant(5, 0, 1), kIndexBuffer, kIndexBuffer, kDrawIndexed});
    add("IB, dispatch", 32, {constant(5, 0, 1), kIndexBuffer, kDispatch});
    add("VB slot 0, IB and constant p0, draw indexed", 56, {vb(0), kIndexBuffer, constant(0, 0, 1), kDrawIndexed});    // Strides against the root arguments' size.
    add("constant p0 [0, 4), draw, stride 31", 31, {constant(0, 0, 4), kDraw});
    add("constant p0 [0, 4), draw, stride 28 (too small)", 28, {constant(0, 0, 4), kDraw});
    add("CBV p1, draw, stride 20 (too small)", 20, {cbv(1), kDraw});
    add("CBV p1, draw, stride 256", 256, {cbv(1), kDraw});
    // The same root arguments without the root signature, and a valid one with it but a root argument after the draw.
    add("constant p0 without a root signature", 20, {constant(0, 0, 1), kDraw}, false);
    add("CBV p1 without a root signature", 24, {cbv(1), kDraw}, false);
    add("VB slot 0, dispatch, without a root signature", 28, {vb(0), kDispatch}, false);
    add("VB slot 32, draw, without a root signature", 32, {vb(32), kDraw}, false);
    add("VB slot 0 twice, draw, without a root signature", 48, {vb(0), vb(0), kDraw}, false);
    add("VB slots 0 and 1, draw, without a root signature", 48, {vb(0), vb(1), kDraw}, false);
    add("IB, draw (not indexed), without a root signature", 32, {kIndexBuffer, kDraw}, false);
    add("IB, draw indexed, without a root signature", 36, {kIndexBuffer, kDrawIndexed}, false);
    add("IB twice, draw indexed, without a root signature", 52, {kIndexBuffer, kIndexBuffer, kDrawIndexed}, false);
    add("draw, then constant p0", 20, {kDraw, constant(0, 0, 1)});
    // Incrementing constants (a root constant the runtime counts up per command) and mesh dispatches.
    add("incrementing constant p5, draw", 20, {incrementing(5, 0), kDraw});
    add("incrementing constant p0 offset 3, draw", 20, {incrementing(0, 3), kDraw});
    add("incrementing constant p0 offset 4: past its 4 values", 20, {incrementing(0, 4), kDraw});
    add("incrementing constant p1: a root CBV", 20, {incrementing(1, 0), kDraw});
    add("incrementing constant p5 without a root signature", 20, {incrementing(5, 0), kDraw}, false);
    add("incrementing constant p5, dispatch", 16, {incrementing(5, 0), kDispatch});
    add("constant p0 [0, 1) and incrementing constant p0 [0, 1)", 24, {constant(0, 0, 1), incrementing(0, 0), kDraw});
    add("dispatch mesh only", 12, {kDispatchMesh});
    add("constant p0 [0, 1), dispatch mesh", 16, {constant(0, 0, 1), kDispatchMesh});
    add("CBV p1, dispatch mesh", 20, {cbv(1), kDispatchMesh});
    add("VB slot 0 and constant p5, dispatch mesh", 32, {constant(5, 0, 1), vb(0), kDispatchMesh});
    add("IB and constant p5, dispatch mesh", 32, {constant(5, 0, 1), kIndexBuffer, kDispatchMesh});
    add("dispatch mesh only, without a root signature", 12, {kDispatchMesh}, false);
    return s;
}

HRESULT create(ID3D12Device* device, const Step& s, ID3D12RootSignature* root, void** out) {
    const D3D12_COMMAND_SIGNATURE_DESC desc{s.stride, UINT(s.args.size()), s.args.data(), 0};
    return device->CreateCommandSignature(&desc, s.with_root_signature ? root : nullptr,
                                          __uuidof(ID3D12CommandSignature), out);
}

HRESULT guarded(ID3D12Device* device, const Step* s, ID3D12RootSignature* root, void** out) {
    __try {
        return create(device, *s, root, out);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(kFaulted);
    }
}

struct Record {
    std::vector<std::string> names;
    std::vector<std::uint64_t> values;
    void add(std::string name, std::uint64_t value) {
        names.push_back(std::move(name));
        values.push_back(value);
    }
};

// The root signature from `blob` on `device`, then every step. A refused call's output is recorded: left alone,
// cleared to null, or (it shouldn't be) an object.
bool run(ID3D12Device* device, const std::vector<std::uint8_t>& blob, Record& out) {
    Ref<ID3D12RootSignature> root;
    const HRESULT made = device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p));
    out.add("CreateRootSignature", std::uint64_t(std::uint32_t(made)));
    if (FAILED(made)) return false;
    for (const Step& s : steps()) {
        void* signature = kSentinel;
        const HRESULT hr = guarded(device, &s, root.p, &signature);
        out.add(s.name, std::uint64_t(std::uint32_t(hr)));
        const bool written = signature != kSentinel;
        if (SUCCEEDED(hr) && signature && written) static_cast<IUnknown*>(signature)->Release();
        out.add(s.name + " (output)", hr == HRESULT(kFaulted) ? kFaulted
                                      : !written           ? 0
                                      : !signature         ? 1
                                                           : 2);
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const bool verbose = GetEnvironmentVariableA("LEONIDA_D3D49_VERBOSE", nullptr, 0) > 0;

    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create_host = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                         reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                                   : nullptr;
    Ref<ID3D12Device> host;
    if (!create_host || FAILED(create_host(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        std::printf("command signatures with a root signature: no host D3D12 runtime, skipped\n");
        return 77;
    }
    const std::vector<std::uint8_t> blob = serialize_root_signature(d3d12);
    EXPECT(!blob.empty());
    Record theirs;
    EXPECT(run(host.p, blob, theirs));

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
    Record ours;
    EXPECT(run(device.p, blob, ours));
    EXPECT(theirs.names == ours.names);

    int mismatches = 0, faulted = 0;
    for (std::size_t i = 0; i < ours.values.size(); ++i) {
        // A mesh dispatch signature the host takes is E_NOTIMPL in the layer until indirect mesh dispatch lands
        // (#3777): that HRESULT and its output row are not compared. Refused mesh signatures still are.
        const std::size_t hr_row = ours.names[i].find("(output)") == std::string::npos ? i : i - 1;
        const bool mesh_pending = ours.names[hr_row].find("dispatch mesh") != std::string::npos &&
                                  ours.values[hr_row] == std::uint32_t(E_NOTIMPL) &&
                                  SUCCEEDED(HRESULT(std::uint32_t(theirs.values[hr_row])));        if (mesh_pending) {
            if (verbose)
                std::printf("command signature %s: ours 0x%llx host 0x%llx (indirect mesh dispatch is #3777; not "
                            "compared)\n", ours.names[i].c_str(), static_cast<unsigned long long>(ours.values[i]),
                            static_cast<unsigned long long>(theirs.values[i]));
            ++faulted;
            continue;
        }
        if (theirs.values[i] == kFaulted) {
            std::printf("command signature %s: the host faults, not compared (ours 0x%llx)\n", ours.names[i].c_str(),
                        static_cast<unsigned long long>(ours.values[i]));
            ++faulted;
            continue;
        }
        const bool same = ours.values[i] == theirs.values[i];
        if (verbose || !same)
            std::printf("command signature %s: ours 0x%llx host 0x%llx%s\n", ours.names[i].c_str(),
                        static_cast<unsigned long long>(ours.values[i]),
                        static_cast<unsigned long long>(theirs.values[i]), same ? "" : "  MISMATCH");
        mismatches += !same;
    }
    std::printf("command signatures with a root signature: %zu values compared with the host (%d more not compared), "
                "%d mismatches\n", ours.values.size() - faulted, faulted, mismatches);
    EXPECT(mismatches == 0);
    std::printf("d3d12_command_signature_root_test: OK\n");
    return 0;
}

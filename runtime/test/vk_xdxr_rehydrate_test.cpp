// GPU rehydration of dehydrated XDXR BLASes (src/xdxr_rehydrate.comp, xdxr_rehydrate.h) against tools/bvh_rehydrate.py,
// the model that matches XboxBvhBuilder.dll byte for byte (docs/bvh.md §4a). runtime/test/xdxr_rehydrate_cases.py
// writes the cases into <dir> first (ctest: xdxr_rehydrate_cases, then this):
//
//   vk_xdxr_rehydrate_test <dir>
//
// Each <case>.src.bin is rehydrated on the device the Vulkan loader picks and compared with <case>.expected.bin byte
// for byte. Also checked: the plan's status is 0, nothing is written past the structure (a guard region keeps its
// fill), a structure that is not dehydrated leaves the destination as it was, and the same buffer as source and
// destination is refused. Exit 77 without a Vulkan device with bufferDeviceAddress. No game files.
#include "vk_test_device.h"
#include "../src/xdxr_rehydrate.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace leonida;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

bool read_file(const std::string& path, std::vector<std::uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    void* map = nullptr;
    std::uint64_t address = 0;
};

constexpr VkBufferUsageFlags kUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

Buffer make(vktest::Device& gpu, std::size_t bytes) {
    Buffer b;
    b.buffer = gpu.buffer(bytes ? bytes : 4, kUsage, &b.map);
    if (b.buffer) b.address = gpu.device_address(b.buffer);
    return b;
}

Buffer upload(vktest::Device& gpu, const void* data, std::size_t bytes) {
    Buffer b = make(gpu, bytes);
    if (b.map) std::memcpy(b.map, data, bytes);
    return b;
}

void barrier_to_host(vktest::Device& gpu, VkCommandBuffer cb) {
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    gpu.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                                                            nullptr);
}

constexpr std::size_t kGuard = 4096;

// Rehydrates one case and compares it. Returns false when the case could not be run at all.
bool run_case(vktest::Device& gpu, XdxrRehydrator& rehydrator, const std::string& dir, const std::string& name) {
    std::vector<std::uint8_t> src, expected;
    if (!read_file(dir + "/" + name + ".src.bin", src) || !read_file(dir + "/" + name + ".expected.bin", expected)) {
        check(false, name + ": case files");
        return false;
    }
    std::ifstream geom(dir + "/" + name + ".geom.txt");
    std::vector<XdxrRehydrateGeometry> geometries;
    std::vector<Buffer> keep;
    std::string line;
    while (std::getline(geom, line)) {
        std::istringstream in(line);
        std::string vertices, indices, transform;
        std::uint32_t stride = 0, index_type = 0;
        in >> vertices >> stride >> indices >> index_type >> transform;
        XdxrRehydrateGeometry g;
        std::vector<std::uint8_t> bytes;
        if (!read_file(dir + "/" + vertices, bytes)) { check(false, name + ": " + vertices); return false; }
        keep.push_back(upload(gpu, bytes.data(), bytes.size()));
        g.vertices = keep.back().address;
        g.vertex_stride = stride;
        if (indices != "-") {
            if (!read_file(dir + "/" + indices, bytes)) { check(false, name + ": " + indices); return false; }
            keep.push_back(upload(gpu, bytes.data(), bytes.size()));
            g.indices = keep.back().address;
            g.index_type = index_type;
        }
        if (transform != "-") {
            float m[12];
            m[0] = std::stof(transform);
            for (int k = 1; k < 12; ++k) in >> m[k];
            keep.push_back(upload(gpu, m, sizeof m));
            g.transform = keep.back().address;
        }
        geometries.push_back(g);
    }

    Buffer s = upload(gpu, src.data(), src.size());
    Buffer d = make(gpu, expected.size() + kGuard);   // mapped buffers start filled with 0xcd
    Buffer plan = make(gpu, std::size_t(XdxrRehydrator::plan_bytes(geometries.size())));
    if (!s.map || !d.map || !plan.map) { check(false, name + ": buffers"); return false; }
    bool recorded = false;
    const bool ran = gpu.run([&](VkCommandBuffer cb) {
        recorded = rehydrator.record(cb, s.address, d.address, plan.address, plan.buffer, 0, geometries);
        barrier_to_host(gpu, cb);
        return recorded;
    });
    check(ran && recorded, name + ": recorded and ran");
    if (!ran) return false;

    std::uint32_t status = 0;
    std::memcpy(&status, static_cast<const char*>(plan.map) + XdxrRehydrator::status_offset, 4);
    check(status == 0, name + ": the plan's status is 0 (got " + std::to_string(status) + ")");
    const auto* got = static_cast<const std::uint8_t*>(d.map);
    std::size_t differ = 0, first = expected.size();
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (got[i] != expected[i]) {
            if (!differ) first = i;
            ++differ;
        }
    char what[200];
    if (differ)
        std::snprintf(what, sizeof what, "%s: %zu bytes, %zu differ from bvh_rehydrate.py (first at 0x%zx)",
                      name.c_str(), expected.size(), differ, first);
    else
        std::snprintf(what, sizeof what, "%s: %zu bytes, identical to bvh_rehydrate.py", name.c_str(), expected.size());
    check(differ == 0, what);
    bool guard = true;
    for (std::size_t i = 0; i < kGuard; ++i) guard = guard && got[expected.size() + i] == 0xcd;
    check(guard, name + ": nothing is written past the structure");
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: vk_xdxr_rehydrate_test <dir written by xdxr_rehydrate_cases.py>\n");
        return 2;
    }
    const std::string dir = argv[1];
    vktest::Device gpu;
    gpu.request_buffer_device_address = true;
    const auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable || (status == vktest::DeviceStatus::ready && !gpu.buffer_device_address)) {
        std::printf("vk_xdxr_rehydrate_test: no Vulkan device with bufferDeviceAddress, skipped\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    XdxrRehydrator rehydrator(gpu.device_proc, gpu.device);
    check(rehydrator.loaded(), "the rehydration pipeline is created");
    if (!rehydrator.loaded()) return 1;

    std::vector<std::string> cases;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const std::string file = entry.path().filename().string();
        const std::string suffix = ".src.bin";
        if (file.size() > suffix.size() && file.compare(file.size() - suffix.size(), suffix.size(), suffix) == 0)
            cases.push_back(file.substr(0, file.size() - suffix.size()));
    }
    std::sort(cases.begin(), cases.end());
    check(cases.size() >= 5, "the cases are there (" + std::to_string(cases.size()) + ")");
    for (const std::string& name : cases) run_case(gpu, rehydrator, dir, name);

    // A structure without the dehydrated flag: status 1 (STATUS_NOT_DEHYDRATED), the destination untouched.
    {
        std::vector<std::uint8_t> plain(256, 0);
        const std::uint32_t flags = 0x80000000u, units = 3;
        std::memcpy(plain.data(), &flags, 4);
        std::memcpy(plain.data() + 4, &units, 4);
        Buffer s = upload(gpu, plain.data(), plain.size());
        Buffer d = make(gpu, 256);
        Buffer plan = make(gpu, std::size_t(XdxrRehydrator::plan_bytes(0)));
        const bool ran = gpu.run([&](VkCommandBuffer cb) {
            const bool ok = rehydrator.record(cb, s.address, d.address, plan.address, plan.buffer, 0, {});
            barrier_to_host(gpu, cb);
            return ok;
        });
        std::uint32_t st = 0;
        std::memcpy(&st, plan.map, 4);
        bool untouched = true;
        for (std::size_t i = 0; i < 256; ++i) untouched = untouched && static_cast<const std::uint8_t*>(d.map)[i] == 0xcd;
        check(ran && st == 1 && untouched, "a structure that is not dehydrated: status 1, the destination untouched");
        bool refused = true;
        gpu.run([&](VkCommandBuffer cb) {
            refused = !rehydrator.record(cb, s.address, s.address, plan.address, plan.buffer, 0, {});
            return true;
        });
        check(refused, "the same buffer as source and destination is refused");
    }

    std::printf("vk_xdxr_rehydrate_test: %d of %d checks failed\n", g_failures, g_checks);
    return g_failures ? 1 : 0;
}

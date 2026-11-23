// Unit tests for runtime/test/d3d12_test_shaders.h:
//   - RFC 1950/1951 multi-block stored deflate streams (0, 65,535, 65,536 and 140,000 bytes) unpacked by
//     leonida::zlib_inflate, leonida::fssm_program and leonida::ShaderLibrary;
//   - move-only RAII semantics of d3d12test::Ref<T> (no copy, single Release on move/self-move/vector growth);
//   - balanced FreeLibrary(d3d12) in d3d12test::serialize_root_signature;
//   - null-object and null-vtable guards in d3d12test::slot.
// Synthetic test inputs only; no game data.

#include "d3d12_test_shaders.h"
#include "../src/shader_library.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check(bool(expr), #expr, __LINE__)

std::vector<std::uint8_t> make_pattern(std::size_t size) {
    std::vector<std::uint8_t> out(size);
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = std::uint8_t((i * 131u + 17u) ^ (i >> 8) ^ (i >> 16));
    }
    return out;
}

void test_zlib_stored_boundaries() {
    // 0-byte payload: one final empty stored block + Adler-32 of empty data (0x00000001).
    {
        const std::vector<std::uint8_t> empty;
        const auto stream = d3d12test::zlib_stored(empty);
        const std::vector<std::uint8_t> expected = {0x78, 0x01, 0x01, 0x00, 0x00, 0xff,
                                                    0xff, 0x00, 0x00, 0x00, 0x01};
        CHECK(stream == expected);
        std::vector<std::uint8_t> out;
        CHECK(leonida::zlib_inflate(stream, out));
        CHECK(out.empty());
    }

    // 65,535-byte payload: exact maximum size of a single RFC 1951 stored block (BFINAL=1).
    {
        const auto payload = make_pattern(65535);
        const auto stream = d3d12test::zlib_stored(payload);
        CHECK(stream.size() == 2 + 5 + 65535 + 4);
        CHECK(stream[2] == 0x01);
        CHECK(stream[3] == 0xff && stream[4] == 0xff);
        CHECK(stream[5] == 0x00 && stream[6] == 0x00);
        std::vector<std::uint8_t> out;
        CHECK(leonida::zlib_inflate(stream, out));
        CHECK(out == payload);
    }

    // 65,536-byte payload: must split into two stored blocks (65,535 with BFINAL=0, then 1 with BFINAL=1).
    {
        const auto payload = make_pattern(65536);
        const auto stream = d3d12test::zlib_stored(payload);
        CHECK(stream.size() == 2 + 5 + 65535 + 5 + 1 + 4);
        CHECK(stream[2] == 0x00);
        CHECK(stream[3] == 0xff && stream[4] == 0xff);
        CHECK(stream[5] == 0x00 && stream[6] == 0x00);
        const std::size_t second_block = 2 + 5 + 65535;
        CHECK(stream[second_block] == 0x01);
        CHECK(stream[second_block + 1] == 0x01 && stream[second_block + 2] == 0x00);
        CHECK(stream[second_block + 3] == 0xfe && stream[second_block + 4] == 0xff);
        std::vector<std::uint8_t> out;
        CHECK(leonida::zlib_inflate(stream, out));
        CHECK(out == payload);

        std::vector<std::uint8_t> fssm64k = payload;
        fssm64k[0] = 'F';
        fssm64k[1] = 'S';
        fssm64k[2] = 'S';
        fssm64k[3] = 'M';
        const auto container = d3d12test::shader_container(fssm64k);
        CHECK(leonida::fssm_program(container) == fssm64k);
    }

    // 140,000-byte synthetic FSSM program (three stored blocks: 65,535 + 65,535 + 8,930), round-tripped through
    // shader_container, leonida::fssm_program, leonida::compute_program and leonida::ShaderLibrary.
    {
        std::vector<std::uint8_t> large_fssm = {'F', 'S', 'S', 'M'};
        large_fssm.reserve(140000);
        for (std::uint32_t i = 1; large_fssm.size() < 140000; ++i) {
            d3d12test::put32(large_fssm, 0xa5000000u ^ (i * 2654435761u));
        }
        CHECK(large_fssm.size() == 140000);

        const auto container = d3d12test::shader_container(large_fssm);
        const auto unpacked = leonida::fssm_program(container);
        CHECK(unpacked == large_fssm);

        const auto prog = leonida::compute_program(container);
        CHECK(prog.kind == leonida::ComputeProgram::Kind::fssm);
        CHECK(prog.bytes == large_fssm);

        const std::string expected_sha = leonida::sha256_hex(large_fssm);
        const std::filesystem::path temp_dir =
            std::filesystem::temp_directory_path() /
            ("leonida_d3d12_test_shaders_" + std::to_string(GetCurrentProcessId()));
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        CHECK(std::filesystem::create_directories(temp_dir, ec));

        {
            std::ofstream index(temp_dir / "index.json", std::ios::binary);
            index << "{\n"
                     "  \"synthetic_large_cs\": {\n"
                     "    \"stage\": \"cs\",\n"
                     "    \"fssm_sha256\": \""
                  << expected_sha
                  << "\"\n"
                     "  }\n"
                     "}\n";
        }

        leonida::ShaderLibrary lib;
        std::string err;
        CHECK(lib.load(temp_dir, &err));
        const leonida::ShaderEntry* entry = lib.find("cs", leonida::sha256_hex(unpacked));
        CHECK(entry != nullptr);
        if (entry) CHECK(entry->name == "synthetic_large_cs");
        const auto candidates = lib.program_candidates("cs", prog);
        CHECK(candidates.size() == 1 && candidates[0] == entry);

        std::filesystem::remove_all(temp_dir, ec);
    }
}

struct MockCom {
    int releases = 0;
    int value = 0;
    ULONG Release() noexcept { return ULONG(++releases); }
};

static_assert(!std::is_copy_constructible_v<d3d12test::Ref<MockCom>>);
static_assert(!std::is_copy_assignable_v<d3d12test::Ref<MockCom>>);
static_assert(std::is_nothrow_move_constructible_v<d3d12test::Ref<MockCom>>);
static_assert(std::is_nothrow_move_assignable_v<d3d12test::Ref<MockCom>>);

void test_ref_move_semantics() {
    // Default construction and reset on empty Ref.
    {
        d3d12test::Ref<MockCom> empty;
        CHECK(empty.p == nullptr);
        empty.reset();
        CHECK(empty.p == nullptr);
    }

    // Move construction transfers ownership and releases once.
    MockCom a{0, 42};
    {
        d3d12test::Ref<MockCom> first(&a);
        CHECK(first->value == 42);
        d3d12test::Ref<MockCom> second(std::move(first));
        CHECK(first.p == nullptr);
        CHECK(second.p == &a);
        CHECK(a.releases == 0);
    }
    CHECK(a.releases == 1);

    // Move assignment releases the previous target once, transfers the new target, and is self-move safe.
    MockCom b{0, 7}, c{0, 9};
    {
        d3d12test::Ref<MockCom> dst(&b);
        d3d12test::Ref<MockCom> src(&c);
        dst = std::move(src);
        CHECK(b.releases == 1);
        CHECK(c.releases == 0);
        CHECK(src.p == nullptr);
        CHECK(dst.p == &c);

        d3d12test::Ref<MockCom>& alias = dst;
        dst = std::move(alias);
        CHECK(dst.p == &c);
        CHECK(c.releases == 0);

        dst.reset();
        CHECK(dst.p == nullptr);
        CHECK(c.releases == 1);
        dst.reset();
        CHECK(c.releases == 1);
    }
    CHECK(b.releases == 1);
    CHECK(c.releases == 1);

    // std::vector<Ref<MockCom>> reallocation moves elements without prematurely releasing them.
    MockCom items[8]{};
    {
        std::vector<d3d12test::Ref<MockCom>> vec;
        for (auto& item : items) vec.emplace_back(&item);
        for (const auto& item : items) CHECK(item.releases == 0);
    }
    for (const auto& item : items) CHECK(item.releases == 1);
}

void test_serialize_root_signature() {
    const bool already_loaded = (GetModuleHandleA("d3d12.dll") != nullptr);

    // Invalid version fails cleanly and balances LoadLibraryExA with FreeLibrary.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC invalid_desc{};
    invalid_desc.Version = static_cast<D3D_ROOT_SIGNATURE_VERSION>(0xff);
    const auto empty_blob = d3d12test::serialize_root_signature(invalid_desc);
    CHECK(empty_blob.empty());
    if (!already_loaded) {
        CHECK(GetModuleHandleA("d3d12.dll") == nullptr);
    }

    // Probe whether the host system32 d3d12.dll exports D3D12SerializeVersionedRootSignature.
    bool has_serializer = false;
    if (const HMODULE probe = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) {
        has_serializer = (GetProcAddress(probe, "D3D12SerializeVersionedRootSignature") != nullptr);
        FreeLibrary(probe);
    }
    if (!already_loaded) {
        CHECK(GetModuleHandleA("d3d12.dll") == nullptr);
    }

    if (has_serializer) {
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        for (int i = 0; i < 3; ++i) {
            const auto blob = d3d12test::serialize_root_signature(desc);
            CHECK(blob.size() >= 4);
            if (blob.size() >= 4) {
                CHECK(std::memcmp(blob.data(), "DXBC", 4) == 0);
            }
            if (!already_loaded) {
                CHECK(GetModuleHandleA("d3d12.dll") == nullptr);
            }
        }
    }
}

void test_slot_guards() {
    CHECK(d3d12test::slot(nullptr, 0) == nullptr);
    CHECK(d3d12test::slot(nullptr, 3) == nullptr);

    struct FakeObject {
        void** vtable = nullptr;
    };
    FakeObject null_vtable{};
    CHECK(d3d12test::slot(&null_vtable, 0) == nullptr);
    CHECK(d3d12test::slot(&null_vtable, 2) == nullptr);

    int fn0 = 10, fn1 = 20, fn2 = 30;
    void* table[3] = {&fn0, &fn1, &fn2};
    FakeObject valid{table};
    CHECK(d3d12test::slot(&valid, 0) == &fn0);
    CHECK(d3d12test::slot(&valid, 1) == &fn1);
    CHECK(d3d12test::slot(&valid, 2) == &fn2);
}

void test_helper_structs() {
    const auto buf = d3d12test::buffer_desc(4096, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    CHECK(buf.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
    CHECK(buf.Width == 4096);
    CHECK(buf.Flags == D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    const auto bar = d3d12test::transition(nullptr, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    CHECK(bar.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);
    CHECK(bar.Transition.StateBefore == D3D12_RESOURCE_STATE_COPY_DEST);
    CHECK(bar.Transition.StateAfter == D3D12_RESOURCE_STATE_GENERIC_READ);
}

}  // namespace

int main() {
    test_zlib_stored_boundaries();
    test_ref_move_semantics();
    test_serialize_root_signature();
    test_slot_guards();
    test_helper_structs();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::puts("d3d12_test_shaders_test: all checks passed");
    return 0;
}

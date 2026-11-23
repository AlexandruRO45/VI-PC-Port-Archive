#include <cstdio>

#include "shims.h"

namespace {

int base_shim() { return 1; }
int exact_shim() { return 2; }

SHIM_TABLE(synthetic_apisets,
    {"api-ms-win-test-registry-l1-1-0.dll", "VersionedExport", reinterpret_cast<void*>(&base_shim)},
    {"api-ms-win-test-registry-l1-1-0.dll", "PreferExact", reinterpret_cast<void*>(&base_shim)},
    {"api-ms-win-test-registry-l1-1-3.dll", "PreferExact", reinterpret_cast<void*>(&exact_shim)},
    {"EXT-MS-WIN-TEST-REGISTRY-L2-10-12.DLL", "VersionedExport", reinterpret_cast<void*>(&base_shim)},
    {"api-ms-win-test-malformed-l1-1-x.dll", "VersionedExport", reinterpret_cast<void*>(&base_shim)},
    {"test-registry-l1-1-0.dll", "VersionedExport", reinterpret_cast<void*>(&base_shim)});

}  // namespace

int main() {
    using rt::shims::find;
    const auto base = reinterpret_cast<void*>(&base_shim);
    const auto exact = reinterpret_cast<void*>(&exact_shim);
    const struct {
        const char* dll;
        const char* name;
        void* expected;
        const char* description;
    } cases[] = {
        {"api-ms-win-test-registry-l1-1-0.dll", "VersionedExport", base, "base exact row"},
        {"api-ms-win-test-registry-l1-1-3.dll", "PreferExact", exact, "exact row wins"},
        {"api-ms-win-test-registry-l1-1-2.dll", "VersionedExport", base, "API-set version fallback"},
        {"API-MS-WIN-TEST-REGISTRY-L1-1-99.DLL", "VersionedExport", base, "case-insensitive DLL lookup"},
        {"ext-ms-win-test-registry-l2-10-0.dll", "VersionedExport", base, "ext-ms version fallback"},
        {"api-ms-win-test-registry-l1-2-0.dll", "VersionedExport", nullptr, "different minor version"},
        {"api-ms-win-test-registry-l2-1-0.dll", "VersionedExport", nullptr, "different major version"},
        {"api-ms-win-test-registry-l1-1-2.dll", "versionedexport", nullptr, "export names are case-sensitive"},
        {"api-ms-win-test-registry-l1-1-x.dll", "VersionedExport", nullptr, "malformed import version"},
        {"api-ms-win-test-malformed-l1-1-2.dll", "VersionedExport", nullptr, "malformed registered version"},
        {"test-registry-l1-1-2.dll", "VersionedExport", nullptr, "ordinary DLL name"},
    };

    int failures = 0;
    for (const auto& test : cases) {
        if (find(test.dll, test.name) != test.expected) {
            std::fprintf(stderr, "FAIL: %s (%s!%s)\n", test.description, test.dll, test.name);
            ++failures;
        }
    }
    std::printf("registry_apiset: %zu checks, %d failed\n", sizeof(cases) / sizeof(cases[0]), failures);
    return failures == 0 ? 0 : 1;
}

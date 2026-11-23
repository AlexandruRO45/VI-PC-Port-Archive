// CPU-only test of runtime/test/test_temp_dir.h: the scratch folder carries the process id and goes away with the
// object, and copy_file overwrites, reports a missing source without throwing, and keeps the bytes. No GPU.
#include "test_temp_dir.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::string slurp(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

void write(const std::filesystem::path& p, const std::string& text) {
    std::ofstream out(p, std::ios::binary);
    out << text;
}

}  // namespace

int main() {
    std::filesystem::path kept;
    {
        testfiles::ScratchDir dir("test_temp_dir_test");
        kept = dir.path();
        expect(std::filesystem::is_directory(kept), "the scratch folder exists while the object lives");
        expect(kept.filename().string().find("test_temp_dir_test_") == 0, "the folder name starts with the test name");

        const auto src = kept / "src.bin", dst = kept / "dst.bin";
        write(src, std::string("\x03\x02\x23\x07", 4) + "module bytes");
        expect(testfiles::copy_file(src, dst), "copy_file copies a new file");
        expect(slurp(dst) == slurp(src), "the copy has the source's bytes");

        write(src, "second version");
        expect(testfiles::copy_file(src, dst), "copy_file overwrites an existing destination");
        expect(slurp(dst) == "second version", "the overwrite replaced the bytes");

        std::fprintf(stderr, "(the next two lines are the expected messages for a missing source)\n");
        expect(!testfiles::copy_file(kept / "absent.bin", dst), "a missing source returns false instead of throwing");
        expect(slurp(dst) == "second version", "a failed copy leaves the destination alone");

        // Paths outside the ANSI code page: the copy works, and a missing source is reported, not thrown
        // (path::string() would throw "no mapping" on Windows while formatting the message).
        const auto unicode_dir = kept / L"m\u00f3dulos-\u4e2d\u6587";
        std::filesystem::create_directories(unicode_dir);
        const auto unicode_dst = unicode_dir / L"c\u00f3pia.bin";
        expect(testfiles::copy_file(src, unicode_dst), "copy_file copies to a path with non-ANSI characters");
        expect(slurp(unicode_dst) == "second version", "the Unicode-path copy has the bytes");
        expect(!testfiles::copy_file(unicode_dir / L"ausente-\u00e9.bin", unicode_dst),
               "a missing source with non-ANSI characters returns false instead of throwing");
        expect(slurp(unicode_dst) == "second version", "the failed Unicode-path copy leaves the destination alone");
    }
    expect(!std::filesystem::exists(kept), "the scratch folder is removed with the object");

    if (failures) {
        std::fprintf(stderr, "%d test_temp_dir test(s) failed\n", failures);
        return 1;
    }
    std::printf("test_temp_dir: all cases passed\n");
    return 0;
}

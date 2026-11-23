// Scratch folders for tests that write files (a translator-style shader library, an index) under the temp directory.
//
// The folder name carries the process id. Under one fixed name, every copy of a test running at the same time (ctest
// -j, or two suites sharing one Wine prefix or one TMPDIR) used the same folder, and one copy's remove_all deleted the
// modules another copy was still loading: with four copies of d3d12_compute at once, 6 of 40 runs failed at PSO
// creation or in copy_file, while 125 runs one at a time all passed (Apple M3 Max, MoltenVK 1.4.3 under Wine 11).
#pragma once

#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace testfiles {

// A path for a diagnostic, as UTF-8. path::string() converts to the ANSI code page on Windows and throws
// std::system_error (1113, "no mapping") for a character outside it, which would end the test in the very message
// that was meant to report the problem.
inline std::string utf8(const std::filesystem::path& p) {
    const auto u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}

// <temp>/<name>_<pid>: a folder no other running process uses.
inline std::filesystem::path process_temp_dir(std::string_view name) {
#ifdef _WIN32
    const long pid = long(_getpid());
#else
    const long pid = long(getpid());
#endif
    return std::filesystem::temp_directory_path() / (std::string(name) + "_" + std::to_string(pid));
}

// process_temp_dir(name), emptied and created on construction (a crashed run may have left the same name behind) and
// removed with its contents on destruction, so early returns don't leave one folder per failed run.
class ScratchDir {
public:
    explicit ScratchDir(std::string_view name) : path_(process_temp_dir(name)) {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        std::filesystem::create_directories(path_, ec);
        if (ec) std::fprintf(stderr, "cannot create %s: %s\n", utf8(path_).c_str(), ec.message().c_str());
    }
    ~ScratchDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    ScratchDir(const ScratchDir&) = delete;
    ScratchDir& operator=(const ScratchDir&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// Copies `from` (a file the build compiled into the test, LEONIDA_TEST_*_SPV) to `to`, overwriting it. Returns false
// and says why when it can't: std::filesystem::copy_file would throw instead, and the uncaught exception ends the
// test without a message (abort; exit code 3 under Wine), which reads as a crash rather than a missing file.
// On Windows the copy is CopyFileW: MSVC's std::filesystem::copy_file calls CopyFile2, which Wine 9 does not have, so
// the CI artifact run under Wine failed every test that stages a module this way (TEST-1126; the runtime's
// file_system shim made the same move in #3655).
inline bool copy_file(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::error_code ec;
#ifdef _WIN32
    if (!::CopyFileW(from.c_str(), to.c_str(), FALSE)) ec.assign(int(::GetLastError()), std::system_category());
#else
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
#endif
    if (ec) std::fprintf(stderr, "cannot copy %s to %s: %s\n", utf8(from).c_str(), utf8(to).c_str(),
                         ec.message().c_str());
    return !ec;
}

}  // namespace testfiles

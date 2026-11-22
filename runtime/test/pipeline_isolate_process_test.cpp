// Child supervision can be tested without a Vulkan loader or driver.
#include "pipeline_isolate_process.h"

#include <cstdio>
#include <string_view>

namespace {

int failures = 0;

void check(bool ok, const char* name) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    using leonida::pipeline_isolate::run_process;
    if (argc >= 2 && std::string_view(argv[1]) == "--echo") {
        for (int i = 2; i < argc; ++i) std::printf("[%s]\n", argv[i]);
        return 23;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--wait") {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        return 0;
    }

    const auto regular = run_process({argv[0], "--echo", "word with spaces", "x\\", "embedded \"quotes\"", ""}, 5.0);
    if (!regular.error.empty()) std::fprintf(stderr, "%s\n", regular.error.c_str());
    check(regular.error.empty() && regular.status == 23 && !regular.signal && !regular.timed_out,
          "capture an ordinary child exit code");
    std::string output;
    for (char c : regular.output)
        if (c != '\r') output.push_back(c);
    check(output == "[word with spaces]\n[x\\]\n[embedded \"quotes\"]\n[]\n",
          "preserve the child argument boundaries and capture stdout");

    const auto stopped = run_process({argv[0], "--wait"}, 0.25);
    if (!stopped.error.empty()) std::fprintf(stderr, "%s\n", stopped.error.c_str());
    check(stopped.error.empty() && stopped.timed_out && !stopped.interrupted, "enforce a child deadline");
    return failures != 0;
}

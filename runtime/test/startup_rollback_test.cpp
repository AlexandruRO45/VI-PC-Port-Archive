// Explicit rollback, reentry and destructor-only teardown use the same terminal lifecycle.
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include "../src/pe_loader.h"

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return 2;
    const bool expect_failure = argc == 2;
    const bool destructor_only = argc == 3 && std::strcmp(argv[2], "--destruct") == 0;
    rt::Loader loader(std::filesystem::path(argv[1]).parent_path());
    loader.load(argv[1]);
    bool failed = false;
    try {
        loader.initialise();
    } catch (const std::exception& error) {
        failed = true;
        std::printf("initialization failure: %s\n", error.what());
    }
    if (failed != expect_failure) return 3;
    if (destructor_only) return 0;
    loader.detach_initialised_modules();
    loader.detach_initialised_modules();
    try {
        loader.initialise();
        return 4;
    } catch (const std::exception& error) {
        if (std::strcmp(error.what(), "loader is detaching") != 0) return 5;
        std::puts("post-detach initialization refused");
    }
    bool handled = false;
    SetLastError(ERROR_SUCCESS);
    if (loader.load_runtime_module("startup_rollback_a.dll", 0, handled) ||
        !handled || GetLastError() != ERROR_DLL_INIT_FAILED) return 6;
    try {
        loader.load(argv[1]);
        return 7;
    } catch (const std::exception& error) {
        if (std::strcmp(error.what(), "loader is detaching") != 0) return 8;
    }
    return 0;
}
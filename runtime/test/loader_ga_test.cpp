// Checks for the loader's /GA TLS rewrite: the fixture EXE reads its static TLS at module index 0, which the loader
// rewrites to a TlsAlloc slot whose per-thread value is a view of the image's block. Every thread that runs the
// image's code needs that view: the loading thread, a thread that already existed at load(), and a thread started
// afterwards (which gets it from the helper DLL's DLL_THREAD_ATTACH, as the title's own threads do).
//
// The fixture is built from repository source like a Windows application (/GA); no game files are involved.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

#include "pe_loader.h"

namespace {
template <class Fn>
Fn find_export(const rt::Module& m, const char* name) {
    const auto it = m.exports.find(name);
    if (it == m.exports.end()) throw std::runtime_error(std::string("missing export ") + name);
    return reinterpret_cast<Fn>(it->second);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: loader_ga_test <tiny_ga_exe>\n");
        return 2;
    }
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };
    try {
        const std::filesystem::path exe = argv[1];

        // A thread that already runs when the image is mapped. Windows gives it the helper DLL's TLS block, but no
        // DLL_THREAD_ATTACH, so load() has to make its view itself.
        {
            rt::Module* mapped = nullptr;
            int(__cdecl* get)() = nullptr;
            void* view = reinterpret_cast<void*>(1);
            int value = -1;
            HANDLE go = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            std::thread early([&] {
                WaitForSingleObject(go, INFINITE);
                view = TlsGetValue(mapped->tls_ga_slot);
                if (view) value = get();
            });
            {
                rt::Loader loader(exe.parent_path());
                mapped = &loader.load(exe);
                get = find_export<int(__cdecl*)()>(*mapped, "tiny_ga_get");
                check(mapped->tls_ga_slot != rt::kNoTlsIndex, "load() rewrites the EXE's index-0 TLS reads");
                check(mapped->tls_ga_sites > 0, "load() counts the rewritten gs:[58h] sites");
                SetEvent(go);
                early.join();
            }
            check(view != nullptr, "a thread that existed at load() has the /GA view in its TlsAlloc slot");
            check(value == 7, "that thread reads the template through its view");
        }

        // The loading thread and a thread started after load(), as --run's entry-point thread and the title's threads
        // are. initialise() delivers the image's TLS callbacks, whose DLL_THREAD_ATTACH reads the template too.
        rt::Loader loader(exe.parent_path());
        rt::Module& m = loader.load(exe);
        const auto get = find_export<int(__cdecl*)()>(m, "tiny_ga_get");
        const auto set = find_export<void(__cdecl*)(int)>(m, "tiny_ga_set");
        const auto bad_attach = find_export<int(__cdecl*)()>(m, "tiny_ga_bad_attach_count");
        loader.initialise();
        check(get() == 7, "the loading thread reads the template through the view");
        int worker_before = -1, worker_after = -1;
        std::thread worker([&] {
            worker_before = get();
            set(42);
            worker_after = get();
        });
        worker.join();
        check(bad_attach() == 0, "the TLS callback read the template on the new thread");
        check(worker_before == 7, "a thread started after load() reads the template through its own view");
        check(worker_after == 42, "that thread's write stays in its own block");
        check(get() == 7, "the loading thread's value is unchanged");
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}

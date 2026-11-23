// Two shim tables serving one name (shims/registry.cpp): a different function is replaced by the later table and
// listed by conflicting_registrations(); the same function twice, in either case spelling of the DLL, is not a
// conflict. Within one file, tables are constructed in definition order, so `second` is the later table here.
#include <cstdio>
#include <string>
#include <vector>

#include "shims.h"

namespace {

int g_failed = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what);
    if (!ok) ++g_failed;
}

int first_fn() { return 1; }
int second_fn() { return 2; }
int shared_fn() { return 3; }
int only_fn() { return 4; }

void* fn(int (*f)()) { return reinterpret_cast<void*>(f); }

const rt::shims::Entry first_entries[] = {
    {"conflict_test.dll", "Overlap", fn(&first_fn)},
    {"conflict_test.dll", "Shared", fn(&shared_fn)},
    {"conflict_test.dll", "Only", fn(&only_fn)},
};
const rt::shims::Table first(first_entries, sizeof(first_entries) / sizeof(first_entries[0]));

const rt::shims::Entry second_entries[] = {
    {"conflict_test.dll", "Overlap", fn(&second_fn)},
    {"CONFLICT_TEST.dll", "Shared", fn(&shared_fn)},
};
const rt::shims::Table second(second_entries, sizeof(second_entries) / sizeof(second_entries[0]));

}  // namespace

int main() {
    const std::vector<std::string> conflicts = rt::shims::conflicting_registrations();
    check(conflicts.size() == 1 && conflicts[0] == "conflict_test.dll!Overlap",
          "exactly the name registered with a different function is listed");
    check(rt::shims::find("conflict_test.dll", "Overlap") == fn(&second_fn), "the later table's function is served");
    check(rt::shims::find("conflict_test.dll", "Shared") == fn(&shared_fn),
          "the same function from two tables (DLL in either case) is served and not listed");
    check(rt::shims::find("conflict_test.dll", "Only") == fn(&only_fn), "a name in one table only is unaffected");
    std::printf("%s\n", g_failed ? "FAILED" : "PASSED");
    return g_failed ? 1 : 0;
}

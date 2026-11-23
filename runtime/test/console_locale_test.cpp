#include <windows.h>

#include <cstdio>
#include <cwchar>
#include <initializer_list>

#include "shims.h"

namespace {
int checks = 0;
int failures = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

void check_locale(const char* dll, const wchar_t* expected, LANGID language) {
    for (auto name : {"GetUserDefaultLocaleName", "GetSystemDefaultLocaleName"}) {
        auto fn = reinterpret_cast<decltype(&GetUserDefaultLocaleName)>(rt::shims::find(dll, name));
        wchar_t locale[LOCALE_NAME_MAX_LENGTH]{};
        const int length = static_cast<int>(std::wcslen(expected)) + 1;
        check(fn && fn(locale, length) == length && std::wcscmp(locale, expected) == 0, name);
        check(fn && fn(nullptr, 0) == length, "locale size query");
        wchar_t small[1]{L'!'};
        check(fn && fn(small, 1) == 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER && small[0] == L'!',
              "undersized locale buffer");
    }
    for (auto name : {"GetUserDefaultUILanguage", "GetSystemDefaultUILanguage"}) {
        auto fn = reinterpret_cast<decltype(&GetUserDefaultUILanguage)>(rt::shims::find(dll, name));
        check(fn && fn() == language, name);
    }
}
}  // namespace

int main() {
    check(SetEnvironmentVariableW(L"LEONIDA_LOCALE", nullptr) != 0, "unset locale");
    check_locale("kernel32.dll", L"en-US", 0x0409);
    check(SetEnvironmentVariableW(L"LEONIDA_LOCALE", L"fr-CA") != 0, "set override");
    check_locale("kernel32.dll", L"fr-CA", 0x0c0c);
    check_locale("kernelbase.dll", L"fr-CA", 0x0c0c);
    check_locale("api-ms-win-core-localization-l1-1-0.dll", L"fr-CA", 0x0c0c);
    check_locale("api-ms-win-core-localization-l1-2-0.dll", L"fr-CA", 0x0c0c);
    check_locale("api-ms-win-core-localization-l1-2-1.dll", L"fr-CA", 0x0c0c);

    constexpr wchar_t sort_name[] = L"de-DE_phoneb";
    const LCID sort_lcid = LocaleNameToLCID(sort_name, LOCALE_ALLOW_NEUTRAL_NAMES);
    check(sort_lcid && sort_lcid != LOCALE_CUSTOM_DEFAULT && sort_lcid != LOCALE_CUSTOM_UNSPECIFIED,
          "Windows accepts the sort name");
    check(SetEnvironmentVariableW(L"LEONIDA_LOCALE", sort_name) != 0, "set sort name");
    check_locale("kernel32.dll", L"en-US", 0x0409);
    check(SetEnvironmentVariableW(L"LEONIDA_LOCALE", L"zz-ZZ") != 0, "set unsupported locale");
    check_locale("kernel32.dll", L"en-US", 0x0409);
    check(SetEnvironmentVariableW(L"LEONIDA_LOCALE", L"") != 0, "set empty locale");
    check_locale("kernel32.dll", L"en-US", 0x0409);
    std::printf("console_locale_test: %d/%d passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}

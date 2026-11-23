#pragma once

#include <windows.h>

#include <cstdio>
#include <cwchar>
#include <mutex>

namespace rt {

struct ConsoleLocale {
    wchar_t name[LOCALE_NAME_MAX_LENGTH] = L"en-US";
    LANGID language = MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US);
};

inline ConsoleLocale console_locale() {
    const DWORD saved_error = GetLastError();
    wchar_t name[LOCALE_NAME_MAX_LENGTH]{};
    SetLastError(ERROR_SUCCESS);
    const DWORD length = GetEnvironmentVariableW(L"LEONIDA_LOCALE", name, LOCALE_NAME_MAX_LENGTH);
    const bool unset = length == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND;
    ConsoleLocale locale;
    if (!unset) {
        // Windows also recognizes names with an underscore suffix for a sort order.
        // Restrict the override to BCP-47 characters before looking up its Windows language ID.
        constexpr wchar_t tag_chars[] = L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-";
        const LCID lcid = length && length < LOCALE_NAME_MAX_LENGTH && std::wcsspn(name, tag_chars) == length
            ? LocaleNameToLCID(name, LOCALE_ALLOW_NEUTRAL_NAMES) : 0;
        if (lcid && lcid != LOCALE_CUSTOM_DEFAULT && lcid != LOCALE_CUSTOM_UNSPECIFIED) {
            std::wmemcpy(locale.name, name, length + 1);
            locale.language = LANGIDFROMLCID(lcid);
        } else {
            static std::once_flag warned;
            std::call_once(warned, [] {
                std::fprintf(stderr, "[rt] invalid or unsupported LEONIDA_LOCALE; using en-US\n");
            });
        }
    }
    SetLastError(saved_error);
    return locale;
}

}  // namespace rt

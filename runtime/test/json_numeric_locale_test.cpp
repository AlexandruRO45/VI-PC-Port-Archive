// JsonReader must parse JSON decimals independently of the caller's numeric locale. No game data.
#include "../src/shader_library.h"

#include <clocale>
#include <cmath>
#include <cstdio>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

int main() {
    using namespace leonida;
    // Restore the caller's numeric locale even when CHECK returns early.
    struct RestoreLocale {
        std::string saved = std::setlocale(LC_NUMERIC, nullptr);
        ~RestoreLocale() { std::setlocale(LC_NUMERIC, saved.c_str()); }
    } restore;
    unsigned comma_locales = 0;
    for (const char* name : {"C", "French_France.1252", "German_Germany.1252", "fr_FR.UTF-8", "de_DE.UTF-8"}) {
        const char* selected = std::setlocale(LC_NUMERIC, name);
        if (!selected) continue;  // POSIX hosts need the non-C locales installed.
        const std::string current(selected);
        const std::string radix(std::localeconv()->decimal_point);
        if (radix == ",") ++comma_locales;
        struct Number { const char* text; double expected; };
        for (const auto& n : {Number{"1.5", 1.5}, {"-0.25", -0.25}, {"1.25e2", 125.0},
                              {"2.5E-1", 0.25}, {"42", 42.0}, {"0", 0.0}, {"-0.0", -0.0}}) {
            Json value;
            errno = ERANGE;  // A previous conversion failure must not poison the next number.
            CHECK(JsonReader(n.text).parse(value));
            CHECK(value.kind == Json::Kind::number && value.number == n.expected);
            CHECK(std::signbit(value.number) == std::signbit(n.expected));
        }
        Json nested;
        CHECK(JsonReader(R"({"values":[1.5,-0.25,{"exponent":1.25e2}],"enabled":true})").parse(nested));
        const Json* values = nested.find("values");
        CHECK(values && values->array.size() == 3 && values->array[0].number == 1.5 &&
              values->array[1].number == -0.25 && values->array[2].find("exponent") &&
              values->array[2].find("exponent")->number == 125.0);
        Json array;
        CHECK(JsonReader("[1,5]").parse(array) && array.array.size() == 2 &&
              array.array[0].number == 1 && array.array[1].number == 5);
        for (const char* invalid : {"1,5", "1.2.3", "1e+", "1e999", "{\"value\":1,5}"}) {
            Json value;
            CHECK(!JsonReader(invalid).parse(value));
        }
        CHECK(current == std::setlocale(LC_NUMERIC, nullptr));
        CHECK(radix == std::localeconv()->decimal_point);
        std::printf("JSON numeric locale: %s (radix %s) passed\n", current.c_str(), radix.c_str());
    }
    if (!comma_locales) {
#ifdef _WIN32
        // UCRT provides the named French/German locales. Missing coverage here is a test failure, including CI.
        std::fputs("No comma-decimal locale available; the JSON locale regression was not exercised\n", stderr);
        return 1;
#else
        // Minimal POSIX installs may provide only C. This is a separate CTest so the missing regression is
        // reported as skipped, while the remaining shader-library checks still run normally.
        std::puts("SKIP: install fr_FR.UTF-8 or de_DE.UTF-8 to exercise the JSON numeric-locale regression");
        return 77;
#endif
    }
    return 0;
}

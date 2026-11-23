// Unit tests for the shader_path.txt text normalization (runtime/src/shader_path_text.h). No Vulkan instance, no GPU
// and no file: the reader hands the file's bytes to a pure function, and this checks the shapes an editor writes.
#include "shader_path_text.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void expect_text(const char* what, const std::string& got, const std::string& want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s: got \"%s\", want \"%s\"\n", what, got.c_str(), want.c_str());
    ++failures;
}

}  // namespace

int main() {
    // The shapes a Windows editor writes. Notepad's "UTF-8" saves a BOM and CRLF, which is the report in RT-852.
    const std::string plain = R"(C:\Users\me\Leonida\work\shaders\spirv)";
    expect_text("plain path is byte-exact", leonida::normalize_shader_path_text(plain), plain);
    expect_text("CRLF is dropped", leonida::normalize_shader_path_text(plain + "\r\n"), plain);
    expect_text("LF is dropped", leonida::normalize_shader_path_text(plain + "\n"), plain);
    expect_text("trailing spaces and tabs are dropped", leonida::normalize_shader_path_text(plain + " \t\r\n"), plain);
    expect_text("a UTF-8 BOM is dropped", leonida::normalize_shader_path_text("\xEF\xBB\xBF" + plain), plain);
    expect_text("a BOM and CRLF together are dropped",
                leonida::normalize_shader_path_text("\xEF\xBB\xBF" + plain + "\r\n"), plain);

    // Inner bytes stay as they are: a space inside the path is part of it, and a UTF-8 name must not be re-encoded.
    const std::string spaced = R"(C:\Users\me\My Games\spirv)";
    expect_text("an inner space is kept", leonida::normalize_shader_path_text(spaced + "\r\n"), spaced);
    const std::string utf8 = "C:\\juego\\shaders\\año";
    expect_text("UTF-8 bytes are kept", leonida::normalize_shader_path_text(utf8 + "\n"), utf8);

    // What the reader tests next: an empty value is not a directory (the caller returns {} on empty).
    expect_text("only whitespace normalizes to empty", leonida::normalize_shader_path_text(" \r\n\t"), "");
    expect_text("an empty file stays empty", leonida::normalize_shader_path_text(""), "");
    // A BOM on its own is a path of zero bytes, not a directory named "\xEF\xBB\xBF".
    expect_text("a lone BOM normalizes to empty", leonida::normalize_shader_path_text("\xEF\xBB\xBF"), "");

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("shader_path_text: every case passed");
    return 0;
}

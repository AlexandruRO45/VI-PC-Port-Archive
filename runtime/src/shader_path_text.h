#pragma once

// The text of shader_path.txt (runtime/shims/d3d12_vulkan_device.cpp, configured_shader_directory). A Windows editor
// writes the file with a UTF-8 BOM and/or CRLF line endings, and either is enough for std::filesystem to look for a
// directory whose name carries those bytes: the shader library is not found and every pipeline fails. A UTF-8 BOM is
// also what a file saved as "UTF-8 with BOM" starts with, which is Notepad's default for a new text file.
//
// Drop a leading UTF-8 BOM and the trailing whitespace an editor adds (CR, LF, space, tab); no Windows path can end
// in one of those, so this cannot merge two different paths. The rest of the value stays byte-exact.

#include <string>

namespace leonida {

inline std::string normalize_shader_path_text(std::string text) {
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
        text.erase(0, 3);
    }
    while (!text.empty()) {
        const char last = text.back();
        if (last != '\r' && last != '\n' && last != ' ' && last != '\t') break;
        text.pop_back();
    }
    return text;
}

}  // namespace leonida

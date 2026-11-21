#include <windows.h>
#include <cstdint>

extern "C" __declspec(dllexport) std::uint32_t read_selector()
{
    BYTE input;
    DWORD count;
    if (!ReadFile(GetStdHandle(STD_INPUT_HANDLE), &input, 1, &count, nullptr) || count != 1)
        return UINT32_MAX;
    return input;
}

static bool write_bytes(DWORD stream, const void* data, DWORD size)
{
    DWORD written;
    return WriteFile(GetStdHandle(stream), data, size, &written, nullptr) && written == size;
}

extern "C" __declspec(dllexport) std::uint32_t emit_trace(const std::uint64_t* words)
{
    // The provider wrote its first two arguments and the result into the entry's buffer.
    return write_bytes(STD_ERROR_HANDLE, words, 16) && write_bytes(STD_OUTPUT_HANDLE, words + 2, 8);
}

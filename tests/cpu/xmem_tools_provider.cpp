#include <windows.h>

#ifndef TOOLS_AVAILABLE
#define TOOLS_AVAILABLE 0
#endif

namespace {
int available = TOOLS_AVAILABLE;
int recorded = 0;
int calls = 0;
}

extern "C" __declspec(dllexport) BOOL WINAPI XMemIsToolsMemAvailable()
{
    ++calls;
    return available ? TRUE : FALSE;
}

extern "C" __declspec(dllexport) int WINAPI xmem_tools_note(int value)
{
    recorded = value;
    return value;
}

extern "C" __declspec(dllexport) int WINAPI xmem_tools_note_value()
{
    return recorded;
}

extern "C" __declspec(dllexport) int WINAPI xmem_tools_calls()
{
    return calls;
}

extern "C" __declspec(dllexport) void WINAPI xmem_tools_reset()
{
    recorded = 0;
    calls = 0;
}
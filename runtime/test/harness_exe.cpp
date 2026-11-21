// Test harness image for the PE loader: imports one function from d3d12.dll and calls it from its entry point.
// Built with /ENTRY:HarnessEntry /NODEFAULTLIB so it has no CRT and only that one import; the loader binds it to the
// shim registered for d3d12.dll!LeonidaRenderTriangle.

extern "C" __declspec(dllimport) int __cdecl LeonidaPresentWindow(void);

extern "C" void HarnessEntry();
extern "C" void HarnessEntry() {
    LeonidaPresentWindow();
}

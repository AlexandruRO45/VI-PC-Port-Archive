// A data-only DLL for --run. Exporting a pointer gives the image a base relocation.
extern "C" __declspec(dllexport) const char* tiny_message = "entryless fixture";

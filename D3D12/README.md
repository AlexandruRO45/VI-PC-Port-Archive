# D3D12 Agility SDK runtime (Windows 10)

Put `D3D12Core.dll` from Microsoft's `Microsoft.Direct3D.D3D12` NuGet package
(`build\native\bin\x64\`) here and build: `victory.exe` then loads it instead of
the system's D3D12 runtime, which on Windows 10 lacks shader model 6.6. The
build copies the DLLs in this folder beside the executable. Git ignores them.

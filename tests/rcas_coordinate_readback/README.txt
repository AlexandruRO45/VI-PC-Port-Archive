Packed RCAS coordinate readback — reproduction
Xbox2Windows base: 9b966513bc4819420c4936db3691f239f0a8063c
Windows x64, LLVM 15.0.7 (clang.exe, llvm-objcopy.exe, LLVM-C.dll), Python 3, Windows SDK and MSVC.
In an x64 VS developer shell inside tests/rcas_coordinate_readback:
  cl /nologo /EHsc readback.cpp /link d3d11.lib d3dcompiler.lib
  python -B verify.py C:\path\to\Xbox2Windows C:\path\to\LLVM\bin
Each of 64 lanes starts at (lane%8,lane/8). Source texel ID = 8*y+x.
Reference uses signed CPU coordinates and zero for out-of-bounds reads.
Only the RCAS coordinate/load fragment is exercised; test bindings target D3D11 SM5.
The last case must fail the left-neighbor oracle.

Windows CI builds the D3D11 host and runs both controls using the synthetic
instruction encoding included alongside its assembly. Reproduce from the root:
  cmake --build --preset dev --target rcas_coordinate_readback
  ctest --preset dev --no-tests=error -R "^rcas_coordinate_(positive|negative)_warp$" -V
The optional LLVM command above independently assembles and decodes the input.

# Reports a configure that found no Python 3 interpreter. The D3D12 COM objects need it at build time (their method
# stubs are generated from the SDK's d3d12.h by tools/gen_d3d12_stubs.py), so without it they are left out, and with
# them the Vulkan D3D12 device in gta6rt (LEONIDA_D3D12=vulkan), d3d12_com_test and every test that links
# gta6rt_d3d12_com. It used to be one STATUS line, easy to miss: ctest then reports "100% tests passed" on a fraction
# of the suite. On Windows the usual cause is a `python` that is only the Microsoft Store alias. LEONIDA_REQUIRE_VULKAN,
# which asks for the Vulkan D3D12 device, makes it a configure error (CI and scripts/build_runtime.ps1 set it).
# Kept in its own file so tests/test_cmake_python_missing.py can run it with `cmake -P`.
function(leonida_python_missing)
  set(text
      "Python 3 not found, so this build leaves out the D3D12 COM objects (their stubs are generated from d3d12.h by "
      "tools/gen_d3d12_stubs.py), the Vulkan D3D12 device in gta6rt (LEONIDA_D3D12=vulkan is unavailable), "
      "d3d12_com_test and every D3D12 layer test that links them. ctest then runs a fraction of the suite and can "
      "still report 100% passed. On Windows, a `python` that is only the Microsoft Store alias (WindowsApps) is not "
      "an interpreter CMake can use, and Pythons that uv or pyenv manage are not on PATH: install Python 3.12+ from "
      "python.org with \"Add python.exe to PATH\", or pass -DPython3_EXECUTABLE=<path to python.exe>, then delete "
      "CMakeCache.txt, which keeps the failed lookup. -DLEONIDA_REQUIRE_VULKAN=ON turns this warning into an error.")
  string(CONCAT text ${text})
  if(LEONIDA_REQUIRE_VULKAN)
    message(FATAL_ERROR "${text}")
  else()
    message(WARNING "${text}")
  endif()
endfunction()

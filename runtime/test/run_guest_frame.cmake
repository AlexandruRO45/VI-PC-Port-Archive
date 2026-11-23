# Runs a multi-frame or multi-queue guest (tiny_guest_m3.exe, tiny_guest_m4.exe) under `gta6rt --run` (tests
# d3d12_m3_frame_guest, d3d12_m3_first_pass_guest and d3d12_m4_compute_guest in CMakeLists.txt). Like
# run_guest_draw.cmake, it passes only if gta6rt exits 0 AND the guest printed its marker line exactly once, which it
# does after every pixel or word check: a loader that returns 0 without running the guest to the end, or a guest that
# prints and then fails, both fail here. CTest's PASS_REGULAR_EXPRESSION would ignore the exit code, which is why the
# checks are not test properties.
#   cmake -DGTA6RT=<gta6rt.exe> -DGUEST=<guest.exe> -DFIXTURE=<build.py's output folder> -DMARKER=<regex of the line>
#         [-DGUEST_ARGS=<;-list of arguments after the folder>] [-DREQUIRE=<;-list of files under FIXTURE>]
#         [-DTIMEOUT=<s>] -P run_guest_frame.cmake
cmake_minimum_required(VERSION 3.24)

foreach(var GTA6RT GUEST FIXTURE MARKER)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "run_guest_frame.cmake: -D${var}=... is required")
  endif()
endforeach()
if(NOT DEFINED TIMEOUT)
  set(TIMEOUT 180)
endif()
foreach(path "${GTA6RT}" "${GUEST}" "${FIXTURE}/spirv/index.json")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "run_guest_frame.cmake: ${path} does not exist")
  endif()
endforeach()
foreach(name IN LISTS REQUIRE)
  if(NOT EXISTS "${FIXTURE}/${name}")
    message(FATAL_ERROR "run_guest_frame.cmake: ${FIXTURE}/${name} does not exist")
  endif()
endforeach()

file(TO_NATIVE_PATH "${GUEST}" guest)
file(TO_NATIVE_PATH "${FIXTURE}" fixture)
file(TO_NATIVE_PATH "${FIXTURE}/spirv" shaders)
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
# Set here rather than as the test's ENVIRONMENT, before starting gta6rt on native Windows.
set(ENV{LEONIDA_D3D12} vulkan)
set(ENV{LEONIDA_SHADERS} "${shaders}")

execute_process(
  COMMAND "${GTA6RT}" --run "${guest}" -- "${fixture}" ${GUEST_ARGS}
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT ${TIMEOUT}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
message("--- gta6rt stdout ---\n${out}\n--- gta6rt stderr ---\n${err}\n--- exit: ${result}")

string(REPLACE "\r\n" "\n" out "\n${out}\n")
string(REPLACE "\r\n" "\n" err "${err}")
set(failures)
if(NOT "${result}" STREQUAL "0")
  list(APPEND failures "exit is '${result}', not 0")
endif()
string(REGEX MATCHALL "\n${MARKER}\n" lines "${out}")
list(LENGTH lines line_count)
if(line_count EQUAL 0)
  list(APPEND failures "stdout has no line matching '${MARKER}'")
elseif(NOT line_count EQUAL 1)
  list(APPEND failures "stdout has ${line_count} lines matching '${MARKER}', not one")
endif()
# The device must be the shim's Vulkan device (shims/d3d12_xs.cpp logs it), not the host's D3D12 device.
set(pointer "(0[xX])?0*[1-9A-Fa-f][0-9A-Fa-f]*")
set(factory_line "\\[rt\\] d3d12_xs: D3D12XboxCreateDevice -> proxy ${pointer} \\(Vulkan device ${pointer}\\)")
if(NOT "\n${err}\n" MATCHES "\n${factory_line}\n")
  list(APPEND failures "stderr does not show D3D12XboxCreateDevice returning the Vulkan device")
endif()
list(LENGTH failures failure_count)
if(failure_count GREATER 0)
  list(JOIN failures "; " failures)
  message(FATAL_ERROR "${guest_name} under gta6rt --run failed: ${failures}")
endif()
message("${guest_name} under gta6rt --run: exit 0 and a line matching '${MARKER}'")

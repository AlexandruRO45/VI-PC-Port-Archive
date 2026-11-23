# Runs tiny_boot_trace_guest.exe under `gta6rt --run` with both first-boot traces (test d3d12_boot_trace_guest).
#   cmake -DGTA6RT=... -DGUEST=... -DPYTHON=... -DCHECK=... -DTOOL=... -DBOOT=... -DLOG=... -DSUMMARY=... -P ...
# CTest's PASS_REGULAR_EXPRESSION would ignore the exit code, so the exit, the guest's marker and both traces are
# checked here, including the same tools/boot_trace.py a first-boot triage uses. Trace files stay in the build tree.
cmake_minimum_required(VERSION 3.24)

foreach(var GTA6RT GUEST PYTHON CHECK TOOL BOOT LOG SUMMARY)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "run_boot_trace_guest.cmake: -D${var}=... is required")
  endif()
endforeach()
foreach(path "${GTA6RT}" "${GUEST}" "${PYTHON}" "${CHECK}" "${TOOL}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "run_boot_trace_guest.cmake: missing ${path}")
  endif()
endforeach()

file(REMOVE "${BOOT}" "${LOG}")
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
# The Vulkan backend is the only one that writes [d3d12-trace] lines (docs/boot.md).
set(ENV{LEONIDA_D3D12} vulkan)

execute_process(
  COMMAND "${GTA6RT}" --run "${GUEST}" --trace "${BOOT}" --trace-d3d12
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT 120
  RESULT_VARIABLE result
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
file(WRITE "${LOG}" "${err}")
message("--- gta6rt stdout ---\n${out}\n--- gta6rt stderr ---\n${err}\n--- exit: ${result}")

# GitHub's Windows runners have no Vulkan device. The guest then fails to create the device and the loader reports
# it on stderr; CTest skips on the marker (SKIP_REGULAR_EXPRESSION) instead of reporting a broken boot path.
string(FIND "${err}" "no Vulkan 1.3 GPU" no_vulkan)
string(FIND "${err}" "no GPU named by LEONIDA_VK_DEVICE" no_filtered)
if(result EQUAL 77 OR NOT no_vulkan EQUAL -1 OR NOT no_filtered EQUAL -1)
  message("SKIP: no Vulkan 1.3 device for d3d12 boot trace guest")
  return()
endif()

set(failures)
if(NOT "${result}" STREQUAL "0")
  list(APPEND failures "exit is '${result}', not 0")
endif()
string(REPLACE "\r\n" "\n" out "\n${out}\n")
set(marker "d3d12 boot guest: PASS, 10 trace calls")
string(FIND "${out}" "\n${marker}\n" first)
string(FIND "${out}" "\n${marker}\n" last REVERSE)
if(first EQUAL -1)
  list(APPEND failures "stdout has no line '${marker}'")
elseif(NOT first EQUAL last)
  list(APPEND failures "stdout has the line '${marker}' more than once")
endif()

execute_process(
  COMMAND "${PYTHON}" -I "${CHECK}" "${BOOT}" "${LOG}"
  RESULT_VARIABLE check_result
  OUTPUT_VARIABLE check_out
  ERROR_VARIABLE check_err)
message("--- boot trace check ---\n${check_out}${check_err}")
if(NOT "${check_result}" STREQUAL "0")
  list(APPEND failures "boot trace check exited '${check_result}'")
endif()

# The triage tool must summarize the very same two files, not a synthetic fixture.
execute_process(
  COMMAND "${PYTHON}" -I "${TOOL}" "${BOOT}" --d3d12 "${LOG}" --json
  OUTPUT_FILE "${SUMMARY}"
  RESULT_VARIABLE tool_result
  ERROR_VARIABLE tool_err)
message("--- boot_trace.py --json ---\n${tool_err}")
if(NOT "${tool_result}" STREQUAL "0")
  list(APPEND failures "boot_trace.py exited '${tool_result}'")
else()
  file(READ "${SUMMARY}" summary)
  foreach(pair "startup:exit:0" "d3d12:calls:13" "d3d12:last_seq:13")
    string(REPLACE ":" ";" parts "${pair}")
    list(GET parts 0 section)
    list(GET parts 1 key)
    list(GET parts 2 expected)
    set(json_err "")
    string(JSON actual ERROR_VARIABLE json_err GET "${summary}" "${section}" "${key}")
    if(json_err OR NOT "${actual}" STREQUAL "${expected}")
      list(APPEND failures "summary ${section}.${key} is '${actual}', not '${expected}'")
    endif()
  endforeach()
endif()

list(LENGTH failures failure_count)
if(failure_count GREATER 0)
  list(JOIN failures "; " failures)
  message(FATAL_ERROR "d3d12_boot_trace_guest failed: ${failures}")
endif()
message("d3d12_boot_trace_guest: exit 0, marker line and both traces verified")

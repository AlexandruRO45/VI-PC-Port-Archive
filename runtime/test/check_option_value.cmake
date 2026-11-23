# The value of --trace, --game-dir, --overlay and --mods is a path: a following option is a missing value, which is the
# usage error (exit 2), never a value to use. `gta6rt --dry-run image.exe --trace --trace-d3d12` used to take
# --trace-d3d12 as the trace file name and exit 0, so a mistyped command line read as a clean run with a file named
# --trace-d3d12 and the D3D12 trace off. The error comes before the image is opened, so a text file stands in for it.
set(root "${CMAKE_CURRENT_BINARY_DIR}/option_value")
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}")
file(WRITE "${root}/image.exe" "not a PE image")
set(failures "")
macro(check_refused)
  execute_process(
    COMMAND ${RUNNER} "${LOADER}" "${root}/image.exe" ${ARGN}
    WORKING_DIRECTORY "${root}"
    TIMEOUT 60
    RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
  string(REPLACE "\r\n" "\n" error "${error}")
  if(NOT code STREQUAL "2" OR NOT error MATCHES "(^|\n)usage: gta6rt"
      OR output MATCHES "module\\(s\\) mapped|running ")
    string(APPEND failures "--- ${ARGN}: returned ${code}\n${output}${error}\n")
  endif()
endmacro()
check_refused(--dry-run --trace --trace-d3d12)
check_refused(--dry-run --game-dir --trace)
check_refused(--dry-run --overlay --mods)
check_refused(--dry-run --mods --overlay)
# A real value is still taken: --trace FILE creates that file (it is opened before the image is read) and the run goes on
# to fail on the image with exit 1, not with the usage.
execute_process(
  COMMAND ${RUNNER} "${LOADER}" "${root}/image.exe" --dry-run --trace "${root}/trace.jsonl"
  WORKING_DIRECTORY "${root}"
  TIMEOUT 60
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
string(REPLACE "\r\n" "\n" error "${error}")
if(NOT code STREQUAL "1" OR error MATCHES "usage: gta6rt" OR NOT EXISTS "${root}/trace.jsonl")
  string(APPEND failures "--- --trace FILE was not taken as the trace file: returned ${code}\n${output}${error}\n")
endif()
if(EXISTS "${root}/--trace-d3d12")
  string(APPEND failures "--- the refused value was used as the trace file name\n")
endif()
if(failures)
  message(FATAL_ERROR "gta6rt used a following option as an option value\n${failures}")
endif()

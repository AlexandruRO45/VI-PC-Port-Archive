# Initialise tiny_reent.dll under the loader (gta6rt --init) and check its verdict from outside, as
# check_loaded_gdk.cmake does: the fixture's DllMain calls LoadLibrary on itself and on a second bundled DLL, and
# ends the process with exit code 42 if its DLL_PROCESS_ATTACH is entered a second time.
# Regular expressions here use [.] for a literal dot: "\." is not an escape CMake accepts in a quoted argument.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --init "${guest_name}"
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT 90
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output)
string(REPLACE "\r\n" "\n" output "${output}")
if(NOT code STREQUAL "0" OR NOT output MATCHES "(^|\n)tiny_reent: PASS(\n|$)" OR output MATCHES "tiny_reent: FAIL")
  message(FATAL_ERROR "DllMain re-entry fixture returned ${code}\n${output}")
endif()

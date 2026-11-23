# Check both the process result and the guest verdict. CTest's PASS_REGULAR_EXPRESSION
# alone can pass a process that prints the marker and then exits nonzero.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --run "${guest_name}" -- hello 42
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT 90
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
string(REPLACE "\r\n" "\n" output "${output}")
if(NOT code STREQUAL "0" OR NOT output MATCHES "(^|\n)tiny_game: OK(\n|$)"
    OR NOT output MATCHES "(^|\n)game exited with code 0(\n|$)")
  message(FATAL_ERROR "Synthetic guest did not finish successfully (loader exit ${code})\n${output}\n${error}")
endif()

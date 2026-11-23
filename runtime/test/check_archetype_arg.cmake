# The value of --archetype-lookup is RVA[,ARG], and ARG is 0 (the hash is the first argument) or 1 (the second). Any
# other ARG used to be read as 0: `--archetype-lookup 1A2B3C0,2` hooked argument 0 and exited 0, so a typo in a mod's
# command line read as a clean hook of the wrong call sites. It is the usage error now, before the image is opened, so a
# text file stands in for the image.
set(root "${CMAKE_CURRENT_BINARY_DIR}/archetype_arg")
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}")
file(WRITE "${root}/image.exe" "not a PE image")
set(failures "")
macro(check_refused)
  execute_process(
    COMMAND ${RUNNER} "${LOADER}" --dry-run "${root}/image.exe" --archetype-lookup ${ARGN}
    WORKING_DIRECTORY "${root}"
    TIMEOUT 60
    RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
  string(REPLACE "\r\n" "\n" error "${error}")
  if(NOT code STREQUAL "2" OR NOT error MATCHES "(^|\n)usage: gta6rt"
      OR error MATCHES "archetype lookup at RVA" OR output MATCHES "module\\(s\\) mapped")
    string(APPEND failures "--- --archetype-lookup ${ARGN}: returned ${code}\n${output}${error}\n")
  endif()
endmacro()
check_refused(1A2B3C0,2)
check_refused(1A2B3C0,1x)
check_refused(1A2B3C0,)
# A real ARG (or none) is still taken, and the run goes on to fail on the image with exit 1.
macro(check_taken)
  execute_process(
    COMMAND ${RUNNER} "${LOADER}" --dry-run "${root}/image.exe" --archetype-lookup ${ARGN}
    WORKING_DIRECTORY "${root}"
    TIMEOUT 60
    RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
  string(REPLACE "\r\n" "\n" error "${error}")
  if(code STREQUAL "2" OR error MATCHES "usage: gta6rt")
    string(APPEND failures "--- --archetype-lookup ${ARGN} was refused: returned ${code}\n${output}${error}\n")
  endif()
endmacro()
check_taken(1A2B3C0)
check_taken(1A2B3C0,0)
check_taken(1A2B3C0,1)
if(failures)
  message(FATAL_ERROR "gta6rt did not handle the --archetype-lookup ARG as it should\n${failures}")
endif()

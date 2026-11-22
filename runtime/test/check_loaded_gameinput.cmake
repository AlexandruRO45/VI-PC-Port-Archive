# Run the ordinal-import GameInput fixture (tiny_gameinput) under the loader and check from outside, as
# check_loaded_gdk.cmake does, that it passed and that the loader's report binds GameInput.dll ordinal 1 to the shim.
# The binding is read from the report because a host GameInput.dll answers the fixture's calls too.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --run "${guest_name}"
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT 90
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output)
string(REPLACE "\r\n" "\n" output "${output}")
if(NOT code STREQUAL "0" OR NOT output MATCHES "(^|\n)tiny_gameinput: PASS(\n|$)"
    OR NOT output MATCHES "(^|\n)game exited with code 0(\n|$)")
  message(FATAL_ERROR "Loaded GameInput ordinal fixture returned ${code}\n${output}")
endif()
if(NOT output MATCHES "(^|\n) +shim +GameInput[.]dll!#1(\n|$)")
  message(FATAL_ERROR "GameInput.dll ordinal 1 was not bound to the shim\n${output}")
endif()

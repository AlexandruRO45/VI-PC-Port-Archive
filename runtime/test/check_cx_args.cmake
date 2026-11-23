# GetCmdArguments (vccorlib140.dll) gives the mapped game's arguments by name and by ordinal, and serving that one
# name does not make the shims claim the whole DLL. STUB is the stand-in host vccorlib140.dll (cx_vccorlib_stub),
# passed to the guest as a Windows path.
file(TO_NATIVE_PATH "${STUB}" stub)
if(RUNNER)  # Wine: the guest sees the Linux root as Z:
  string(REPLACE "/" "\\" stub "Z:${STUB}")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env "LEONIDA_TEST_CX_STUB=${stub}" ${RUNNER} "${LOADER}" --run "${GUEST}" -- hello "two words"
  TIMEOUT 90
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT code STREQUAL "0" OR NOT output MATCHES "tiny_cx_args: PASS" OR NOT output MATCHES "game exited with code 0")
  message(FATAL_ERROR "GetCmdArguments run returned ${code}\n${output}\n${error}")
endif()
if(error MATCHES "vccorlib140.dll is served by the shims")
  message(FATAL_ERROR "vccorlib140.dll was claimed by the shims at run time\n${output}\n${error}")
endif()

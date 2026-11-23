file(MAKE_DIRECTORY "${TEST_DIR}")
file(WRITE "${TEST_DIR}/empty.spv" "")

foreach(shader missing.spv empty.spv)
  execute_process(COMMAND ${RUNNER} "${TOOL}" "${shader}" missing-fragment.spv out.raw 0 92 c00000 14500000 0
                  WORKING_DIRECTORY "${TEST_DIR}"
                  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 10)
  string(FIND "${log}" "failed to read shader module: ${shader}" diagnostic)
  if(NOT code STREQUAL "2" OR diagnostic EQUAL -1 OR log MATCHES "device: ")
    message(FATAL_ERROR "${shader}: expected a shader input error before Vulkan setup, got ${code}\n${log}")
  endif()
endforeach()

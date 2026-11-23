# Argument errors need neither a shader nor a Vulkan device.
function(expect_usage)
  execute_process(COMMAND ${RUNNER} "${TOOL}" "unused-for-cli-test.spv" ${ARGN}
                  RESULT_VARIABLE result ERROR_VARIABLE error TIMEOUT 15)
  if(NOT result STREQUAL "2" OR NOT error MATCHES "^usage: vk_store_test ")
    message(FATAL_ERROR "Expected usage and exit 2 for [${ARGN}], got [${result}]: ${error}")
  endif()
endfunction()

expect_usage(laod)
expect_usage(load extra)

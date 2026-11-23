file(MAKE_DIRECTORY "${TEST_DIR}")
file(WRITE "${TEST_DIR}/empty.spv" "")
file(WRITE "${TEST_DIR}/short.spv" "1234")
file(WRITE "${TEST_DIR}/unaligned.spv" "123456789012345678901")

function(check_shader_input name reason)
  set(path "${TEST_DIR}/${name}")
  execute_process(COMMAND ${RUNNER} "${TOOL}" "${path}" "${TEST_DIR}/unused.spv" "${TEST_DIR}/unused.raw"
                  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 15)
  if(NOT code STREQUAL "1" OR NOT error STREQUAL "${reason} '${path}'\n" OR NOT output STREQUAL "")
    message(FATAL_ERROR "${name}: expected a shader input error, got ${code}\n${output}${error}")
  endif()
endfunction()

check_shader_input(missing.spv "cannot open SPIR-V file")
foreach(name empty.spv short.spv unaligned.spv)
  check_shader_input("${name}" "invalid SPIR-V file size for")
endforeach()

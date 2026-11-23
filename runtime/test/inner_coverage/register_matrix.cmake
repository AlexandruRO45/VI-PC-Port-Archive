# Shared by the normal runtime build and the standalone fixture build.
option(LEONIDA_TEST_INNER_COVERAGE_MATRIX "Register the synthetic GPU mask/capture matrix (requires validation)" OFF)
function(leonida_register_inner_coverage_matrix)
  if(NOT BUILD_TESTING OR NOT LEONIDA_TEST_INNER_COVERAGE_MATRIX)
    return()
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_program(LEONIDA_INNER_GLSLANG NAMES glslangValidator glslang REQUIRED)
  find_program(LEONIDA_INNER_SPIRV_AS NAMES spirv-as REQUIRED)
  find_program(LEONIDA_INNER_SPIRV_VAL NAMES spirv-val REQUIRED)
  add_test(NAME inner_coverage_matrix
    COMMAND "${Python3_EXECUTABLE}" -I "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run.py"
      --harness "$<TARGET_FILE:vk_inner_coverage>"
      --glslang "${LEONIDA_INNER_GLSLANG}"
      --spirv-as "${LEONIDA_INNER_SPIRV_AS}"
      --spirv-val "${LEONIDA_INNER_SPIRV_VAL}")
  set_tests_properties(inner_coverage_matrix PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 600 LABELS "gpu;conservative;inner_coverage")
endfunction()
leonida_register_inner_coverage_matrix()

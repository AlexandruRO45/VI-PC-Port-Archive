# The driver may build the pipeline even when binding 20 is missing. Check the validation output too.
if(NOT "$ENV{LEONIDA_VK_VALIDATION}" STREQUAL "1")
  message("SKIP: set LEONIDA_VK_VALIDATION=1 on a machine with Vulkan validation")
  return()
endif()

file(MAKE_DIRECTORY "${TEST_DIR}")
file(WRITE "${TEST_DIR}/fetch.comp" [=[#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 1) in;
layout(set = 0, binding = 20) uniform texture2DMS src;
layout(set = 0, binding = 2, std430) buffer Output { vec4 value; } dst;
void main() { dst.value = texelFetch(src, ivec2(0), 0); }
]=])
execute_process(COMMAND "${GLSLANG}" -V --target-env vulkan1.1 -o "${TEST_DIR}/fetch.spv" "${TEST_DIR}/fetch.comp"
                COMMAND_ERROR_IS_FATAL ANY)
file(WRITE "${TEST_DIR}/index.json" [=[{"fetch":{"stage":"cs","bindings":[2,20],"push_constant_bytes":16}}]=])

set(ENV{VK_LOADER_DEBUG} "layer")
execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}" --compute-only --jobs 1
                RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 60)
if(NOT code STREQUAL "0" OR NOT log MATCHES "cs pipelines: 1 ok, 0 failed, 0 skipped" OR
   log MATCHES "VUID-|Validation Error" OR NOT log MATCHES "Insert instance layer \"VK_LAYER_KHRONOS_validation\"")
  message(FATAL_ERROR "Expected one pipeline with active Vulkan validation and no errors (exit ${code})\n${log}")
endif()

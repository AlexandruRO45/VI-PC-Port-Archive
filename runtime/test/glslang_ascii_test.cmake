# RT-518: cmake/glslang_ascii.cmake compiles a shader whose source and output folders have names outside the ASCII
# range (Cyrillic and CJK, which no single ANSI code page spells) and reports a failing compile.
#
#   cmake -DGLSLANG=<glslangValidator> -DSCRIPT=<glslang_ascii.cmake> -DSOURCE=<a .comp shader>
#         -DWORK=<scratch folder> -DSTAGING=<ASCII-only folder> -P glslang_ascii_test.cmake
cmake_minimum_required(VERSION 3.24)

foreach(variable GLSLANG SCRIPT SOURCE WORK STAGING)
  if(NOT ${variable})
    message(FATAL_ERROR "glslang_ascii_test: ${variable} is not set")
  endif()
endforeach()

set(folder "${WORK}/Алексей 用户")
file(REMOVE_RECURSE "${WORK}" "${STAGING}")
file(MAKE_DIRECTORY "${folder}")
get_filename_component(name "${SOURCE}" NAME)
file(COPY_FILE "${SOURCE}" "${folder}/${name}")
set(output "${folder}/вывод/${name}.spv")

execute_process(COMMAND "${CMAKE_COMMAND}" "-DLEONIDA_GLSLANG_STAGING_DIR=${STAGING}" -P "${SCRIPT}" --
                        "${GLSLANG}" -V --target-env vulkan1.3 -o "${output}" "${folder}/${name}"
                RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "FAIL: compiling ${folder}/${name} returned ${result}")
endif()
if(NOT EXISTS "${output}")
  message(FATAL_ERROR "FAIL: ${output} was not written")
endif()
file(READ "${output}" magic LIMIT 4 HEX)
if(NOT magic STREQUAL "03022307")
  message(FATAL_ERROR "FAIL: ${output} does not start with the SPIR-V magic number (read ${magic})")
endif()

file(WRITE "${folder}/broken.comp" "#version 450\nvoid main( {\n")
execute_process(COMMAND "${CMAKE_COMMAND}" "-DLEONIDA_GLSLANG_STAGING_DIR=${STAGING}" -P "${SCRIPT}" --
                        "${GLSLANG}" -V --target-env vulkan1.3 -o "${folder}/broken.spv" "${folder}/broken.comp"
                RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
if(result EQUAL 0 OR EXISTS "${folder}/broken.spv")
  message(FATAL_ERROR "FAIL: a shader that does not compile was reported as compiled")
endif()

file(GLOB leftovers "${STAGING}/*")
if(leftovers)
  message(FATAL_ERROR "FAIL: the staging folder keeps ${leftovers}")
endif()
file(REMOVE_RECURSE "${WORK}" "${STAGING}")
message(STATUS "PASS: compiled through ${STAGING} and copied to ${output}")

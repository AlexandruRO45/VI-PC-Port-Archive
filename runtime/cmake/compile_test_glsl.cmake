# All runtime GLSL fixtures are standalone sources. Additional arguments are glslang flags
# (for example, two descriptor tests compile the same source with different defines).
function(compile_test_glsl output source target_env)
  get_filename_component(output_dir "${output}" DIRECTORY)
  get_filename_component(shader "${source}" NAME)
  set(flags -V --target-env "${target_env}" ${ARGN})
  if(CMAKE_HOST_WIN32 AND "${source};${output};${CMAKE_CURRENT_BINARY_DIR}" MATCHES "[^ -~]")
    # The SDK compiler uses narrow filenames and also narrows its working directory.
    if(NOT TARGET Python3::Interpreter)
      find_package(Python3 REQUIRED COMPONENTS Interpreter)
    endif()
    set(helper "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compile_test_glsl.py")
    set(command "$<TARGET_FILE:Python3::Interpreter>" -I "${helper}" "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}"
                "${source}" "${output}" ${flags})
  else()
    set(helper "")
    set(command "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}" ${flags} -o "${output}" "${source}")
  endif()
  add_custom_command(OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
    COMMAND ${command}
    DEPENDS "${source}" ${helper}
    COMMENT "Compiling ${shader}" VERBATIM)
endfunction()

set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/offscreen_invalid_operation")
set(invalid_operation "--bvh-wave")
set(unexpected_image "${work_dir}/${invalid_operation}")
file(MAKE_DIRECTORY "${work_dir}")
file(REMOVE "${unexpected_image}")

execute_process(COMMAND "${RENDERER}" warp "${invalid_operation}"
    WORKING_DIRECTORY "${work_dir}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)

if(EXISTS "${unexpected_image}")
    file(REMOVE "${unexpected_image}")
    message(FATAL_ERROR "Invalid operation was treated as an output path (exit ${result})")
endif()
if(NOT result STREQUAL "2" OR NOT error MATCHES "Unknown GPU operation or invalid arguments: --bvh-wave")
    message(FATAL_ERROR "Expected an invalid-argument error, got exit ${result}\n${output}${error}")
endif()

execute_process(COMMAND "${RENDERER}" warp --check-pixel dummy.cso invalid
    WORKING_DIRECTORY "${work_dir}"
    RESULT_VARIABLE bad_flag_result
    OUTPUT_VARIABLE bad_flag_output
    ERROR_VARIABLE bad_flag_error)
if(NOT bad_flag_result STREQUAL "2" OR NOT bad_flag_error MATCHES "Unknown GPU operation or invalid arguments: --check-pixel")
    message(FATAL_ERROR "Expected --check-pixel invalid-flag error, got exit ${bad_flag_result}\n${bad_flag_output}${bad_flag_error}")
endif()

set(unexpected_draw "${work_dir}/unexpected_draw.bmp")
file(REMOVE "${unexpected_draw}")
execute_process(COMMAND "${RENDERER}" warp unexpected_draw.bmp extra_arg
    WORKING_DIRECTORY "${work_dir}"
    RESULT_VARIABLE bad_draw_result
    OUTPUT_VARIABLE bad_draw_output
    ERROR_VARIABLE bad_draw_error)
if(EXISTS "${unexpected_draw}")
    file(REMOVE "${unexpected_draw}")
    message(FATAL_ERROR "Invalid positional draw argc created an output image")
endif()
if(NOT bad_draw_result STREQUAL "2")
    message(FATAL_ERROR "Expected exit 2 for invalid positional draw argc, got ${bad_draw_result}\n${bad_draw_output}${bad_draw_error}")
endif()

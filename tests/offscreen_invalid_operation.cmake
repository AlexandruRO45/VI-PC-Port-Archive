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

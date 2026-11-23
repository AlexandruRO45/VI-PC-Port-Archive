function(check_cli case expected)
  # The invalid environment value stops well-formed calls before loading Vulkan.
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=VKH_DRAW_MESH VKH_SGPRS=invalid
            ${RUNNER} "${TOOL}" vs.spv fs.spv out.raw ${ARGN}
    RESULT_VARIABLE result
    ERROR_VARIABLE error
    TIMEOUT 5
  )
  if(NOT result STREQUAL "2" OR NOT error MATCHES "(^|\n)${expected}")
    message(FATAL_ERROR "${case}: expected exit 2 and '${expected}', got ${result}: ${error}")
  endif()
endfunction()

set(extent_error "width and height must be decimal integers")
check_cli(zero "${extent_error}" 0 64)
check_cli(negative "${extent_error}" 64 -1)
check_cli(width_overflow "${extent_error}" 4294967296 64)
check_cli(height_overflow "${extent_error}" 64 4294967296)
check_cli(width_junk "${extent_error}" 64px 64)
check_cli(height_junk "${extent_error}" 64 64px)
check_cli(hex "${extent_error}" 0x40 64)
set(arity_error "expected width and height, optionally followed by two clip shifts")
check_cli(width_only "${arity_error}" 64)
check_cli(one_clip_shift "${arity_error}" 64 64 0)
check_cli(extra_argument "${arity_error}" 64 64 0 0 extra)
check_cli(incomplete_mesh_grid "--mesh-group-readback requires" --mesh-group-readback 2 3)

check_cli(default_viewport "VKH_SGPRS:")
check_cli(rectangular_viewport "VKH_SGPRS:" 64 128)
check_cli(clip_pair "VKH_SGPRS:" 64 64 -1.0 0)
check_cli(mesh_grid "VKH_SGPRS:" --mesh-group-readback 2 3 4)

# Argument errors must be diagnosed before opening shaders or starting Vulkan.
function(expect_invalid diagnostic)
  execute_process(COMMAND ${RUNNER} "${TOOL}" unused.vert unused.frag unused.raw ${ARGN}
                  RESULT_VARIABLE code ERROR_VARIABLE error TIMEOUT 10)
  if(NOT code STREQUAL "2" OR NOT error MATCHES "${diagnostic}")
    message(FATAL_ERROR "${ARGN}: expected '${diagnostic}' and exit 2, got ${code}: ${error}")
  endif()
endfunction()

expect_invalid("s0 must" typo 0 0 0)
expect_invalid("s1 must" 0 0x 0 0)
expect_invalid("s2 must" 0 0 ffg 0)
expect_invalid("s3 must" 0 0 0 100000000)
expect_invalid("s0 must" -1 0 0 0)
foreach(range "16;17" "1;0" "0;2" "-1;0" "0;1x" "4294967296;1")
  expect_invalid("decimal mip levels" 0x0 FFFFFFFF 0X0 0 ${range})
endforeach()

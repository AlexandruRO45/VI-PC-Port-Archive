# Runs two copies of one test program at the same time and fails unless both exit 0 (77 from both is a skip).
#
#   cmake -DTEST=<program> -P run_two_copies.cmake
#
# A test that wrote to one fixed folder under the temp directory raced with a second copy of itself (ctest -j from two
# checkouts, or two suites sharing one TMPDIR or Wine prefix): one copy's remove_all deleted the files the other was
# still using. The commands of one execute_process call run concurrently, so this starts both copies together. The
# first copy's output is piped into the second copy's standard input, which the tests do not read.
if(NOT TEST)
  message(FATAL_ERROR "run_two_copies.cmake: pass -DTEST=<program>")
endif()

execute_process(
  COMMAND "${TEST}"
  COMMAND "${TEST}"
  RESULTS_VARIABLE results
  OUTPUT_VARIABLE output
  ERROR_VARIABLE errors)

message("${output}")
message("${errors}")
if(results STREQUAL "77;77")
  message("SKIP: both copies skipped")
  return()
endif()
if(NOT results STREQUAL "0;0")
  message(FATAL_ERROR "two copies at once: exit codes ${results} (both must be 0)")
endif()
message("two copies at once: both passed")

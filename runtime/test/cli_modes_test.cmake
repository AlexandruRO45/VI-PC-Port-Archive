# gta6rt must accept exactly one of --dry-run, --init and --run. Combining modes used to pass validation and then
# run the executable's entry point even with --dry-run, which promises that no image code runs.
#
# usage() returns 2. The image path need not exist: the mode check runs before the image is mapped.
function(expect_usage label)
  execute_process(COMMAND "${GTA6RT}" ${ARGN} RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
  if(NOT rc EQUAL 2)
    message(FATAL_ERROR "cli_modes: ${label} returned ${rc}, expected 2")
  endif()
endfunction()

expect_usage("--dry-run --run" --dry-run --run x.exe)
expect_usage("--dry-run --init" --dry-run --init x.exe)
expect_usage("--init --run" --init --run x.exe)
expect_usage("no mode" x.exe)
expect_usage("no image" --run)

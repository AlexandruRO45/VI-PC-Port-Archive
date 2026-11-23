# A command line that names no image has to be the usage error: exit 2 and "usage: gta6rt", nothing mapped. Two shapes
# reach the image argument in the parser: a value-taking option with its value missing at the end (--dry-run --game-dir)
# and an option this parser does not know (--init --nonsense). Before, the option itself became the image and the run
# ended with "error: cannot open --game-dir" and exit 1, so a mistyped option read as a broken game copy.
set(failures "")
macro(check_refused)
  execute_process(
    COMMAND ${RUNNER} "${LOADER}" ${ARGN}
    TIMEOUT 60
    RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
  string(REPLACE "\r\n" "\n" error "${error}")
  if(NOT code STREQUAL "2" OR NOT error MATCHES "(^|\n)usage: gta6rt"
      OR output MATCHES "module\\(s\\) mapped|running ")
    string(APPEND failures "--- ${ARGN}: returned ${code}\n${output}${error}\n")
  endif()
endmacro()
check_refused(--dry-run --game-dir)
check_refused(--run --trace)
check_refused(--init --nonsense)
if(failures)
  message(FATAL_ERROR "gta6rt did not refuse an argument that names no image\n${failures}")
endif()

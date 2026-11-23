# A --game-dir the loader cannot use is a usage error (exit 2): a file, and a path that does not exist. Before, the
# dry run mapped only the image and exited 0 with the package's imports unimplemented, so a mistyped path read as a
# clean dry run. The image is never opened: the check runs before the loader maps it, so a text file stands in for it.
set(root "${CMAKE_CURRENT_BINARY_DIR}/game_dir_folder")
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}/package")
file(WRITE "${root}/afile.txt" "not a folder")
file(WRITE "${root}/image.exe" "not a PE image")
set(failures "")
macro(check_refused)
  execute_process(
    COMMAND ${RUNNER} "${LOADER}" --dry-run "${root}/image.exe" --game-dir "${ARGN}"
    TIMEOUT 60
    RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
  string(REPLACE "\r\n" "\n" error "${error}")
  if(NOT code STREQUAL "2" OR NOT error MATCHES "(^|\n)error: --game-dir is not a folder:"
      OR output MATCHES "module\\(s\\) mapped|running ")
    string(APPEND failures "--- ${ARGN}: returned ${code}\n${output}${error}\n")
  endif()
endmacro()
check_refused("${root}/afile.txt")
check_refused("${root}/nosuch")
# A folder the loader can use is not refused here: the run goes on and stops at the image.
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --dry-run "${root}/image.exe" --game-dir "${root}/package"
  TIMEOUT 60
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
string(REPLACE "\r\n" "\n" error "${error}")
if(code STREQUAL "2" AND error MATCHES "--game-dir is not a folder")
  string(APPEND failures "--- an existing folder was refused\n${output}${error}\n")
endif()
if(failures)
  message(FATAL_ERROR "gta6rt accepted a --game-dir that is not a folder\n${failures}")
endif()

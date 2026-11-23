# The overlay is where a title's writes land, and never the package: a path that is a file cannot hold them (the first
# write fails later with ERROR_PATH_NOT_FOUND), and the game folder itself would put them inside the package the loader
# promises to leave as installed. Both are the usage error (exit 2) here, before anything is mapped. The image is never
# opened, so a text file stands in for it, and a text image keeps the accepted case short (it stops in the loader).
set(root "${CMAKE_CURRENT_BINARY_DIR}/overlay_folder")
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}/package")
file(WRITE "${root}/afile.txt" "not a folder")
file(WRITE "${root}/image.exe" "not a PE image")
set(failures "")
macro(check_refused)
  execute_process(
    COMMAND ${RUNNER} "${LOADER}" --dry-run "${root}/image.exe" --game-dir "${root}/package" --overlay "${ARGN}"
    TIMEOUT 60
    RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
  string(REPLACE "\r\n" "\n" error "${error}")
  if(NOT code STREQUAL "2" OR NOT error MATCHES "(^|\n)error: --overlay " OR output MATCHES "module\\(s\\) mapped")
    string(APPEND failures "--- ${ARGN}: returned ${code}\n${output}${error}\n")
  endif()
endmacro()
check_refused("${root}/afile.txt")
check_refused("${root}/package")
# A folder of its own is not refused: the run goes on and stops at the image. The path need not exist yet (the default
# <game-dir>/overlay is created on the title's first write), so a not-yet-created overlay is the accepted case too.
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --dry-run "${root}/image.exe" --game-dir "${root}/package"
          --overlay "${root}/new-overlay"
  TIMEOUT 60 RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
string(REPLACE "\r\n" "\n" error "${error}")
if(NOT code STREQUAL "1" OR error MATCHES "error: --overlay " OR NOT error MATCHES "not an MZ image")
  message(FATAL_ERROR "a usable overlay folder was refused or the run did not stop at the image (exit ${code})\n"
                      "${output}${error}")
endif()
if(failures)
  message(FATAL_ERROR "gta6rt did not refuse an unusable overlay\n${failures}")
endif()

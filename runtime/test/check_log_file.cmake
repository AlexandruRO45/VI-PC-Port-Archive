# gta6rt --log FILE: what the loader and the guest print is appended to FILE instead of the console (under Proton's
# steam.exe launch there is no console to print to). Run in a fresh folder; FILE is relative to it.
set(log "${CMAKE_CURRENT_BINARY_DIR}/gta6rt-log-test.txt")
file(REMOVE "${log}")

execute_process(
  COMMAND ${RUNNER} "${LOADER}" --dry-run "${GUEST}" --log "${log}"
  TIMEOUT 90 RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT code STREQUAL "0" OR NOT EXISTS "${log}")
  message(FATAL_ERROR "--dry-run with --log failed (exit ${code}) or wrote no file\n${output}\n${error}")
endif()
file(READ "${log}" text)
if(NOT text MATCHES "1 module\\(s\\) mapped")
  message(FATAL_ERROR "The dry-run report is not in the log file:\n${text}")
endif()
if(output MATCHES "module\\(s\\) mapped")
  message(FATAL_ERROR "The dry-run report still went to stdout:\n${output}")
endif()

# A second run appends: the guest's own printf (the CRT the title imports) and the loader's lines land after the
# first run's report, and the guest's exit code still comes back.
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --run "${GUEST}" --log "${log}" -- hello 42
  TIMEOUT 90 RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
file(READ "${log}" text)
if(NOT code STREQUAL "0")
  message(FATAL_ERROR "--run with --log exited ${code}\n${output}\n${error}\n${text}")
endif()
if(NOT text MATCHES "1 module\\(s\\) mapped.*tiny_game: started, argc=3.*tiny_game: OK.*game exited with code 0")
  message(FATAL_ERROR "The log file lacks the first run's report followed by the guest's output:\n${text}")
endif()
if(output MATCHES "tiny_game:" OR error MATCHES "tiny_game:")
  message(FATAL_ERROR "The guest's output still went to the console:\n${output}\n${error}")
endif()

# A log file that can't be opened is refused before anything is mapped.
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --dry-run "${GUEST}" --log "${CMAKE_CURRENT_BINARY_DIR}/no-such-folder/log.txt"
  TIMEOUT 90 RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(code STREQUAL "0" OR NOT error MATCHES "cannot open the log file" OR output MATCHES "module\\(s\\) mapped")
  message(FATAL_ERROR "An unopenable log file was not refused (exit ${code})\n${output}\n${error}")
endif()
file(REMOVE "${log}")

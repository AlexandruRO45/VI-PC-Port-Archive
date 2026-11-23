# PE TLS DLL_PROCESS_DETACH on immediate-termination paths (_exit, _Exit, quick_exit, ExitProcess).
# Runs tls_immediate_exit natively and through `gta6rt --run`. Both must attach, run main, and return the
# expected exit code without running atexit. Native Windows may omit EXE TLS detach; when present it must
# see 0x42, and the mapped launch always requires detach with that value.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)

set(expect_native_no_detach "tls_attach;main")
set(expect_seq "tls_attach;main;tls_detach=0x42")
set(code__exit 8)
set(code__Exit 9)
set(code_quick_exit 10)
set(code_ExitProcess 11)

function(guest_lines output result)
  string(REPLACE "\r\n" "\n" output "${output}")
  string(REPLACE ";" "," output "${output}")
  string(REPLACE "\n" ";" output "${output}")
  set(lines)
  foreach(line IN LISTS output)
    if(line MATCHES "^tls_immediate_exit: (.*)$")
      list(APPEND lines "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  set(${result} "${lines}" PARENT_SCOPE)
endfunction()

set(failures "")
foreach(mode _exit _Exit quick_exit ExitProcess)
  execute_process(COMMAND ${RUNNER} "${GUEST}" ${mode}
    WORKING_DIRECTORY "${guest_dir}" TIMEOUT 60
    RESULT_VARIABLE native_code OUTPUT_VARIABLE native_out ERROR_VARIABLE native_err)
  execute_process(COMMAND ${RUNNER} "${LOADER}" --run "${guest_name}" -- ${mode}
    WORKING_DIRECTORY "${guest_dir}" TIMEOUT 60
    RESULT_VARIABLE run_code OUTPUT_VARIABLE run_out ERROR_VARIABLE run_err)
  guest_lines("${native_out}" native_lines)
  guest_lines("${run_out}" run_lines)
  if(NOT native_code STREQUAL code_${mode} OR
     (NOT native_lines STREQUAL expect_native_no_detach AND NOT native_lines STREQUAL expect_seq))
    string(CONCAT native_diff "[${mode}] native run differs from the expected sequence"
      "\n  expected (${code_${mode}}): ${expect_native_no_detach}, optionally followed by tls_detach=0x42\n  native   (${native_code}): ${native_lines}")
    if(RUNNER)
      message(STATUS "${native_diff}\n  (the host CRT/loader differs from Windows; not a gta6rt failure)")
    else()
      string(APPEND failures "\n${native_diff}\n  (the fixture or the host changed)")
    endif()
  endif()
  if(NOT run_code STREQUAL code_${mode} OR NOT run_lines STREQUAL expect_seq)
    string(APPEND failures "\n[${mode}] gta6rt --run differs from the expected sequence"
      "\n  expected (${code_${mode}}): ${expect_seq}\n  --run    (${run_code}): ${run_lines}\n${run_err}")
  endif()
endforeach()
if(failures)
  message(FATAL_ERROR "immediate-exit TLS detach differences:${failures}")
endif()
message(STATUS "tls_immediate_exit: 4 immediate-exit paths delivered PE TLS DLL_PROCESS_DETACH")

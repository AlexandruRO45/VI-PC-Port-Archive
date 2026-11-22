function(check_invalid label)
  execute_process(COMMAND ${RUNNER} "${LOADER}" ${ARGN}
                  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output TIMEOUT 10)
  if(NOT code STREQUAL "2" OR NOT output MATCHES "usage: gta6rt" OR output MATCHES "module\\(s\\) mapped")
    message(FATAL_ERROR "${label}: expected usage error before mapping, got ${code}\n${output}")
  endif()
endfunction()

check_invalid(game_dir_missing --dry-run "${GUEST}" --game-dir)
check_invalid(game_dir_before_flag --dry-run "${GUEST}" --game-dir --trace-d3d12)
check_invalid(overlay_before_separator --dry-run "${GUEST}" --overlay --)
check_invalid(trace_before_flag --dry-run "${GUEST}" --trace --trace-d3d12)
check_invalid(unknown_option --dry-run --misspelled-option)

# Path options before the separator still accept paths. Similar tokens after it are passed to the guest.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
string(RANDOM LENGTH 12 nonce)
set(trace "${guest_dir}/loader path options ${nonce}.jsonl")
execute_process(COMMAND ${RUNNER} "${LOADER}" --run "${GUEST}"
                --game-dir "${guest_dir}" --overlay "${guest_dir}/overlay" --trace "${trace}"
                -- hello 42 --trace-d3d12 --game-dir --overlay --trace
                RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output TIMEOUT 90)
if(NOT code STREQUAL "0" OR NOT output MATCHES "tiny_game: started, argc=7"
    OR NOT output MATCHES "game exited with code 0")
  message(FATAL_ERROR "Guest did not receive all game arguments, exit ${code}\n${output}")
endif()
if(NOT EXISTS "${trace}")
  message(FATAL_ERROR "Expected a startup trace at ${trace}\n${output}")
endif()
file(STRINGS "${trace}" first_event LIMIT_COUNT 1)
file(REMOVE "${trace}")
if(NOT first_event MATCHES "trace_open")
  message(FATAL_ERROR "Expected the startup trace to begin with trace_open: ${first_event}")
endif()

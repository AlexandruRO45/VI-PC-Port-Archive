# Exercise the CLI with a locally built stand-in image; no game files.
if(NOT DEFINED LOADER OR NOT DEFINED GUEST)
  message(FATAL_ERROR "LOADER and GUEST are required")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef nonce)
set(dir "${CMAKE_CURRENT_BINARY_DIR}/trace_creation_${nonce}")
file(MAKE_DIRECTORY "${dir}")
set(trace_name "trace_é_中.jsonl")
set(trace "${dir}/${trace_name}")
set(image "${dir}/synthetic.exe")
file(COPY_FILE "${GUEST}" "${image}")
# Relative guest arguments also work under Wine; only the host CMake handles the absolute paths.
execute_process(COMMAND ${RUNNER} "${LOADER}" --dry-run synthetic.exe --trace "${trace_name}"
  WORKING_DIRECTORY "${dir}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 20)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "fresh trace failed (${result}): ${output}${error}")
endif()
file(STRINGS "${trace}" lines LIMIT_COUNT 1)
string(JSON kind ERROR_VARIABLE json_error GET "${lines}" kind)
if(json_error OR NOT kind STREQUAL "trace_open")
  message(FATAL_ERROR "fresh trace does not begin with a valid trace_open event")
endif()

# Refusing an existing report must preserve it, including a file also used as input.
file(SHA256 "${trace}" before)
execute_process(COMMAND ${RUNNER} "${LOADER}" --dry-run synthetic.exe --trace "${trace_name}"
  WORKING_DIRECTORY "${dir}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 20)
file(SHA256 "${trace}" after)
if(NOT result STREQUAL "2" OR NOT before STREQUAL after OR NOT error MATCHES "trace file already exists")
  message(FATAL_ERROR "existing trace refusal failed (${result}): ${output}${error}")
endif()
file(SHA256 "${image}" before)
execute_process(COMMAND ${RUNNER} "${LOADER}" --dry-run synthetic.exe --trace synthetic.exe
  WORKING_DIRECTORY "${dir}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 20)
file(SHA256 "${image}" after)
if(NOT result STREQUAL "2" OR NOT before STREQUAL after)
  message(FATAL_ERROR "input image was not protected (${result}): ${output}${error}")
endif()

foreach(path "." "missing/trace.jsonl")
  execute_process(COMMAND ${RUNNER} "${LOADER}" --dry-run synthetic.exe --trace "${path}"
    WORKING_DIRECTORY "${dir}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 20)
  if(NOT result STREQUAL "2" OR NOT error MATCHES "cannot create the trace file")
    message(FATAL_ERROR "bad trace destination was not refused cleanly (${result}): ${output}${error}")
  endif()
endforeach()
file(REMOVE "${trace}" "${image}")
file(REMOVE_RECURSE "${dir}")
message(STATUS "startup trace creation, collision preservation and error paths passed")

# Isolate the loader from the helper without moving files used by other tests.
if(NOT EXISTS "${LOADER}" OR IS_DIRECTORY "${LOADER}" OR NOT EXISTS "${GUEST}" OR IS_DIRECTORY "${GUEST}")
  message(FATAL_ERROR "LOADER and GUEST must name existing executables")
endif()
string(RANDOM LENGTH 16 suffix)
set(scratch "${CMAKE_CURRENT_BINARY_DIR}/missing-tls-helper-${suffix}")
file(MAKE_DIRECTORY "${scratch}")
get_filename_component(loader_name "${LOADER}" NAME)
file(COPY_FILE "${LOADER}" "${scratch}/${loader_name}")
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
execute_process(
  COMMAND ${RUNNER} "${scratch}/${loader_name}" --dry-run "${guest_name}"
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT 60 RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output)
file(REMOVE_RECURSE "${scratch}")
# The system message is localized. ERROR_MOD_NOT_FOUND (126) identifies the failed OS load.
if(NOT code STREQUAL "1" OR NOT output MATCHES "cannot load gta6rt_tls_host[.]dll [(]Win32 error 126[)]")
  message(FATAL_ERROR "Unexpected missing-helper result (${code}): ${output}")
endif()

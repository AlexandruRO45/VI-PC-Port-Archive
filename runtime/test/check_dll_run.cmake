get_filename_component(dll_dir "${DLL}" DIRECTORY)
get_filename_component(dll_name "${DLL}" NAME)

execute_process(COMMAND ${RUNNER} "${LOADER}" --run "${dll_name}"
                WORKING_DIRECTORY "${dll_dir}"
                RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output TIMEOUT 30)
if(NOT code STREQUAL "1" OR NOT output MATCHES "--run requires an executable image; use --init for a DLL"
    OR output MATCHES "module\\(s\\) initialised|running ")
  message(FATAL_ERROR "Expected --run to reject the DLL before initialisation, got ${code}\n${output}")
endif()

execute_process(COMMAND ${RUNNER} "${LOADER}" --init "${dll_name}"
                WORKING_DIRECTORY "${dll_dir}"
                RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output TIMEOUT 30)
if(NOT code STREQUAL "0" OR NOT output MATCHES "[1-9][0-9]* module\\(s\\) initialised")
  message(FATAL_ERROR "Expected --init to initialise the same DLL, got ${code}\n${output}")
endif()

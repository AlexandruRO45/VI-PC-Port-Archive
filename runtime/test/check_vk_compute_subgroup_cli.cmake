# Invalid settings must fail before device selection or shader loading.
# The device filter prevents reaching pipeline creation even on builds lacking the check.
foreach(value IN ITEMS "32junk" "48" "-32" "+32" " 32" "32 " "0x20" "4294967328")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env
                  "LEONIDA_VK_DEVICE=leonida-compute-cli-no-matching-device"
                  "VKC_SUBGROUP_SIZE=${value}"
                  ${RUNNER} "${TOOL}" unused.spv 1 dump
                  RESULT_VARIABLE code ERROR_VARIABLE log TIMEOUT 10)
  if(NOT code STREQUAL "2" OR NOT log MATCHES "VKC_SUBGROUP_SIZE: expected")
    message(FATAL_ERROR "VKC_SUBGROUP_SIZE=\"${value}\": expected usage error, got ${code}\n${log}")
  endif()
endforeach()

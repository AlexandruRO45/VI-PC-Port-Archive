file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")

# Minimal 20-byte SPIR-V header: little-endian magic 0x07230203 followed by four 32-bit header words.
string(ASCII 3 2 35 7 spv_magic)
set(spv_header "${spv_magic}0123456789abcdef")

set(vs_spv "${TEST_DIR}/vs.spv")
set(fs_spv "${TEST_DIR}/fs.spv")
set(out_raw "${TEST_DIR}/out.raw")
file(WRITE "${vs_spv}" "${spv_header}")
file(WRITE "${fs_spv}" "${spv_header}")
file(SIZE "${vs_spv}" vs_size)
file(READ "${vs_spv}" vs_magic_hex LIMIT 4 HEX)
if(NOT vs_size EQUAL 20 OR NOT vs_magic_hex STREQUAL "03022307")
  message(FATAL_ERROR "Failed to construct 20-byte SPIR-V fixture (${vs_size} bytes, magic ${vs_magic_hex})")
endif()

# ${RUNNER} is CMAKE_CROSSCOMPILING_EMULATOR: CTest does not apply it to a script
# test (the command is the host CMake), so the script runs the tool through it itself.
function(expect_exit2 substring)
  execute_process(COMMAND ${RUNNER} "${TOOL}" ${ARGN}
                  WORKING_DIRECTORY "${TEST_DIR}"
                  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
  string(FIND "${log}" "${substring}" found)
  if(NOT code STREQUAL "2" OR found EQUAL -1)
    message(FATAL_ERROR "${ARGN}: expected exit 2 containing '${substring}', got ${code}\n${log}")
  endif()
endfunction()

# 1. Unknown flag-like arguments must print usage and exit 2 without creating files named after the flag.
foreach(flag --help -h --unknown)
  expect_exit2("usage:" "${flag}")
  if(EXISTS "${TEST_DIR}/${flag}")
    message(FATAL_ERROR "Flag '${flag}' was treated as an output file path")
  endif()
  expect_exit2("usage:" cube "${flag}")
  if(EXISTS "${TEST_DIR}/${flag}")
    message(FATAL_ERROR "Autonomous 'cube ${flag}' treated '${flag}' as an output file path")
  endif()
endforeach()

# 2. Unknown 10th option must print usage and exit 2.
expect_exit2("usage:" "${vs_spv}" "${fs_spv}" "${out_raw}" 0 0 0 0 0 --two-mip-bad)

# 3. Invalid ptr_sgpr (outside 0..31 or non-decimal) and invalid hex s0..s3 must exit 2 before Vulkan setup.
foreach(bad_sgpr -1 32 999 abc 1x +1)
  expect_exit2("invalid ptr_sgpr" "${vs_spv}" "${fs_spv}" "${out_raw}" "${bad_sgpr}" 0 0 0 0)
endforeach()
execute_process(COMMAND ${RUNNER} "${TOOL}" "${vs_spv}" "${fs_spv}" "${out_raw}" "" 0 0 0 0
                WORKING_DIRECTORY "${TEST_DIR}"
                RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
if(NOT code STREQUAL "2" OR NOT log MATCHES "invalid ptr_sgpr")
  message(FATAL_ERROR "empty ptr_sgpr: expected exit 2 with 'invalid ptr_sgpr', got ${code}\n${log}")
endif()

foreach(bad_hex -1 100000000 0x100000000 0x1g xyz +1 0x)
  expect_exit2("invalid S# word s0" "${vs_spv}" "${fs_spv}" "${out_raw}" 0 "${bad_hex}" 0 0 0)
  expect_exit2("invalid S# word s1" "${vs_spv}" "${fs_spv}" "${out_raw}" 0 0x92 "${bad_hex}" 0 0)
  expect_exit2("invalid S# word s2" "${vs_spv}" "${fs_spv}" "${out_raw}" 0 0x92 0xc00000 "${bad_hex}" 0)
  expect_exit2("invalid S# word s3" "${vs_spv}" "${fs_spv}" "${out_raw}" 0 0x92 0xc00000 0x14500000 "${bad_hex}")
endforeach()
execute_process(COMMAND ${RUNNER} "${TOOL}" "${vs_spv}" "${fs_spv}" "${out_raw}" 0 "" 0 0 0
                WORKING_DIRECTORY "${TEST_DIR}"
                RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
if(NOT code STREQUAL "2" OR NOT log MATCHES "invalid S# word s0")
  message(FATAL_ERROR "empty s0: expected exit 2 with 'invalid S# word s0', got ${code}\n${log}")
endif()

# 4. Missing, empty, truncated (< 20 bytes), unaligned (% 4 != 0), and bad-magic SPIR-V inputs exit 2 naming the path.
set(missing_vs "${TEST_DIR}/missing_vs.spv")
set(missing_fs "${TEST_DIR}/missing_fs.spv")
expect_exit2("${missing_vs}" "${missing_vs}" "${fs_spv}" "${out_raw}" 0 0 0 0 0)
expect_exit2("${missing_fs}" "${vs_spv}" "${missing_fs}" "${out_raw}" 0 0 0 0 0)

set(empty_spv "${TEST_DIR}/empty.spv")
set(trunc_spv "${TEST_DIR}/truncated.spv")
set(unaligned_spv "${TEST_DIR}/unaligned.spv")
set(bad_magic_spv "${TEST_DIR}/bad_magic.spv")
file(WRITE "${empty_spv}" "")
file(WRITE "${trunc_spv}" "${spv_magic}0123")
file(WRITE "${unaligned_spv}" "${spv_header}x")
file(WRITE "${bad_magic_spv}" "not a spirv module!!")

foreach(bad_spv "${empty_spv}" "${trunc_spv}" "${unaligned_spv}" "${bad_magic_spv}")
  expect_exit2("${bad_spv}" "${bad_spv}" "${fs_spv}" "${out_raw}" 31 0x92 c00000 0X14500000 0xffffffff)
  expect_exit2("${bad_spv}" "${vs_spv}" "${bad_spv}" "${out_raw}" 31 0x92 c00000 0X14500000 0xffffffff)
endforeach()

# 5. Output path aliasing vs.spv or fs.spv must exit 2 before touching Vulkan or modifying either shader file.
file(SHA256 "${vs_spv}" vs_before)
file(SHA256 "${fs_spv}" fs_before)
expect_exit2("must not overwrite" "${vs_spv}" "${fs_spv}" "${vs_spv}" 0 0 0 0 0)
expect_exit2("must not overwrite" "${vs_spv}" "${fs_spv}" "${fs_spv}" 0 0 0 0 0)
expect_exit2("must not overwrite" vs.spv fs.spv ./vs.spv 0 0 0 0 0)
expect_exit2("must not overwrite" vs.spv fs.spv ./fs.spv 0 0 0 0 0)

file(CREATE_LINK "${vs_spv}" "${TEST_DIR}/vs-link.spv" RESULT hard_link)
if(hard_link STREQUAL "0")
  expect_exit2("must not overwrite" "${vs_spv}" "${fs_spv}" "${TEST_DIR}/vs-link.spv" 0 0 0 0 0)
  file(REMOVE "${TEST_DIR}/vs-link.spv")
endif()

file(SHA256 "${vs_spv}" vs_after)
file(SHA256 "${fs_spv}" fs_after)
if(NOT vs_before STREQUAL vs_after OR NOT fs_before STREQUAL fs_after)
  message(FATAL_ERROR "Input/output path conflict modified vs.spv or fs.spv")
endif()

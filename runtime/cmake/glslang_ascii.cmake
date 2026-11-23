# Runs glslangValidator on a copy of its input in an ASCII-only folder, with relative paths, and copies the
# result back (RT-518).
#
#   cmake -DLEONIDA_GLSLANG_STAGING_DIR=<ascii folder> -P glslang_ascii.cmake -- <glslangValidator> <arguments>...
#
# On Windows glslangValidator takes its arguments through the ANSI code page, so it cannot open an input or output
# path the code page can't spell ("unable to open input file"). It also asks std::filesystem for the working
# directory as a narrow string, which throws and aborts the process when the working directory is such a folder,
# even for relative paths or --stdin. A checkout under a user name in another script hits both. CMake itself uses
# the wide Windows APIs, so it does the copying.
#
# The value after -o is the output; every other argument that names an existing file is an input, staged under its
# file name. The test shaders include no other files, so only the named inputs are copied.
cmake_minimum_required(VERSION 3.24)

if(NOT LEONIDA_GLSLANG_STAGING_DIR)
  message(FATAL_ERROR "glslang_ascii.cmake: set LEONIDA_GLSLANG_STAGING_DIR to an ASCII-only folder")
endif()
if(LEONIDA_GLSLANG_STAGING_DIR MATCHES "[^ -~]")
  message(FATAL_ERROR "glslang_ascii.cmake: LEONIDA_GLSLANG_STAGING_DIR is not ASCII-only: "
                      "${LEONIDA_GLSLANG_STAGING_DIR}")
endif()

set(command)
set(found_separator OFF)
math(EXPR last "${CMAKE_ARGC} - 1")
foreach(index RANGE 1 ${last})
  if(found_separator)
    list(APPEND command "${CMAKE_ARGV${index}}")
  elseif(CMAKE_ARGV${index} STREQUAL "--")
    set(found_separator ON)
  endif()
endforeach()
list(LENGTH command count)
if(count LESS 2)
  message(FATAL_ERROR "glslang_ascii.cmake: usage: cmake -P glslang_ascii.cmake -- <glslangValidator> <arguments>...")
endif()
list(POP_FRONT command glslang)

set(arguments)
set(inputs)
set(output "")
set(next_is_output OFF)
foreach(argument IN LISTS command)
  if(next_is_output)
    set(output "${argument}")
    get_filename_component(name "${argument}" NAME)
    list(APPEND arguments "${name}")
    set(next_is_output OFF)
  elseif(argument STREQUAL "-o")
    list(APPEND arguments "${argument}")
    set(next_is_output ON)
  elseif(IS_ABSOLUTE "${argument}" AND EXISTS "${argument}" AND NOT IS_DIRECTORY "${argument}")
    get_filename_component(name "${argument}" NAME)
    list(APPEND inputs "${argument}")
    list(APPEND arguments "${name}")
  else()
    list(APPEND arguments "${argument}")
  endif()
endforeach()
if(output STREQUAL "" OR NOT IS_ABSOLUTE "${output}")
  message(FATAL_ERROR "glslang_ascii.cmake: expected -o <absolute output path>")
endif()
if(NOT inputs)
  message(FATAL_ERROR "glslang_ascii.cmake: no input file among the arguments")
endif()

# One folder per output, so a parallel build compiles its shaders side by side.
string(SHA1 key "${output}")
string(SUBSTRING "${key}" 0 16 key)
set(work "${LEONIDA_GLSLANG_STAGING_DIR}/${key}")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")
foreach(input IN LISTS inputs)
  get_filename_component(name "${input}" NAME)
  file(COPY_FILE "${input}" "${work}/${name}" ONLY_IF_DIFFERENT)
endforeach()

execute_process(COMMAND "${glslang}" ${arguments} WORKING_DIRECTORY "${work}" RESULT_VARIABLE result)
get_filename_component(name "${output}" NAME)
if(NOT result EQUAL 0 OR NOT EXISTS "${work}/${name}")
  file(REMOVE_RECURSE "${work}")
  message(FATAL_ERROR "glslangValidator failed (${result}) on ${inputs}, compiled in ${work}")
endif()
get_filename_component(directory "${output}" DIRECTORY)
file(MAKE_DIRECTORY "${directory}")
file(COPY_FILE "${work}/${name}" "${output}")
file(REMOVE_RECURSE "${work}")

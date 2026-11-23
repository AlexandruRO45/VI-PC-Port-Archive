# Runs glslangValidator with ASCII paths on Windows:
#   cmake -P glslang_ascii_paths.cmake -- <glslangValidator> <arguments...>
#
# glslangValidator (Vulkan SDK 1.4.363, Glslang 16.6; glslc too) reads its arguments in the ANSI code page, so it
# cannot open a source or -o path with characters outside it ("Error: unable to open input file"), and it exits
# 0xC0000409 when its working directory has such a name, even for ASCII arguments. This script maps the folder of
# each such path argument, and the working directory, to the folder's 8.3 short name. The short name is read by
# running cmd in that folder, so the non-ASCII path is never a command-line argument. Where the volume has no 8.3
# names, the path is left as it is and glslangValidator reports the file it cannot open.
function(ascii_dir out dir)
  set(result "${dir}")
  if(dir MATCHES "[^\t -~]" AND IS_DIRECTORY "${dir}")
    execute_process(COMMAND cmd /d /c for %I in (.) do @echo %~sI
                    WORKING_DIRECTORY "${dir}"
                    OUTPUT_VARIABLE short OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE rc)
    if(rc EQUAL 0 AND short AND NOT short MATCHES "[^\t -~]")
      file(TO_CMAKE_PATH "${short}" result)
    endif()
  endif()
  set(${out} "${result}" PARENT_SCOPE)
endfunction()

set(command)
set(seen_separator FALSE)
math(EXPR last "${CMAKE_ARGC} - 1")
foreach(i RANGE 1 ${last})
  set(arg "${CMAKE_ARGV${i}}")
  if(NOT seen_separator)
    if(arg STREQUAL "--")
      set(seen_separator TRUE)
    endif()
    continue()
  endif()
  if(arg MATCHES "[^\t -~]" AND IS_ABSOLUTE "${arg}")
    get_filename_component(folder "${arg}" DIRECTORY)
    get_filename_component(name "${arg}" NAME)
    ascii_dir(folder "${folder}")
    set(arg "${folder}/${name}")
  endif()
  list(APPEND command "${arg}")
endforeach()
if(NOT command)
  message(FATAL_ERROR "usage: cmake -P glslang_ascii_paths.cmake -- <glslangValidator> <arguments...>")
endif()

ascii_dir(cwd "${CMAKE_CURRENT_BINARY_DIR}")
execute_process(COMMAND ${command} WORKING_DIRECTORY "${cwd}" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "glslangValidator failed (${rc})")
endif()

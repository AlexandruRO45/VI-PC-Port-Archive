# Runs glslangValidator for a checkout whose folder is not ASCII (RT-518):
#   cmake -DSOURCE_DIR=<dir> -DSOURCE_SHORT=<8.3 name> -DBINARY_DIR=<dir> -DBINARY_SHORT=<8.3 name>
#         -P glslang_short_paths.cmake -- <glslangValidator> <arguments>
# glslangValidator reads its arguments in the ANSI code page, and with -V it aborts (0xC0000409) when its working
# directory cannot be spelled in that page, so a folder such as 東京 on a Western Windows fails every GLSL test
# shader however its paths are written. This runs it in the build folder's 8.3 short name, with the source and
# build folders in its arguments spelled the same way. CMakeLists.txt computes the short names at configure time.
set(command)
set(passthrough FALSE)
math(EXPR last "${CMAKE_ARGC} - 1")
foreach(i RANGE 1 ${last})
  set(arg "${CMAKE_ARGV${i}}")
  if(passthrough)
    string(REPLACE "${BINARY_DIR}" "${BINARY_SHORT}" arg "${arg}")
    string(REPLACE "${SOURCE_DIR}" "${SOURCE_SHORT}" arg "${arg}")
    list(APPEND command "${arg}")
  elseif(arg STREQUAL "--")
    set(passthrough TRUE)
  endif()
endforeach()
execute_process(COMMAND ${command} WORKING_DIRECTORY "${BINARY_SHORT}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "glslangValidator failed (${result})")
endif()

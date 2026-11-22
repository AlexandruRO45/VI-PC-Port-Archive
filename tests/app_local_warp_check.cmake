# Configures one build directory with OFFSCREEN_WARP_DIR set and then cleared, using a stand-in DLL.
# Run with: cmake -DSOURCE_DIR=<repo> -DWORK_DIR=<scratch> -P tests/app_local_warp_check.cmake
include("${SOURCE_DIR}/cmake/app_local_warp.cmake")
set(package "${WORK_DIR}/warp-package")
set(build "${WORK_DIR}/build")
file(MAKE_DIRECTORY "${package}" "${build}")
# Clear only the stand-in files this test creates; anything else under WORK_DIR is left alone.
file(REMOVE "${package}/d3d10warp.dll" "${build}/d3d10warp.dll")
file(WRITE "${package}/d3d10warp.dll" "stand-in WARP")

offscreen_app_local_warp("${package}" "${build}")
if(NOT EXISTS "${build}/d3d10warp.dll")
    message(FATAL_ERROR "Setting OFFSCREEN_WARP_DIR did not place d3d10warp.dll")
endif()
file(READ "${build}/d3d10warp.dll" copied)
if(NOT copied STREQUAL "stand-in WARP")
    message(FATAL_ERROR "The placed d3d10warp.dll differs from the package copy")
endif()

offscreen_app_local_warp("" "${build}")
if(EXISTS "${build}/d3d10warp.dll")
    message(FATAL_ERROR "Clearing OFFSCREEN_WARP_DIR left d3d10warp.dll beside the executable")
endif()

# A configure without the option and no earlier copy must also succeed.
offscreen_app_local_warp("" "${build}")
message(STATUS "app_local_warp set=placed cleared=removed repeat_clear=ok")

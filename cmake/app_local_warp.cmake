# Places or removes the app-local WARP beside offscreen_check.
# DXGI's WARP adapter loads d3d10warp.dll from the executable's directory before the system copy,
# so clearing the option must also delete a copy left by an earlier configure of the same build
# directory. Otherwise the stale copy keeps replacing the in-box WARP.
function(offscreen_app_local_warp warp_dir binary_dir)
    if(warp_dir)
        configure_file("${warp_dir}/d3d10warp.dll" "${binary_dir}/d3d10warp.dll" COPYONLY)
    else()
        file(REMOVE "${binary_dir}/d3d10warp.dll")
    endif()
endfunction()

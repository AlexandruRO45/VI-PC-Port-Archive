# SDL3 options of the pinned SDL3 fetch (runtime/CMakeLists.txt) that depend on the build host, not on gamepads.
#
# libusb: SDL3's hidapi has an optional libusb backend. On Windows it reaches only a device that has a WinUSB driver
# installed for it; pads come through XInput and SDL's Windows hidapi (docs/gameinput.md). A cross build would build it
# against the build host's libusb instead: the clang-cl-xwin toolchain finds libraries and headers on the host as well
# (find modes BOTH), so on a Mac with Homebrew's libusb SDL3 takes its Mach-O libusb-1.0.a and configure stops in
# SDL's import-library check (dumpbin, or llvm-dlltool: "No DLL import name found"). The backend is therefore off by
# default in a cross build. -DLEONIDA_SDL3_LIBUSB=ON turns it on.
#
# SDL_HIDAPI_LIBUSB is set with FORCE: a configure that already failed there has SDL's own ON in its cache.
# tests/test_cmake_sdl3_fetch_options.py runs this file with cmake -P.
macro(leonida_sdl3_fetch_options)
  if(CMAKE_CROSSCOMPILING)
    set(_leonida_sdl3_libusb_default OFF)
  else()
    set(_leonida_sdl3_libusb_default ON)
  endif()
  option(LEONIDA_SDL3_LIBUSB "Build SDL3's libusb hidapi backend (default: off in a cross build, SDL's default otherwise)"
         ${_leonida_sdl3_libusb_default})
  set(SDL_HIDAPI_LIBUSB ${LEONIDA_SDL3_LIBUSB} CACHE BOOL "" FORCE)
endmacro()

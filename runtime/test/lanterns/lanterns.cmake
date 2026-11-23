# Lanterns, a synthetic title-shaped guest run by `gta6rt --run` (runtime/test/lanterns/lanterns.cpp, docs/lanterns.md).
# Included from runtime/CMakeLists.txt inside the Vulkan device block, after tiny_frame; opt-in, because it needs a
# Vulkan GPU at run time:
#
#   -DLEONIDA_TEST_LANTERNS=ON, then build the `lanterns` and `gta6rt` targets.
#
# The shaders: fssm_pack.py (this folder) assembles shaders/*.asm with llvm-mc, packs each into a synthetic FSSM program
# inside a DXBC/XDAT/XBCH container (<build>/generated/lanterns/<name>.dxbc, what the guest hands to
# CreateGraphicsPipelineState) and runs tools/fssm_unpack.py, shader_disasm.py and rdna2spirv.py on that folder, which
# write spirv/index.json keyed by each program's FSSM SHA-256 (what LEONIDA_SHADERS points the runtime at).
# --wave64-emulate makes the wave64 compute shaders exact on 32-lane GPUs; a native variant is translated without it.
# The guest links nothing from src/ or shims/: its device comes from the D3D12XboxCreateDevice it imports from
# d3d12_xs.dll, which only gta6rt's shim registry can answer. The import library is tiny_frame's failing stub
# (d3d12_xs_stub, test/fixtures/d3d12_xs_stub.cpp), built into a folder the loader does not search.
#
# Running it: run_lanterns.py (native Windows, or Wine with --wine) checks the exit code, the guest's PASS line and the
# device line, and writes PNGs of the frames. On a native Windows build it is also the ctest `lanterns_stage1`.
option(LEONIDA_TEST_LANTERNS
       "Build lanterns.exe, a synthetic title-shaped guest for gta6rt --run (needs a Vulkan GPU to run)" OFF)
if(LEONIDA_TEST_LANTERNS AND NOT TARGET d3d12_xs_stub)
  message(STATUS "LEONIDA_TEST_LANTERNS: lanterns.exe needs gta6rt and the d3d12_xs_stub target; not built")
elseif(LEONIDA_TEST_LANTERNS)
  set(_lanterns_src ${CMAKE_CURRENT_SOURCE_DIR}/test/lanterns)
  add_executable(lanterns ${_lanterns_src}/lanterns.cpp ${_lanterns_src}/lanterns_gpu.cpp
                          ${_lanterns_src}/lanterns_edges_gpu.cpp ${_lanterns_src}/lanterns_textures_gpu.cpp
                          ${_lanterns_src}/lanterns_shadow_gpu.cpp ${_lanterns_src}/lanterns_post_gpu.cpp
                          ${_lanterns_src}/lanterns_deferred_gpu.cpp ${_lanterns_src}/lanterns_rt_gpu.cpp
                          ${_lanterns_src}/lanterns_samplers_gpu.cpp ${_lanterns_src}/lanterns_xy_gpu.cpp
                          ${_lanterns_src}/lanterns_gbuf6_gpu.cpp
                          ${_lanterns_src}/lanterns_scale.cpp
                          ${_lanterns_src}/lanterns_scale_gpu.cpp
                          ${_lanterns_src}/lanterns_scale_prof.cpp)
  target_link_libraries(lanterns PRIVATE d3d12_xs_stub Vulkan::Headers)
  target_compile_definitions(lanterns PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
  target_compile_options(lanterns PRIVATE /W4 /permissive- /utf-8)
  set_target_properties(lanterns PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/lanterns/$<CONFIG>")

  if(NOT LEONIDA_LLVM_MC)
    if(DEFINED ENV{LLVM_MC})
      set(LEONIDA_LLVM_MC "$ENV{LLVM_MC}")
    else()
      find_program(LEONIDA_LLVM_MC llvm-mc HINTS "C:/msys64/ucrt64/bin")
    endif()
  endif()
  find_program(LEONIDA_SPIRV_AS spirv-as HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
  if(LEONIDA_LLVM_MC AND LEONIDA_SPIRV_AS AND TARGET Python3::Interpreter)
    set(LEONIDA_LANTERNS_SHADERS ${CMAKE_CURRENT_BINARY_DIR}/generated/lanterns)
    set(_lanterns_tools ${CMAKE_CURRENT_SOURCE_DIR}/../tools)
    # Every tools/*.py and tools/*.json: the translator imports many of the scripts (fssm_header, shader_ir,
    # shader_meta, mc_command, rdna2_decode, ...) and reads data such as gfx10_formats.json (rdna2spirv.py,
    # rdna2_decode.py), and a stale index.json must not survive a change to any of them.
    file(GLOB _lanterns_tool_py CONFIGURE_DEPENDS ${_lanterns_tools}/*.py ${_lanterns_tools}/*.json)
    set(_lanterns_asm ${_lanterns_src}/shaders/vs_lanterns.asm ${_lanterns_src}/shaders/ps_lanterns_lit.asm
                      ${_lanterns_src}/shaders/ps_lanterns_floor.asm ${_lanterns_src}/shaders/vs_lanterns_w64.asm
                      ${_lanterns_src}/shaders/ps_lanterns_lit_w64.asm ${_lanterns_src}/shaders/vs_lanterns_edges.asm
                      ${_lanterns_src}/shaders/ps_lanterns_edges.asm ${_lanterns_src}/shaders/cs_lanterns_rise.asm
                      ${_lanterns_src}/shaders/ps_lanterns_tex_floor.asm
                      ${_lanterns_src}/shaders/ps_lanterns_tex_solid.asm
                      ${_lanterns_src}/shaders/ps_lanterns_shadow.asm ${_lanterns_src}/shaders/cs_lanterns_bloom_h.asm
                      ${_lanterns_src}/shaders/cs_lanterns_bloom_h_nobarrier.asm
                      ${_lanterns_src}/shaders/cs_lanterns_bloom_v.asm ${_lanterns_src}/shaders/vs_lanterns_full.asm
                      ${_lanterns_src}/shaders/ps_lanterns_composite.asm
                      ${_lanterns_src}/shaders/vs_lanterns_gbuf.asm ${_lanterns_src}/shaders/ps_lanterns_gbuf.asm
                      ${_lanterns_src}/shaders/cs_lanterns_tiled.asm
                      ${_lanterns_src}/shaders/cs_lanterns_rt_shadow.asm
                      ${_lanterns_src}/shaders/cs_lanterns_samplers.asm
                      ${_lanterns_src}/shaders/cs_lanterns_xy.asm
                      ${_lanterns_src}/shaders/ps_lanterns_gbuf6.asm ${_lanterns_src}/shaders/cs_lanterns_gbuf_decode.asm
                      ${_lanterns_src}/shaders/cs_lanterns_rt_alpha.asm)
    get_filename_component(_lanterns_spirv_tools ${LEONIDA_SPIRV_AS} DIRECTORY)
    file(TO_NATIVE_PATH "${_lanterns_spirv_tools}" _lanterns_spirv_tools)
    add_custom_command(
      OUTPUT ${LEONIDA_LANTERNS_SHADERS}/spirv/index.json
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${LEONIDA_LANTERNS_SHADERS}
      COMMAND ${CMAKE_COMMAND} -E env "LLVM_MC=${LEONIDA_LLVM_MC}" "LEONIDA_SPIRV_TOOLS=${_lanterns_spirv_tools}"
              $<TARGET_FILE:Python3::Interpreter> -I ${_lanterns_src}/fssm_pack.py
              ${LEONIDA_LANTERNS_SHADERS} ${_lanterns_asm} --translate --wave64-emulate
      DEPENDS ${_lanterns_asm} ${_lanterns_src}/fssm_pack.py ${_lanterns_tool_py}
      COMMENT "Packing and translating the Lanterns shaders"
      VERBATIM)
    add_custom_target(lanterns_shaders DEPENDS ${LEONIDA_LANTERNS_SHADERS}/spirv/index.json)
    add_dependencies(lanterns lanterns_shaders)
    if(CMAKE_HOST_WIN32 AND NOT CMAKE_CROSSCOMPILING)
      # One test per stage (the runner gives each its own output folder under --out).
      foreach(_stage 1 3 5 6 8 9 10 11 20 22 23 24 25)
        add_test(NAME lanterns_stage${_stage}
                 COMMAND $<TARGET_FILE:Python3::Interpreter> -I ${_lanterns_src}/run_lanterns.py
                         --gta6rt $<TARGET_FILE:gta6rt> --guest $<TARGET_FILE:lanterns>
                         --shaders ${LEONIDA_LANTERNS_SHADERS} --out ${CMAKE_CURRENT_BINARY_DIR}/lanterns_out
                         --stage ${_stage})
        set_tests_properties(lanterns_stage${_stage} PROPERTIES TIMEOUT 300)
      endforeach()
    endif()
  else()
    message(STATUS "llvm-mc (set LLVM_MC), spirv-as or Python 3 not found: lanterns.exe builds without its shaders")
  endif()
endif()

cmake_minimum_required(VERSION 3.20)
project(InnerCoverageConformance LANGUAGES CXX)
include(CTest)
add_executable(inner_coverage_mask_layout mask_layout_test.cpp)
target_compile_features(inner_coverage_mask_layout PRIVATE cxx_std_20)
add_test(NAME inner_coverage_mask_layout COMMAND inner_coverage_mask_layout)

# Standalone on purpose: building this synthetic fixture does not enable runtime Tier 3.
find_path(INNER_VULKAN_INCLUDE vulkan/vulkan.h REQUIRED)
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${INNER_VULKAN_INCLUDE}")
check_cxx_source_compiles("#include <vulkan/vulkan.h>
#ifndef VK_KHR_pipeline_binary
#error Vulkan headers must declare VK_KHR_pipeline_binary
#endif
int main() { return 0; }" INNER_HEADERS_HAVE_PIPELINE_BINARY)
unset(CMAKE_REQUIRED_INCLUDES)
if(NOT INNER_HEADERS_HAVE_PIPELINE_BINARY)
  message(FATAL_ERROR "Use newer Vulkan headers declaring VK_KHR_pipeline_binary for this harness")
endif()
add_executable(vk_inner_coverage vk_inner_coverage.cpp)
target_compile_features(vk_inner_coverage PRIVATE cxx_std_20)
target_include_directories(vk_inner_coverage PRIVATE .. "${INNER_VULKAN_INCLUDE}")
target_link_libraries(vk_inner_coverage PRIVATE ${CMAKE_DL_LIBS})
if(WIN32)
  target_compile_definitions(vk_inner_coverage PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

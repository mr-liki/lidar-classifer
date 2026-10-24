cmake_minimum_required(VERSION 3.20)
project(naksha_ground_vulkan_compute LANGUAGES CXX)

# Standalone compute-only DLL: leaves the existing naksha_vulkan renderer DLL
# and the Stage 3/4 ground_ptd_exact DLL unchanged.
find_package(Vulkan REQUIRED)
find_program(NAKSHA_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin32")
if(NOT NAKSHA_GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK shader compiler, or set NAKSHA_GLSLC.")
endif()
set(COMPUTE_SPV "${CMAKE_CURRENT_BINARY_DIR}/ground_distance.comp.spv")
add_custom_command(OUTPUT "${COMPUTE_SPV}"
    COMMAND "${NAKSHA_GLSLC}" --target-env=vulkan1.1
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ground_distance.comp" -o "${COMPUTE_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ground_distance.comp"
    COMMENT "Compiling Ground PTD Vulkan FP64 compute shader"
    VERBATIM)
add_custom_target(naksha_ground_vulkan_shader DEPENDS "${COMPUTE_SPV}")
add_library(naksha_ground_vulkan_compute SHARED ground_vulkan_compute.cpp)
target_compile_features(naksha_ground_vulkan_compute PRIVATE cxx_std_17)
target_link_libraries(naksha_ground_vulkan_compute PRIVATE Vulkan::Vulkan)
if(MSVC)
    target_compile_options(naksha_ground_vulkan_compute PRIVATE /W4 /O2 /fp:strict)
    target_compile_definitions(naksha_ground_vulkan_compute PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(naksha_ground_vulkan_compute PRIVATE -Wall -Wextra -Wpedantic -O2 -fno-fast-math -ffp-contract=off)
endif()
set_target_properties(naksha_ground_vulkan_compute PROPERTIES PREFIX "")
add_dependencies(naksha_ground_vulkan_compute naksha_ground_vulkan_shader)
add_custom_command(TARGET naksha_ground_vulkan_compute POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${COMPUTE_SPV}" "$<TARGET_FILE_DIR:naksha_ground_vulkan_compute>/ground_distance.comp.spv"
    VERBATIM)

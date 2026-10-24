cmake_minimum_required(VERSION 3.20)
project(naksha_ground_vulkan_locator LANGUAGES CXX)
find_package(Vulkan REQUIRED)
find_program(NAKSHA_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin32")
if(NOT NAKSHA_GLSLC)
    message(FATAL_ERROR "glslc not found in Vulkan SDK")
endif()
set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/ground_locate.comp.spv")
add_custom_command(OUTPUT "${SPIRV}"
    COMMAND "${NAKSHA_GLSLC}" --target-env=vulkan1.1
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ground_locate.comp" -o "${SPIRV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ground_locate.comp"
    COMMENT "Compiling Vulkan FP64 Ground simplex lookup shader"
    VERBATIM)
add_custom_target(ground_vulkan_locator_spv DEPENDS "${SPIRV}")
add_library(naksha_ground_vulkan_locator SHARED ground_vulkan_locator.cpp)
target_compile_features(naksha_ground_vulkan_locator PRIVATE cxx_std_17)
target_link_libraries(naksha_ground_vulkan_locator PRIVATE Vulkan::Vulkan)
if(MSVC)
  target_compile_options(naksha_ground_vulkan_locator PRIVATE /W4 /O2 /fp:strict)
  target_compile_definitions(naksha_ground_vulkan_locator PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  target_compile_options(naksha_ground_vulkan_locator PRIVATE -Wall -Wextra -Wpedantic -O2 -fno-fast-math -ffp-contract=off)
endif()
set_target_properties(naksha_ground_vulkan_locator PROPERTIES PREFIX "")
add_dependencies(naksha_ground_vulkan_locator ground_vulkan_locator_spv)
add_custom_command(TARGET naksha_ground_vulkan_locator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SPIRV}"
            "$<TARGET_FILE_DIR:naksha_ground_vulkan_locator>/ground_locate.comp.spv"
    VERBATIM)

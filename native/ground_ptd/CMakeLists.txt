cmake_minimum_required(VERSION 3.18)
project(naksha_ground_ptd_exact LANGUAGES CXX)

add_library(naksha_ground_ptd_exact SHARED ground_ptd_exact.cpp)
target_compile_features(naksha_ground_ptd_exact PRIVATE cxx_std_17)
if(MSVC)
  # No fast-math, implicit FMA contractions, or arithmetic reassociation.
  target_compile_options(naksha_ground_ptd_exact PRIVATE /O2 /fp:strict /W4)
else()
  target_compile_options(naksha_ground_ptd_exact PRIVATE
    -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra -Wpedantic)
endif()
set_target_properties(naksha_ground_ptd_exact PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

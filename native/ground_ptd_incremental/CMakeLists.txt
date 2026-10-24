cmake_minimum_required(VERSION 3.18)
project(naksha_ground_ptd_incremental LANGUAGES CXX)

add_library(naksha_ground_ptd_incremental SHARED ground_ptd_incremental.cpp)
target_compile_features(naksha_ground_ptd_incremental PRIVATE cxx_std_17)
if(MSVC)
  # Bit-exact NumPy replica: no fast-math, FMA contraction or reassociation.
  target_compile_options(naksha_ground_ptd_incremental PRIVATE /O2 /fp:strict /W4 /EHsc)
else()
  target_compile_options(naksha_ground_ptd_incremental PRIVATE
    -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra)
endif()
set_target_properties(naksha_ground_ptd_incremental PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

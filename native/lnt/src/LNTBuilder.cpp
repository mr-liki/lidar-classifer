// LNTBuilder.cpp - the builder as its own translation unit.
//
// This file exists for the split build:
//     g++ -O3 -std=c++17 -DLNT_EXTERN_BUILDER main.cpp src/LNTBuilder.cpp -o main
//
// It is deliberately a one-liner. The implementation lives in
// include/LNTBuilderImpl.h and is pulled in by exactly one TU - here, or
// main.cpp when building header-only - so the definitions never appear twice
// and there is no ODR hazard either way.
//
// LNT_BUILDER_TU removes the `inline` on the definition. Without it the linker
// finds no LNTBuilder::build at all: an inline function that its own TU never
// references is not emitted, so this object file would be empty of it.

#define LNT_BUILDER_TU 1

#include "../include/LNTBuilderImpl.h"

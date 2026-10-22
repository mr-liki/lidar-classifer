@echo off
set "PATH=H:\tools\mingw1310_64\bin;H:\tools\CMake_64\bin;H:\tools\Ninja;H:\tools\Bin;H:\tools\Lib;H:\tools\Include;%PATH%"
set "VULKAN_SDK=H:\tools"
cd /d "%~dp0"
md build 2>nul
cd build
cmake .. -G Ninja -DCMAKE_MAKE_PROGRAM=H:/tools/Ninja/ninja.exe -DCMAKE_C_COMPILER=H:/tools/mingw1310_64/bin/gcc.exe -DCMAKE_CXX_COMPILER=H:/tools/mingw1310_64/bin/g++.exe -DCMAKE_BUILD_TYPE=Release
ninja

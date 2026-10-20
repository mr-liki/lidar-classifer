@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
"H:\tools\CMake_64\bin\cmake.exe" -S native/naksha_vulkan -B native/naksha_vulkan/build_continuity -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_MAKE_PROGRAM=H:/tools/Ninja/ninja.exe -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_WORKS=TRUE -DCMAKE_C_COMPILER_WORKS=TRUE -DCMAKE_CXX_ABI_COMPILED=TRUE -DCMAKE_C_ABI_COMPILED=TRUE
if errorlevel 1 exit /b 1
venv\Scripts\python.exe audit\build_native_commands.py %*

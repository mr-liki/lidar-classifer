@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "H:\new beignning"
cl /nologo /O2 /std:c++17 /EHsc /Fe:audit\tin_policy_probe.exe /Fo:audit\tin_policy_probe.obj audit\tin_policy_probe.cpp
echo EXITCODE=%ERRORLEVEL%

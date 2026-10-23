@echo off
setlocal EnableExtensions
set "SCRIPT=%~dp0convert_las14_to_laz12.py"
set "INPUT=\\192.168.10.6\d\NT895\A13\LAS"
set "OUTPUT=\\192.168.10.6\d\NT895\A13\LAS\LAZ_1_2"
rem Optional: set LASZIP_EXE to your real LASzip executable, e.g.
rem set "LASZIP_EXE=C:\LAStools\bin\laszip64.exe"
set "LASZIP_EXE="
set "PYTHON=E:\lidar-classifer\venv\Scripts\python.exe"
if exist "%PYTHON%" goto PY_OK
set "PYTHON=C:\Users\nt01095\AppData\Local\Programs\Python\Python310\python.exe"
if exist "%PYTHON%" goto PY_OK
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python was not found. Set PYTHON to a valid python.exe in this BAT.
  pause
  exit /b 2
)
set "PYTHON=python"
:PY_OK
echo Source: %INPUT%
echo Output: %OUTPUT%
echo.
if not "%LASZIP_EXE%"=="" (
    "%PYTHON%" "%SCRIPT%" --input "%INPUT%" --output "%OUTPUT%" --laszip "%LASZIP_EXE%"
) else (
    "%PYTHON%" "%SCRIPT%" --input "%INPUT%" --output "%OUTPUT%"
)
set "RESULT=%ERRORLEVEL%"
echo.
echo Script exit code: %RESULT%
pause
exit /b %RESULT%

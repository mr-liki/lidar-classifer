@echo off
setlocal
set "PYTHON=E:\lidar-classifer\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%~dp0convert_las14_to_laz12.py" --scan-only --input "\\192.168.10.6\d\NT895\A13\LAS"
pause

@echo off
setlocal
set "PYTHON=E:\lidar-classifer\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=C:\Users\nt01095\AppData\Local\Programs\Python\Python310\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
echo Converting only FIRST input file as a pilot...
"%PYTHON%" "%~dp0convert_las14_to_laz12.py" --input "\\192.168.10.6\d\NT895\A13\LAS" --limit 1 --deep-verify
pause

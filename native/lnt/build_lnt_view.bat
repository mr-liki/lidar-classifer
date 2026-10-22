@echo off
rem Builds main_display_modes_peaks.dll (the deployed viewer API), the legacy
rem main_shading.dll when it is not locked, and lnt_viewer.exe.
rem Needs g++ (MinGW-w64) on PATH. Run from this folder.
set SRC=LNTReader.cpp src/LNTStreamer.cpp src/LNTLASReader.cpp src/LNTCameraController.cpp src/LNTInteractiveLoop.cpp src/LNTBuilder.cpp
rem Finish both builds before touching the working binaries. If the viewer is
rem running, COPY fails and leaves its loaded DLL intact; close it and retry.
g++ -O3 -std=c++17 -static -shared -DLNT_EXTERN_BUILDER -o main_shading_next.dll lnt_view_api.cpp %SRC% || exit /b 1
g++ -O3 -std=c++17 -static -DLNT_EXTERN_BUILDER -o lnt_viewer_next.exe lnt_viewer.cpp %SRC% || exit /b 1
copy /y main_shading_next.dll main_display_modes_peaks.dll >nul || exit /b 1
copy /y main_shading_next.dll main_display_modes_lighting.dll >nul 2>nul
copy /y main_shading_next.dll main_display_modes.dll >nul 2>nul
rem Existing sessions may still map the earlier filename; update it when unlocked.
copy /y main_shading_next.dll main_display_modes_zoom.dll >nul 2>nul
rem A running older application may still map this legacy filename. Its update
rem is best-effort because new launches load main_display_modes_peaks.dll.
copy /y main_shading_next.dll main_shading.dll >nul 2>nul
copy /y lnt_viewer_next.exe lnt_viewer.exe >nul || exit /b 1
echo built main_display_modes_peaks.dll and lnt_viewer.exe

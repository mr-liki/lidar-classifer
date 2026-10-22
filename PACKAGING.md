# Packaging NakshaAI-LiDAR (Windows x64)

The packaged app is a PyInstaller **onedir** build: `NakshaAI-LiDAR.exe` plus an
`_internal` folder that holds Python, Qt, the native renderers, shaders, AI
models and the plugin payload. Ship the whole `dist\Naksha` folder — the exe
does not work without `_internal`.

## 1. Build environment

```bat
py -3.10 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install pyinstaller==6.22.3
:: GPU AI (CUDA 12.4 wheels; CUDA 12.x minor-version compatibility covers
:: R525+ drivers). CPU-only fallback: drop the --index-url line.
venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
venv\Scripts\python.exe -m pip install snt_core-1.3.1-cp310-cp310-win_amd64.whl
```

The native Vulkan renderer is prebuilt (`native/naksha_vulkan/build_msvc`). To
rebuild it from source you also need VS 2022 C++ build tools + CMake, then:

```bat
powershell -ExecutionPolicy Bypass -File build_native.ps1
```

`build_native.ps1` also writes the build stamp the startup self-check reads.

## 2. Build

```bat
venv\Scripts\python.exe -m PyInstaller --clean --noconfirm naksha.spec
```

Output: `dist\Naksha\NakshaAI-LiDAR.exe` + `dist\Naksha\_internal`.

The exe keeps a console window for runtime diagnostics. For a windowed build:

```bat
set NAKSHA_BUILD_CONSOLE=0
venv\Scripts\python.exe -m PyInstaller --clean --noconfirm naksha.spec
```

Build the offline Windows x64 installer after the bundle passes its smoke test:

```powershell
& 'H:\Inno Setup 7\ISCC.exe' naksha_setup.iss
```

Output: `installer_output\Setup.exe` (version 2.0.5). The full, signed Microsoft
`redist\VC_redist.x64.exe` is mandatory. Use the complete download from
[Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist),
not the stripped bootstrapper in the Windows package cache. No Python, pip,
Vulkan SDK, or Visual Studio installation is required on the target computer.
The target must have a Vulkan-capable GPU and its vendor's graphics driver.

Installed builds default to `NAKSHA_RENDER_BACKEND=vulkan` and
`NAKSHA_VULKAN_MAIN_VIEWPORT=1` in the runtime hook. Explicit diagnostic
environment overrides remain available.

## 3. What the spec bundles

| Payload | Destination in `_internal` | Consumed by |
| --- | --- | --- |
| `native/naksha_vulkan/build_msvc/naksha_vulkan.dll` + 32 `.spv` shaders + `shaders/` | `native/naksha_vulkan/build_msvc` | `gui/render_backend.py` (Vulkan viewport / sections) |
| `native/lnt/main_display_modes_peaks.dll` | `native/lnt` | `gui/lnt_native.py` (LNT streaming viewer) |
| C API header + `native_build_stamp.json` | `native/naksha_vulkan/...` | `main.py` startup self-check |
| `vulkan-1.dll` (Vulkan SDK, else `%SystemRoot%\System32`) | `_internal` and next to the native DLL | volk → `vkCreateInstance` |
| `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`, `concrt140.dll` | next to the native DLL | MSVC-built renderer |
| `plugins/` (no `__pycache__`, no `.install-cache`) | `plugins/` (**copied to `%LOCALAPPDATA%\NakshaAI-LiDAR\plugins` on first launch**) | `gui/plugin_manager.py` |
| `gui/AI/**` (models, engine `.py`, vehicle runners) | `gui/AI` | Basic/Advanced/Premium/Vehicle AI |
| `models/`, `Advance_Model/`, `assets/` | same names | AI + GIS icons |
| Qt, VTK, Open3D, laspy, numpy/scipy, torch | PyInstaller hooks / `collect_all` | app + AI |

Shaders are loaded from the renderer DLL's own directory, so the `.spv` files
must stay beside `naksha_vulkan.dll`.

## 4. Frozen-runtime behaviour

- `PROJECT_ROOT` = `sys._MEIPASS` (`...\_internal`), so every relative asset
  path resolves unchanged.
- `main.py` never re-execs into `venv\Scripts\python.exe` when `sys.frozen`.
- Plugins live in `%LOCALAPPDATA%\NakshaAI-LiDAR\plugins`. `gui/plugin_seed.py`
  copies the bundled payload once and re-copies when a bundled plugin's
  `manifest.json` version changes.
- Premium/vehicle engines run as child processes. With no `python.exe` beside
  the exe, `sys.executable --run-python-script <engine.py>` re-enters `main.py`
  and runs the script in the frozen interpreter; `multiprocessing.freeze_support()`
  is called first so spawn workers can start.
- The native build stamp is reported, not re-derived, in frozen builds (native
  sources are not shipped). The DLL export-surface check still runs.

## 5. Smoke test of a built package

Any script can be executed inside the frozen interpreter (this is the same hook
Premium/vehicle use):

```bat
dist\Naksha\NakshaAI-LiDAR.exe --run-python-script some_probe.py
dist\Naksha\NakshaAI-LiDAR.exe
```

The probe should print `frozen: True` with a `_MEIPASS` under `_internal`, and
`import torch` must report `cuda available: True` on a CUDA machine.

The GUI log (stdout) must contain `Status : VULKAN READY`, `Plugin: 'Naksha
Converter' loaded.` and `[NAKSHA RUNTIME SELF-CHECK] PASS`. Start it from a
folder other than the build tree once to confirm the package is relocatable.

The offline dependency and GPU probe is:

```bat
dist\Naksha\NakshaAI-LiDAR.exe --run-python-script audit\frozen_installer_smoke.py audit\frozen_installer_smoke.json
```

The probe checks that imports resolve inside the bundle, loads the LNT engine,
and renders a steep class-coloured triangle through Vulkan.

The 2026-10-22 LOD update refines from resident parent tiles and retains resident
siblings during refinement. Fresh tiles can render while an untrusted archive's
source fingerprint is checked in the background. GPU acceptance covers a 4-million-point fixture,
32 moving viewports, bounded-cache eviction, persistence, class edits and bad
leaf fallback. Read-only 47,233,034-point and 634,108,274-point sessions are also exercised. These
checks do not establish performance for a real 1-2-billion-point dataset.

## 6. Known limits

- Premium AI's spawn-based PCA workers are wired but not yet validated against a
  packaged run; the engine needs `torch` in the build environment.
- The NVIDIA T400 has 4 GB VRAM — large clouds still fall back to CPU paths.
- `dist` with CUDA torch is several GB; keep the build machine's temp folder on
  a drive with room (pip's temp copy alone needs ~3 GB).

"""Execute CMake-generated native commands when Ninja completion stalls."""
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
build = root / 'native/naksha_vulkan/build_continuity'
plan = subprocess.check_output([r'H:/tools/Ninja/ninja.exe', '-t', 'commands', 'naksha_vulkan'], cwd=build, text=True)
commands = plan.splitlines()
for i, cmd in enumerate(commands, 1):
    if '--incremental-edit' in __import__('sys').argv:
        if not any(key in cmd for key in ('SurfaceRenderer.cpp', 'naksha_vulkan_c_api.cpp', 'lib.exe', 'link.exe')):
            continue
    print(f'Native build command {i}/{len(commands)}', flush=True)
    result = subprocess.run(cmd, cwd=build, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout); print(result.stderr)
        raise SystemExit(result.returncode)
print('Native build succeeded', flush=True)

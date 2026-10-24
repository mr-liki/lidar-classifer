"""Serial real GPU checks and bounded regression-harness diagnostics."""
from pathlib import Path
import json
import os
import subprocess
import sys
from time import perf_counter
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'audit/ground_incremental_resume/hardware_and_harness'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               QT_QPA_PLATFORM='offscreen', NAKSHA_GROUND_INCREMENTAL='0',
               NAKSHA_GROUND_GPU='0', NAKSHA_GROUND_ENGINE_V2='0',
               NAKSHA_GROUND_VK_TIN='0', NAKSHA_GROUND_VK_LOCATOR='0',
               NAKSHA_GROUND_VULKAN_COMPUTE='0')
    jobs = [
        ('stage3_correct_dll', ['-m','pytest','-q','tests/test_ground_ptd_stage3_grid.py',
            '--tb=short','--junitxml='+str(OUT/'stage3_correct_dll.xml')],
         dict(NAKSHA_GROUND_CPP_LIBRARY=str(ROOT/'native/ground_ptd/build_stage3/Release/naksha_ground_ptd_exact.dll'))),
        ('main_view_utf8', ['-m','pytest','-q','tests/test_main_view_2d_policy.py',
            '--tb=short','--junitxml='+str(OUT/'main_view_utf8.xml')], {}),
        ('peer_cancel_with_qapp', ['-c', 'from PySide6.QtWidgets import QApplication; '
            'qapp=QApplication([]); import pytest; raise SystemExit(pytest.main('
            +repr(['-q','tests/test_cross_section_peer_classification_cancel.py','--tb=short',
                   '--junitxml='+str(OUT/'peer_cancel_with_qapp.xml')])+'))'], {}),
    ]
    for name in ('validate_ground_vulkan_runtime', 'validate_ground_vulkan_tin_runtime',
                 'validate_ground_vulkan_locator_runtime'):
        jobs.append((name, ['-c', 'import runpy; runpy.run_path('
            +repr(str(ROOT/'scripts'/(name+'.py')))+", run_name='__main__')"], {}))
    rows = []
    for name, args, overrides in jobs:
        log = OUT/(name+'.log')
        if log.exists():
            raise ValueError('Prior output exists: '+str(log))
        started = perf_counter()
        with log.open('wb') as handle:
            proc = subprocess.Popen([sys.executable, *args], cwd=ROOT,
                env=dict(env, **overrides), stdout=handle, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                for child in reversed(psutil.Process(proc.pid).children(recursive=True)):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                proc.kill(); proc.wait(timeout=10)
                code = 'TIMEOUT_90_SECONDS'
        row = dict(name=name, exit_code=code, seconds=perf_counter()-started,
                   environment_overrides=overrides)
        rows.append(row)
        (OUT/'summary.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

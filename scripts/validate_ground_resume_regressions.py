"""Isolated, bounded regressions; compare failures with saved baseline evidence."""
from pathlib import Path
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit/ground_incremental_resume/regressions'


def failures(path):
    if not path.exists():
        return None
    return sorted((t.attrib.get('name'), e.tag)
                  for t in ET.parse(path).iter('testcase')
                  for e in t if e.tag in ('failure', 'error'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((ROOT/'audit/ground_ptd_v2/regression_comparison.json').read_text())
    old = {Path(r['file'].replace('\\', '/')).name: r for r in previous['existing_failures']}
    files = sorted(set(ROOT.glob('tests/test_ground*.py')) - {
        ROOT/'tests/test_ground_ptd_incremental.py', ROOT/'tests/test_ground_ptd_engine_v2.py',
        ROOT/'tests/test_ground_parity.py'})
    files += [ROOT/'tests'/name for name in old]
    files += [ROOT/'tests'/name for name in (
        'test_cross_section_peer_classification_cancel.py', 'test_stage1_parity.py',
        'test_stage2_parity.py', 'test_stale_vtk_dolly_reset.py', 'test_native_backup.py',
        'test_lidar_classification_tools.py') if name not in old]
    runs = []
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', NAKSHA_GROUND_INCREMENTAL='0',
               NAKSHA_GROUND_GPU='0', NAKSHA_GROUND_ENGINE_V2='0')
    for file in files:
        xml = OUT/(file.stem+'.xml')
        log = OUT/(file.stem+'.log')
        if xml.exists() or log.exists():
            raise ValueError('Prior regression output exists; do not overwrite: '+str(file))
        cmd = [sys.executable, '-m', 'pytest', '-q', str(file), '--disable-warnings',
               '--tb=short', '--junitxml='+str(xml)]
        # Archived baseline already demonstrated this modal picker hang.
        # Preserve that evidence; run all other tests in the file.
        if file.name == 'test_lidar_classification_tools.py':
            cmd += ['-k', 'not test_surface_fence_picker_is_visible_and_accepts_restored_point_vertices']
        with log.open('w', encoding='utf-8') as handle:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=90)
                status = 'PASS' if process.returncode == 0 else 'FAIL'
            except subprocess.TimeoutExpired:
                status = 'TIMEOUT_90_SECONDS'
                # Only this test subprocess and its own children are stopped.
                owned = psutil.Process(process.pid).children(recursive=True)
                handle.write('\nTimed-out test child PIDs: '+str([p.pid for p in owned])+'\n')
                handle.flush()
                for child in reversed(owned):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait(timeout=10)
        row = dict(file=str(file.relative_to(ROOT)), status=status, failed_tests=failures(xml))
        if file.name in old:
            row['saved_baseline_failed_tests'] = sorted((t['test'], t['type'])
                for t in old[file.name]['baseline_failures'])
            row['same_failure_ids_as_saved_baseline'] = row['failed_tests'] == row['saved_baseline_failed_tests']
        runs.append(row)
        (OUT/'summary.json').write_text(json.dumps(runs, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

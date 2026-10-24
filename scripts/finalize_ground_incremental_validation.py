"""Finish authorized validation serially, preserving every prior artifact.

Waits only for this run's recorded native subprocess exit. Then resumes exact
original checkpoints and compares complete plans/contexts and packed LNT data.
Never starts a competing benchmark or touches the running Naksha application.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'audit/ground_incremental_resume'


def main():
    status_file = OUT/'full-native_verified_process.json'
    deadline = time.monotonic()+7200
    while not status_file.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Native validation did not record completion within two hours.')
        time.sleep(10)
    process = json.loads(status_file.read_text())
    assert process['exit_code'] == 0, process
    native_dir = OUT/'full_native_verified'
    native = json.loads((native_dir/'result.json').read_text())
    assert native['status'] == 'COMPLETE' and not native['resumed']
    assert native['source_sha256_before'] == native['source_sha256_after']
    assert native['refused_native_iterations'] == 0 and not native['audit_failures']
    print('Native measurement complete. Starting original-reference resume serially.', flush=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    result = subprocess.run([sys.executable, str(ROOT/'scripts/validate_ground_incremental_resume.py'),
        'resume-original', '--record-process'], cwd=ROOT, env=env)
    assert result.returncode == 0, 'Original-reference resume failed; see its process log.'
    original_dir = OUT/'full_original_resumed'
    original = json.loads((original_dir/'result.json').read_text())
    assert original['status'] == 'COMPLETE'
    assert original['resumed'] and original['original_reference_resume']['checkpoint_hits'] > 0
    assert native['params'] == original['params']
    assert native['source_sha256_before'] == original['source_sha256_before']
    assert original['source_sha256_before'] == original['source_sha256_after']
    comparison = OUT/'full_original_native_comparison.json'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/compare_ground_ptd_v2_full.py'),
        '--v1', str(original_dir), '--v2', str(native_dir), '--output', str(comparison)],
        cwd=ROOT, env=env)
    assert comparison.exists(), 'Exact plan/context comparator did not produce evidence.'
    report = json.loads(comparison.read_text())
    report['original_timing_is_resumed_accuracy_only'] = True
    report['native_performance_600_seconds'] = native['performance_600_seconds']
    report['native_full_operation_seconds'] = native['full_operation_seconds']
    report['persistent_tin_updates'] = 0
    from compare_ground_lnt_classes import compare_lnt
    report['packed_lnt_comparison'] = compare_lnt(
        original_dir/'MARCENA_validation_copy.lnt', native_dir/'MARCENA_validation_copy.lnt')
    # If an identity/class difference occurs, retain complete exact plan data,
    # not just a count or an example. Both original plans remain untouched.
    if not report['accuracy_pass']:
        import numpy as np
        a, b = [dict(np.load(p/'plan.npz', allow_pickle=False)) for p in (original_dir, native_dir)]
        ka = (a['node'].astype(np.uint64)<<np.uint64(32)) | a['local']
        kb = (b['node'].astype(np.uint64)<<np.uint64(32)) | b['local']
        common, i, j = np.intersect1d(ka, kb, return_indices=True)
        bad = a['new_classes'][i] != b['new_classes'][j]
        np.savez(OUT/'exact_plan_differences.npz',
            missing_ids=np.setdiff1d(ka,kb), extra_ids=np.setdiff1d(kb,ka),
            differing_common_ids=common[bad], original_classes=a['new_classes'][i[bad]],
            native_classes=b['new_classes'][j[bad]])
    report['production_ready'] = False
    report['production_blockers'] = [
        'Persistent reference-compatible incremental TIN is not implemented; changed TINs rebuild.',
        'Broader saved-baseline regressions and full visual VTK/Vulkan parity remain unresolved.']
    if not native['performance_600_seconds']:
        report['production_blockers'].append('Full native runtime exceeds the 600-second target.')
    report['status'] = 'COMPLETE' if result.returncode == 0 and report['packed_lnt_comparison']['identical'] else 'ACCURACY_FAILURE'
    (OUT/'validation_outcome.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if report['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

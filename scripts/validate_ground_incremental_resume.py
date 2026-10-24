"""Serial validation adapter; preserves all earlier V2 scripts and artifacts.

Explicitly selects the new native evaluator, never the default-on stale DLL.
Full runs reuse the existing copy/commit/readback harness, not old checkpoints.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
BASELINE = ROOT.parent / 'Naksha_Ground_Incremental_Backup_20261024/gui/lidar_classification_tools.py'
OUT = ROOT / 'audit/ground_incremental_resume'
SOURCE = Path(r'E:\TEST FOLDER VULKAN\NT-298\DX2033391 MARCENA\MARCENA000018.lnt')


def settings(enabled, audit=0):
    return dict(NAKSHA_GROUND_INCREMENTAL=str(enabled), NAKSHA_GROUND_INC_AUDIT=str(audit),
        NAKSHA_GROUND_ENGINE_V2='0', NAKSHA_GROUND_CPP='0', NAKSHA_GROUND_GPU='0',
        NAKSHA_GROUND_VULKAN_COMPUTE='0', NAKSHA_GROUND_VK_TIN='0',
        NAKSHA_GROUND_VK_LOCATOR='0', NAKSHA_GROUND_LOCATOR='0',
        NAKSHA_GROUND_EARLY_REJECT='0', NAKSHA_GROUND_PREFLIGHT='0',
        NAKSHA_GROUND_DENSE_ADAPTIVE='0', NAKSHA_GROUND_DENSE_PREFER_LARGE='0',
        NAKSHA_GROUND_BLAS_SINGLE=str(enabled), NAKSHA_GROUND_TIN_REUSE=str(enabled),
        NAKSHA_GROUND_PROFILE='1')


def original_function(tools, source=BASELINE):
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'classify_ground_ptd')
    namespace = dict(tools.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['classify_ground_ptd']


def tile():
    import numpy as np
    from gui import lidar_classification_tools as tools, ground_ptd_incremental as inc
    from benchmark_ground_ptd_v2 import environment, Meter
    data = np.load(OUT / 'tile_1634.npz', allow_pickle=False)
    xyz, initial = data['xyz'], data['cls0']
    params = dict(from_classes=[0], to_class=1, current_ground=[1],
                  terrascan_compatible_densification=True)
    baseline = original_function(tools)
    report = dict(points=len(xyz), params=params, runs=[], status='RUNNING',
                  baseline_source_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest())
    reference = None
    sampler = tools._subsample_ground_tin_points
    for label, enabled, audit in [('original', 0, 0), ('native_raw', 1, 0), ('native_audited', 1, 1)]:
        classes, anchors, messages = initial.copy(), [], []
        def sample(points, ids, *a, **kw):
            anchors.append(hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest())
            return sampler(points, ids, *a, **kw)
        function = baseline if not enabled else tools.classify_ground_ptd
        profile = OUT / (label + '_tile_profile.jsonl')
        if profile.exists():
            raise ValueError('Refusing to append a completed tile profile: ' + str(profile))
        inc.AUDIT_FAILURES.clear()
        baseline.__globals__['_subsample_ground_tin_points'] = sample
        env = settings(enabled, audit)
        env['NAKSHA_GROUND_PROFILE_FILE'] = str(profile)
        with environment(**env), patch.object(tools, '_subsample_ground_tin_points', sample), Meter() as meter:
            result = function(xyz, classes, **params,
                progress_cb=lambda pct, msg: messages.append((pct, msg)))
        baseline.__globals__['_subsample_ground_tin_points'] = sampler
        assert not inc.AUDIT_FAILURES, inc.AUDIT_FAILURES
        np.savez(OUT / (label + '_tile_result.npz'), classes=classes, indices=result['indices'])
        run = dict(label=label, audit=bool(audit), **meter.metrics,
                   changed=result['changed'], iterations=result['iterations'],
                   audit_failures=list(inc.AUDIT_FAILURES))
        if reference is None:
            reference = (classes, result, anchors, messages)
        else:
            run['class_differences'] = int(np.count_nonzero(classes != reference[0]))
            run['point_id_symmetric_difference'] = int(len(np.setxor1d(result['indices'], reference[1]['indices'])))
            assert run['class_differences'] == run['point_id_symmetric_difference'] == 0
            for key in result:
                np.testing.assert_equal(result[key], reference[1][key], err_msg=key)
            assert anchors == reference[2]
            assert messages == reference[3]
            run.update(exact_result_fields=True, exact_ordered_ids=True, exact_anchor_traces=True)
            payload = json.loads(profile.read_text().splitlines()[-1])
            stats = payload['backend']['incremental']
            assert stats['queries'] > 0 and stats['cache_cert'] == stats['grid_cert'] == 0
            assert bool(stats['audit_checks']) == bool(audit)
            run['native_stats'] = stats
        report['runs'].append(run)
        (OUT / 'tile_differential.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(run), flush=True)
    report['native_speedup'] = report['runs'][0]['wall_seconds'] / report['runs'][1]['wall_seconds']
    report['status'] = 'COMPLETE'
    (OUT / 'tile_differential.json').write_text(json.dumps(report, indent=2))


def full(enabled, resume_original=False):
    import validate_ground_ptd_v2_full as harness
    from gui import ground_ptd_incremental as inc
    from threading import Lock
    original_environment, original_dump = harness.environment, harness.dump
    output = OUT / ('full_original_resumed' if resume_original else
                    'full_native_verified' if enabled else 'full_original')
    if (output / 'progress.json').exists():
        raise ValueError('Full output already exists; inspect before resuming.')
    profile = output / 'profile.jsonl'
    def environment(**kw):
        kw.update(settings(enabled))
        kw['NAKSHA_GROUND_PROFILE_FILE'] = str(profile)
        return original_environment(**kw)
    code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [ROOT/'gui/ground_ptd_incremental.py',
                            ROOT/'gui/lidar_classification_tools.py',
                            ROOT/'gui/native_ground_below_stream.py',
                            ROOT/'native/ground_ptd_incremental/build_verified/Release/naksha_ground_ptd_incremental.dll']}
    counters = dict(attempted_native_iterations=0, refused_native_iterations=0)
    counter_lock = Lock()
    legacy = None
    if resume_original:
        from probe_ground_reference_checkpoints import input_key
        proof = json.loads((OUT/'original_checkpoint_proof.json').read_text())
        assert proof['status'] == 'VERIFIED'
        matches = [c for c in proof['candidates'] if c['exact_match']]
        assert len(matches) == 1
        scheme = matches[0]
        metadata = json.loads((ROOT/'audit/ground_ptd_v2/full_v1/progress.json').read_text())
        archive = ROOT.parent/'Naksha_Ground_V2_Backup_20261023/gui/lidar_classification_tools.py'
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata['baseline_source_sha256']
        reference = original_function(harness.tools, source=archive)
        legacy = dict(checkpoint_hits=0, freshly_computed_calls=0, verified_input_keys=[],
            baseline_source_sha256=metadata['baseline_source_sha256'],
            archived_code_sha256=scheme['code_sha256'],
            historical_key_binds_baseline_source=False,
            baseline_source_verified_separately=True)
        allowed_keys = {c['input_key'] for c in metadata['cells']}
        def cached_original(**kw):
            key = input_key(kw, metadata, scheme['code_sha256'],
                include_baseline=scheme['include_baseline'], include_layout=scheme['include_layout'])
            saved = ROOT/'audit/ground_ptd_v2/full_v1/checkpoints'/(key+'.npz')
            if key in allowed_keys and saved.exists():
                with harness.np.load(saved, allow_pickle=False) as cached:
                    kw['classification'][:] = cached['classification']
                    result = json.loads(str(cached['result']))
                    result['indices'] = cached['indices'].copy()
                with counter_lock:
                    legacy['checkpoint_hits'] += 1
                    legacy['verified_input_keys'].append(key)
                print('VERIFIED ORIGINAL CHECKPOINT '+key, flush=True)
                return result
            reference.__globals__['Delaunay'] = harness.tools.Delaunay
            with counter_lock:
                legacy['freshly_computed_calls'] += 1
            return reference(**kw)
    real_run = inc.IncrementalPTD.run
    def native_run(self, *a, **kw):
        result = real_run(self, *a, **kw)
        with counter_lock:
            counters['attempted_native_iterations'] += 1
            counters['refused_native_iterations'] += int(result is None)
        return result
    def dump(path, value):
        value.update(validation_backend='native_ordered_evaluation' if enabled else 'original_scipy_numpy',
                     validation_code_sha256=code_hashes, performance_target_seconds=600,
                     persistent_tin_updates=0, topology_mode='reference_qhull_rebuild')
        with counter_lock:
            value.update(counters)
        value['audit_failures'] = list(inc.AUDIT_FAILURES)
        if legacy is not None:
            assert value['source_sha256_before'] == metadata['source_sha256_before']
            import copy
            with counter_lock:
                value['original_reference_resume'] = copy.deepcopy(legacy)
            value['resumed'] = bool(legacy['checkpoint_hits'])
            value['engine'] = '0'
            value['v2_import_seconds'] = 0.
        if profile.exists():
            payloads = [json.loads(line) for line in profile.read_text().splitlines() if line.strip()]
            native = [p['backend'].get('incremental') or {} for p in payloads]
            value['native_component_totals'] = {key: sum(p.get(key, 0) for p in native)
                for key in ('calls', 'queries', 'transform_s', 'map_s', 'locate_s',
                            'evaluate_s', 'select_s', 'cache_cert', 'grid_cert', 'audit_checks')}
            value['profiled_ptd_calls'] = len(payloads)
        if 'full_operation_seconds' in value:
            value['performance_600_seconds'] = value['full_operation_seconds'] <= 600 and not value['resumed']
        return original_dump(path, value)
    sys.argv = [str(Path(harness.__file__)), '--source', str(SOURCE), '--output', str(output),
                '--engine', '1' if resume_original else str(enabled), '--workers', '2',
                '--baseline-source', str(BASELINE)]
    # Harness uses fresh copies and original region/context/worker rules.
    from contextlib import nullcontext
    select_function = patch.object(harness.tools, 'classify_ground_ptd', cached_original) if resume_original else nullcontext()
    with select_function, patch.object(harness, 'environment', environment), patch.object(harness, 'dump', dump), patch.object(
            inc.IncrementalPTD, 'run', native_run):
        harness.main()


def real_io():
    import validate_ground_ptd_v2_real_io as harness
    from benchmark_ground_ptd_v2 import environment
    saved_environment = harness.environment
    def select(**kw):
        # The legacy adapter sets ENGINE_V2=0 around its original comparison.
        enabled = int(str(kw.get('NAKSHA_GROUND_ENGINE_V2', '1')) != '0')
        kw.update(settings(enabled, audit=enabled))
        return saved_environment(**kw)
    output = OUT / 'real_native'
    if (output / 'result.json').exists():
        raise ValueError('Real IO output already exists.')
    sys.argv = [str(Path(harness.__file__)), '--source', str(SOURCE), '--output', str(output),
                '--baseline-source', str(BASELINE), '--export']
    with environment(**settings(1, audit=1)), patch.object(harness, 'environment', select):
        harness.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['tile', 'full-native', 'full-original', 'resume-original', 'real-io'])
    parser.add_argument('--record-process', action='store_true',
                        help='Capture actual child exit status, avoiding PowerShell stderr translation.')
    args = parser.parse_args()
    mode = args.mode
    if args.record_process:
        import subprocess
        process_label = mode + ('_verified' if mode == 'full-native' else '')
        log = OUT/(process_label+'_process.log')
        if log.exists():
            raise ValueError('Process log exists: '+str(log))
        started = perf_counter()
        with log.open('wb') as handle:
            process = subprocess.Popen([sys.executable, __file__, mode], cwd=ROOT,
                                       stdout=handle, stderr=subprocess.STDOUT)
            code = process.wait()
        status = dict(mode=mode, child_pid=process.pid, exit_code=code,
                      process_seconds=perf_counter()-started)
        (OUT/(process_label+'_process.json')).write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
        raise SystemExit(code)
    if mode == 'tile':
        tile()
    elif mode == 'real-io':
        real_io()
    elif mode == 'resume-original':
        full(0, resume_original=True)
    else:
        full(int(mode == 'full-native'))

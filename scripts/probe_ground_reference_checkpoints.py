"""Read-only proof of an archived original checkpoint's exact input hash.

No checkpoint is trusted by bounds alone. Recognizes only a complete SHA256
over source hash, archived code hash, original function hash, ordered arrays,
array metadata and parameters. Reports the matching historical key scheme.
"""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def input_key(kw, metadata, code_hash, runtime=None, include_baseline=True, include_layout=True):
    value = hashlib.sha256()
    prefix = metadata['source_sha256_before'] + code_hash + '0'
    if runtime is not None:
        prefix += json.dumps(runtime, sort_keys=True)
    if include_baseline:
        prefix += metadata['baseline_source_sha256']
    value.update(prefix.encode())
    import numpy as np
    for key in ('xyz', 'classification', 'fence_mask'):
        array = kw.get(key)
        if array is not None:
            if include_layout:
                value.update(str((key, array.shape, array.dtype)).encode())
            value.update(np.ascontiguousarray(array).tobytes())
    value.update(json.dumps(metadata['params'], sort_keys=True).encode())
    return value.hexdigest()


def main():
    from gui import native_ground_below_stream as stream
    from gui.native_lnt_directory import inspect_directory
    metadata = json.loads((ROOT/'audit/ground_ptd_v2/full_v1/progress.json').read_text())
    source = Path(metadata['source'])
    info = inspect_directory(source, max_points=2**63-1)
    cell = metadata['cells'][0]
    tile = stream._read_window(source, info, cell['bounds'], allowed_classes={0,1},
                               max_context=2_000_000)
    kw = dict(xyz=tile['xyz'], classification=tile['classification'], fence_mask=None)
    candidates = []
    for folder in ('Naksha_Ground_V2_Resume_20261024', 'Naksha_Ground_V2_Backup_20261023',
                   'Naksha_Ground_Incremental_Backup_20261024'):
        saved = ROOT.parent/folder
        engine, function = saved/'gui/ground_ptd_engine_v2.py', saved/'gui/lidar_classification_tools.py'
        if engine.exists() and function.exists():
            for scheme, code in [('engine+caller', engine.read_bytes()+function.read_bytes()),
                                 ('engine_only', engine.read_bytes()),
                                 ('caller_only', function.read_bytes()),
                                 ('caller+engine', function.read_bytes()+engine.read_bytes())]:
                code_hash = hashlib.sha256(code).hexdigest()
                for include_baseline in (True, False):
                    for include_layout in (True, False):
                        key = input_key(kw, metadata, code_hash, include_baseline=include_baseline,
                                        include_layout=include_layout)
                        candidates.append(dict(archive=str(saved), code_scheme=scheme, code_sha256=code_hash,
                            include_baseline=include_baseline, include_layout=include_layout,
                            computed_key=key, expected_key=cell['input_key'], exact_match=key==cell['input_key']))
    report = dict(source=str(source), points=len(tile['xyz']), candidates=candidates,
                  matching_key_scheme='see exact matching candidate include_baseline/include_layout fields',
                  status='VERIFIED' if any(c['exact_match'] for c in candidates) else 'UNVERIFIED')
    (ROOT/'audit/ground_incremental_resume/original_checkpoint_proof.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report['status'] != 'VERIFIED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

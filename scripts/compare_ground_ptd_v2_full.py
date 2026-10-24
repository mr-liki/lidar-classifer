"""Compare completed full-run plans by exact point identity and class bytes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--v2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    paths = [args.v1/'result.json', args.v2/'result.json']
    while not all(p.exists() for p in paths):
        if not args.wait:
            raise ValueError('Both complete result.json files are required.')
        time.sleep(5)
    reports = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    assert reports[0]['source_sha256_before'] == reports[1]['source_sha256_before']
    for report in reports:
        assert report['source_sha256_before'] == report['source_sha256_after']
        assert report['status'] == 'COMPLETE'
        assert report['readback_class_differences'] == 0
    plans = [dict(np.load(p/'plan.npz', allow_pickle=False)) for p in (args.v1, args.v2)]
    keys = [(p['node'].astype(np.uint64) << np.uint64(32)) | p['local'] for p in plans]
    for key in keys:
        assert len(np.unique(key)) == len(key), 'Duplicate planned writes'
    common, i, j = np.intersect1d(*keys, assume_unique=True, return_indices=True)
    missing = np.setdiff1d(keys[0], keys[1], assume_unique=True)
    extra = np.setdiff1d(keys[1], keys[0], assume_unique=True)
    old_diff = int(np.count_nonzero(plans[0]['old_classes'][i] != plans[1]['old_classes'][j]))
    new_diff = int(np.count_nonzero(plans[0]['new_classes'][i] != plans[1]['new_classes'][j]))
    result = dict(total_points=reports[0]['total_points'], v1_changes=len(keys[0]),
        v2_changes=len(keys[1]), missing_changed_point_ids=len(missing),
        extra_changed_point_ids=len(extra), old_class_differences=old_diff,
        destination_class_differences_on_common_ids=new_diff,
        final_per_point_class_differences=len(missing)+len(extra)+new_diff,
        planned_order_equal=bool(np.array_equal(*keys)), context_differences=[],
        v1_full_operation_seconds=reports[0]['full_operation_seconds'],
        v2_full_operation_seconds=reports[1]['full_operation_seconds'],
        v2_performance_360_seconds=reports[1]['performance_360_seconds'])
    contexts = [{tuple(c['bounds']): c for c in r['cells']} for r in reports]
    assert contexts[0].keys() == contexts[1].keys(), 'Different contexts were processed'
    for bounds, left in contexts[0].items():
        right = contexts[1][bounds]
        with np.load(args.v1/'checkpoints'/(left['input_key']+'.npz'), allow_pickle=False) as a, np.load(
                args.v2/'checkpoints'/(right['input_key']+'.npz'), allow_pickle=False) as b:
            differences = int(np.count_nonzero(a['classification'] != b['classification']))
            fields_equal = json.loads(str(a['result'])) == json.loads(str(b['result']))
            indices_equal = np.array_equal(a['indices'], b['indices'])
        if differences or not fields_equal or not indices_equal:
            result['context_differences'].append(dict(bounds=bounds,
                class_differences=differences, result_fields_equal=fields_equal,
                exact_changed_indices=bool(indices_equal)))
    result['contexts_compared'] = len(contexts[0])
    result['accuracy_pass'] = not (result['final_per_point_class_differences'] or old_diff
                                   or result['context_differences'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['accuracy_pass'] else 1)


if __name__ == '__main__':
    main()

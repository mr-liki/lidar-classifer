"""Compare V2 directly with the SAVED pre-edit function, loaded in memory."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_ground_ptd_v2 import synthetic, environment
from gui import lidar_classification_tools as tools
from gui import native_ground_below_stream as stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lnt', type=Path)
    args = parser.parse_args()
    source = args.baseline_source.read_bytes()
    tree = ast.parse(source.decode('utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'classify_ground_ptd')
    namespace = dict(tools.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.baseline_source), 'exec'), namespace)
    baseline = namespace['classify_ground_ptd']
    cases = [('synthetic', *synthetic(8000))]
    if args.lnt:
        info = stream.inspect_directory(args.lnt, max_points=2**63-1)
        core = stream._regions(info, [], 110.)[81]
        cx, cy = (core[0]+core[2])/2, (core[1]+core[3])/2
        tile = stream._read_window(args.lnt, info, (cx-35, cy-35, cx+35, cy+35),
                                   allowed_classes={0, 1})
        cases.append(('MARCENA_cell81_window', tile['xyz'], tile['classification']))
    report = dict(baseline_sha256=hashlib.sha256(source).hexdigest(), cases=[])
    for name, xyz, original in cases:
        values, results, times = [], [], []
        for fn, enabled in ((baseline, '0'), (tools.classify_ground_ptd, '1')):
            cls = original.copy()
            with environment(NAKSHA_GROUND_ENGINE_V2=enabled, NAKSHA_GROUND_GPU=0,
                             NAKSHA_GROUND_LOCATOR=0):
                started = perf_counter()
                result = fn(xyz, cls, from_classes=[0], to_class=1, current_ground=[1],
                            terrascan_compatible_densification=True)
                times.append(perf_counter()-started)
            values.append(cls); results.append(result)
        np.testing.assert_array_equal(*values)
        for key in results[0]:
            np.testing.assert_equal(results[0][key], results[1][key], err_msg=key)
        report['cases'].append(dict(name=name, points=len(original), class_differences=0,
            all_result_fields_equal=True, iterations=results[0]['iterations'],
            baseline_seconds=times[0], v2_seconds=times[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Run existing regression files in isolated processes with visible timeouts.

No failures are excluded: logs, JUnit reports and timeout/error statuses are
preserved per file. Isolation also prevents one modal Qt test stalling all
remaining regression coverage on Windows.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'audit/ground_ptd_v2/regressions')
    parser.add_argument('--timeout', type=float, default=90.)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--files', nargs='*')
    parser.add_argument('--baseline-source', type=Path,
                        help='Execute the saved pre-edit Ground module in memory, without changing any source file')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', NAKSHA_GROUND_GPU='0',
               NAKSHA_GROUND_LOCATOR='0', NAKSHA_GROUND_ENGINE_V2='1',
               NAKSHA_GROUND_WORKERS='2', PYTHONIOENCODING='utf-8')
    files = [ROOT / x for x in args.files] if args.files else sorted((ROOT / 'tests').glob('test_*.py'))
    def run(path):
        started = perf_counter()
        command = [sys.executable, '-m', 'pytest', '-q', str(path), '--tb=short',
                   '--disable-warnings', '-o', 'faulthandler_timeout=45',
                   f'--junitxml={args.output / (path.stem + ".xml")}']
        if args.baseline_source:
            bootstrap = (
                'import sys; from pathlib import Path; '
                'from gui import lidar_classification_tools as module; '
                'exec(compile(Path(sys.argv[1]).read_text(encoding="utf-8"), '
                'module.__file__, "exec"), module.__dict__); '
                'import pytest; raise SystemExit(pytest.main(sys.argv[2:]))')
            command = [sys.executable, '-c', bootstrap, str(args.baseline_source), *command[3:]]
        try:
            process = subprocess.run(command, cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
            code, output = process.returncode, process.stdout
            status = 'PASS' if code == 0 else 'FAIL'
        except subprocess.TimeoutExpired as exc:
            code, output, status = None, exc.stdout or b'', 'TIMEOUT'
        log = args.output / (path.stem + '.log')
        log.write_bytes(output)
        return dict(file=str(path.relative_to(ROOT)), status=status,
                    returncode=code, seconds=perf_counter()-started, log=str(log))
    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(run, path) for path in files]
        for future in as_completed(pending):
            record = future.result()
            records.append(record)
            print(f'{record["status"]}: {record["file"]} ({record["seconds"]:.1f}s)', flush=True)
            (args.output / 'summary.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    raise SystemExit(0 if all(x['status'] == 'PASS' for x in records) else 1)


if __name__ == '__main__':
    main()

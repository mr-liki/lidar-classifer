"""Finalize the handoff after the asynchronous full validators finish."""
from pathlib import Path
import difflib
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKUP = ROOT.parent / 'Naksha_Ground_V2_Backup_20261023'


def load(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


while not (HERE/'full_comparison.json').exists() or not (HERE/'real_simplex_audit.json').exists():
    time.sleep(5)
while not load('real_simplex_audit.json').get('complete'):
    time.sleep(5)

v1, v2 = load('full_v1/result.json'), load('full_v2/result.json')
comparison, audit = load('full_comparison.json'), load('real_simplex_audit.json')
native, regression = load('real_native/result.json'), load('regression_comparison.json')
cell = load('streamed_cell60.json')
tests = ET.parse(HERE/'differential_resume.xml').getroot()[0].attrib
accuracy = comparison['accuracy_pass'] and audit['passed'] and native['passed']
status = dict(ACCURACY='PASS' if accuracy else 'FAIL', REGRESSION='FAIL',
    PERFORMANCE='PASS' if v2['performance_360_seconds'] else 'FAIL',
    FULL_MARCENA_TEST='PASS' if accuracy and v2['performance_360_seconds'] else 'FAIL',
    READY_FOR_MANUAL_TESTING='YES' if accuracy else 'NO',
    READY_FOR_PRODUCTION='NO', READY_FOR_SETUP_EXE='NO')
(HERE/'final_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')

existing = ['gui/lidar_classification_tools.py', 'gui/ground_ptd_locator.py']
new = ['gui/ground_ptd_engine_v2.py', 'gui/ground_ptd_engine_v2_LICENSE.txt',
       'tests/test_ground_ptd_engine_v2.py'] + [str(p.relative_to(ROOT)).replace('\\', '/')
       for p in (ROOT/'scripts').glob('*ground*v2*.py')]
manifest, patch = [], []
for relative in existing + sorted(set(new)):
    path = ROOT/relative
    previous = (BACKUP/relative).read_text(encoding='utf-8').splitlines(keepends=True) if relative in existing else []
    current = path.read_text(encoding='utf-8').splitlines(keepends=True)
    patch.append('diff --git a/'+relative+' b/'+relative+'\n')
    if relative not in existing:
        patch.append('new file mode 100644\n')
    patch.extend(difflib.unified_diff(previous, current,
        fromfile='a/'+relative if relative in existing else '/dev/null', tofile='b/'+relative))
    manifest.append(dict(path=relative, type='modified pre-existing source' if relative in existing else 'new file',
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(HERE/'ground_ptd_v2.patch').write_text(''.join(patch), encoding='utf-8', newline='\n')
(HERE/'modified_sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
check = subprocess.run(['git', 'apply', '--reverse', '--check',
    str(HERE/'ground_ptd_v2.patch')], cwd=ROOT, capture_output=True, text=True)
(HERE/'patch_reverse_check.json').write_text(json.dumps(dict(returncode=check.returncode,
    stdout=check.stdout, stderr=check.stderr), indent=2), encoding='utf-8')
baseline_hashes = json.loads((BACKUP/'baseline_hashes.json').read_text(encoding='utf-8-sig'))
changed_existing = []
for item in baseline_hashes:
    p = ROOT/item['path']
    if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest().lower() != item['sha256'].lower():
        changed_existing.append(item['path'])
(HERE/'preservation_check.json').write_text(json.dumps(dict(
    changed_existing_files=changed_existing, expected_only=existing,
    pass_check=set(changed_existing)==set(existing)), indent=2), encoding='utf-8')

peaks = {'0': v1['peak_rss_gb'], '1': v2['peak_rss_gb']}
sample_file = HERE/'full_resource_samples.jsonl'
if sample_file.exists():
    for line in sample_file.read_text(encoding='utf-8').splitlines():
        for process in json.loads(line)['processes']:
            peaks[process['engine']] = max(peaks[process['engine']], process['peak_rss_gb'])

def number(value):
    return f'{value:.3f}'

lines = [
    'NAKSHA GROUND PTD V2 - FINAL VALIDATION REPORT',
    '2026-10-24 (Asia/Calcutta)', '',
    'PRODUCTION DECISION: NOT READY. The 360-second requirement failed and',
    'the broader regression suite has existing failures and a modal timeout.', '',
    *[key+': '+value for key, value in status.items()], '',
    'DATA AND BASELINE',
    f"Source: {v2['source']}", f"Points: {v2['total_points']:,}",
    'Source/default class 0; trusted Ground/destination class 1, as defined',
    'by both project ENEL PTC palettes. No class-2 assumption.',
    'Workers: 2 for each engine. Original streamed cells, overlapping contexts,',
    'fences, seed rules, thresholds, anchor growth and iteration limits retained.',
    'V1: the saved pre-edit classify_ground_ptd function executed in memory.',
    'V2: opt-in engine using the existing streamed planner.',
    f"Original source SHA-256: {v2['source_sha256_before']}",
    'Original LNT was only read. Native writes were confined to disposable copies.', '',
    'FULL OPERATION - MEASURED SECONDS',
    'Stage                                      V1             V2',
]
for label, key in [('LNT reads (cumulative workers)', 'cumulative_read_seconds'),
                   ('TIN builds (cumulative workers)', 'cumulative_tin_seconds'),
                   ('Triangle lookup (cumulative workers)', 'cumulative_lookup_seconds'),
                   ('Planning wall time', 'planning_seconds'),
                   ('Native classification commit', 'native_commit_seconds'),
                   ('Completion class readback', 'readback_seconds'),
                   ('Full cached-LNT operation', 'full_operation_seconds'),
                   ('Including V2 import delay', 'user_visible_seconds_including_v2_import')]:
    lines.append(f'{label:40s} {number(v1[key]):>13s} {number(v2[key]):>14s}')
lines += [f"Recorded peak process RSS (GB)          {peaks['0']:13.3f} {peaks['1']:14.3f}",
    f"Planning CPU (% of 20 logical CPUs)     {v1['cpu_percent_of_machine']:13.3f} {v2['cpu_percent_of_machine']:14.3f}",
    '', 'Cumulative worker stages overlap; they are not summed to obtain wall time.',
    'Timings include LNT opening, bounded reads, planning, native commit, frame',
    'completion and full-class readback. Preparing the disposable copy and',
    'validation-only source snapshots are excluded. This used the cached LNT;',
    'a fresh LAZ-to-LNT conversion was not timed. Filesystem/Numba disk caches',
    'were warm, and no checkpoint results were reused in either complete run.',
    'V2 exceeded 360 seconds before the V1 full validation was launched.',
    'The runs subsequently overlapped; real-data export and simplex auditing',
    'also ran on this workstation. These full times are observed loaded-machine',
    'measurements, not an isolated engine speed ratio. See benchmark_conditions.json.', '',
    'SEPARATE ORIGINAL STREAMED CELL 60 COMPARISON',
    f"V1 {cell['runs'][0]['wall_seconds']:.3f}s; V2 {cell['runs'][1]['wall_seconds']:.3f}s; {cell['speedup']:.3f}x.",
    '1,326,588 context points, 52 iterations, 74,991 exact planned changes.',
    'This paired cell test ran before the full validators. It does not establish',
    'the six-minute full-cloud target. Earlier synthetic speedup was 1.34x;',
    'the six-iteration MARCENA window measured 1.16x. The dense one-iteration',
    'case measured about 2% slower and uses the original SciPy route in V2.', '',
    'EXACT ACCURACY',
    f"Changed points: V1 {comparison['v1_changes']:,}; V2 {comparison['v2_changes']:,}.",
    f"Missing changed IDs: {comparison['missing_changed_point_ids']}; extra IDs: {comparison['extra_changed_point_ids']}.",
    f"Old-class differences: {comparison['old_class_differences']}.",
    f"Destination-class differences: {comparison['destination_class_differences_on_common_ids']}.",
    f"Final per-point class differences: {comparison['final_per_point_class_differences']}.",
    f"Contexts compared: {comparison['contexts_compared']}; context differences: {len(comparison['context_differences'])}.",
    'Every context classification array, exact changed indices and returned',
    'seed/iteration/stop metadata were compared, as well as full planned IDs.',
    'Full anchor sequences were not logged; separate differential tests compare',
    'anchor sequences and iteration traces on representative synthetic terrain.',
    f"Real simplex audit: {audit['audited_queries']:,} complete-array IDs audited, {audit['equivalence_failures']} mismatches.",
    'The audit used the original complete SciPy query, not an uncertain subset.', '',
    'NATIVE REAL-DATA FENCE / HISTORY / SAVE',
    f"Real fenced changes: {native['planned_changes']}; outside-fence changes: {native['outside_fence_changes']}.",
    'Original large-cloud GUI/native dispatch was exercised; no whole-cloud XYZ',
    'snapshot or increased 8M limit was used. Exact commit, Undo and Redo passed.',
    f"LAZ export/reload: {native['exported_points']:,} points, {native['export_class_differences']} class differences.",
    f"Real-data save/export: {native['save_export_seconds']:.3f}s (separate from Ground runtime).", '',
    'AUTOMATED AND BROADER REGRESSIONS',
    f"New differential tests: {tests['tests']} tests; {tests['failures']} failures; {tests['errors']} errors.",
    'They cover adversarial simplex geometries/orders, floating thresholds,',
    'duplicate XY, complete PTD classifications/IDs/anchors, streamed fences,',
    'one/two workers, cancellation, revision rejection and runtime fallback.',
    '52 existing test files passed in the broad run. Thirteen existing failing',
    'files reproduce on the saved baseline with the same failing test IDs.',
    'They include missing plugin imports, outdated mocks/expectations and the',
    'same Windows 0xC0000409 peer-classification test process failure.',
    'The restored Surface-fence picker test enters the same modal warning on',
    'both sources. It timed out; the 38 preceding and three remaining tests passed.',
    'The live cyan-highlight assertion also fails on both sources. It was not',
    'suppressed or repaired by changing unrelated code. Full VTK/Vulkan parity',
    'is therefore NOT VERIFIED. Native Vulkan initialized and exercised the',
    'actual NVIDIA T400 on disposable real/generated datasets.', '',
    'SOURCE PRESERVATION AND DELIVERY',
    'Existing files changed by V2: gui/lidar_classification_tools.py and the',
    'pre-existing untracked gui/ground_ptd_locator.py. Baseline hashes confirm',
    'all other pre-existing source, including rendering, streaming, history,',
    'main.py and settings integrations, is unchanged. Sub_master was not modified.',
    'modified_sources.json lists all engine/test/tool additions and hashes.',
    'ground_ptd_v2.patch contains only V2 changes relative to the saved source.',
    'The new ignored test file is included. Large benchmark copies are excluded.',
    'The original backup, resume snapshot, initial Git status and pre-existing',
    'patch are retained. No commit, push, executable or Setup.exe was made.', '',
    'ENABLE / DISABLE / LIMITATIONS',
    'See TESTING_AND_ROLLBACK.txt for exact launch and reverse-check commands.',
    'NAKSHA_GROUND_ENGINE_V2 defaults to 0. Set it to 0 to use the SciPy baseline.',
    'Keep NAKSHA_GROUND_LOCATOR=0 and NAKSHA_GROUND_GPU=0; the old subset',
    'locator is quarantined. No CUDA or Vulkan numerical acceleration is required.',
    'The compiled locator is version-gated to SciPy 1.15.3 and falls back on',
    'unsupported inputs/runtime failure. NAKSHA_GROUND_V2_AUDIT=1 adds full',
    'runtime query comparison. Immutable caches require bit-identical inputs.',
    'Performance remains dominated by repeated Qhull TIN builds and ordered',
    'triangle searches. Accuracy evidence is specific to tested data/settings;',
    'the six-minute target and full cross-feature GUI/render sign-off remain unmet.',
]
(HERE/'FINAL_VALIDATION_REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(status, indent=2), flush=True)

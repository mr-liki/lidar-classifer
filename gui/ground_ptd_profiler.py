"""Thread-safe, opt-in Ground PTD stage and backend diagnostics.

Distinctly identify CPU versus native C++/Numba and actual Ground CUDA
compute; Vulkan rendering is NOT CUDA Ground classification. This module is
log-only: no input geometry, classes, ordering or Delaunay state are changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock, current_thread

_PRINT_LOCK = Lock()
_TRUE = {'1', 'true', 'yes', 'on'}


def enabled():
    return os.environ.get('NAKSHA_GROUND_PROFILE', '0').strip().lower() in _TRUE


def log(message):
    with _PRINT_LOCK:
        print(message, flush=True)


def report_ptd(data):
    """Write one indivisible summary per PTD call; JSONL is optional."""
    if not enabled():
        return
    payload = dict(data)
    payload['thread'] = current_thread().name
    steps = payload.get('stages_seconds', {})
    backend = payload.get('backend', {})
    inc = backend.get('incremental') or {}
    cpp_queries = backend.get('cpp_queries', 0) + inc.get('queries', 0)
    gpu = payload.get('gpu', {})
    hot = sorted(steps.items(), key=lambda pair: -pair[1])
    hot_text = ', '.join(f'{key}={value:.3f}s' for key, value in hot[:5])
    log('[GROUND PROFILE] points={:,} candidate={:,} iter={} total={:.3f}s | {}'
        .format(payload.get('working_points', 0),
                payload.get('source_candidates', 0),
                payload.get('iterations', 0),
                payload.get('total_seconds', 0.0), hot_text))
    log('[GROUND CONFIG] V2={} C++requested={} StatefulC++={} GPUrequested={} EarlyReject={} '
        'Audit={} | Vulkan renderer is separate from Ground compute'.format(
            'YES' if backend.get('v2_active') else 'NO',
            'YES' if str(backend.get('cpp_requested', '0')).lower() in _TRUE else 'NO',
            'YES' if inc.get('queries', 0) else 'NO',
            'YES' if gpu.get('requested') else 'NO',
            'YES' if payload.get('early_reject') else 'NO',
            'YES' if backend.get('audited_queries') or inc.get('audit_checks') else 'NO'))
    log('[GROUND BACKENDS] CPU=YES C++={} ({} queries) Numba={} ({} queries) '
        'SciPy={} ({} queries) audit={} | GroundGPU={} (calls={}, rejected={}, {:.3f}s) '
        '| VulkanCompute={} ({} calls) CUDA={} ({} calls) CPUshadowAudit={} | VulkanRendering=SEPARATE'.format(
            'YES' if cpp_queries else 'NO',
            cpp_queries,
            'YES' if backend.get('numba_queries', 0) else 'NO',
            backend.get('numba_queries', 0),
            'YES' if backend.get('scipy_queries', 0) else 'NO',
            backend.get('scipy_queries', 0),
            backend.get('audited_queries', 0),
            'YES' if gpu.get('calls', 0) else 'NO',
            gpu.get('calls', 0), gpu.get('rejected', 0), gpu.get('seconds', 0.0),
            'YES' if gpu.get('vulkan_calls', 0) else 'NO', gpu.get('vulkan_calls', 0),
            'YES' if gpu.get('cuda_calls', 0) else 'NO', gpu.get('cuda_calls', 0),
            'YES' if gpu.get('cpu_shadow_verified', False) else 'NO'))
    if backend.get('grid_queries'):
        log('[GROUND GRID] indexed_queries={:,} certified_interior={:,} '
            'ordered_fallback={:,}'.format(
                backend.get('grid_queries', 0), backend.get('grid_fast_hits', 0),
                backend.get('grid_ordered_fallbacks', 0)))
    if backend.get('stage4_queries'):
        log('[GROUND STAGE4] queries={:,} triangle_probes={:,} AABB_rejected={:,} '
            'fine_bin_queries={:,} ambiguous={:,} fine_bins_built={:,} fine_entries={:,} '
            'index_build={:.3f}s native_query={:.3f}s'.format(
                backend.get('stage4_queries', 0),
                backend.get('stage4_probes', 0),
                backend.get('stage4_bbox_pruned', 0),
                backend.get('stage4_refined_queries', 0),
                backend.get('stage4_ambiguous', 0),
                backend.get('stage4_refined_bins', 0),
                backend.get('stage4_refined_entries', 0),
                backend.get('stage4_build_seconds', 0.0),
                backend.get('stage4_lookup_seconds', 0.0)))
    inc = backend.get('incremental')
    if inc:
        log('[GROUND TOPOLOGY] mode={} persistent_updates={} locator={} shortcuts={}'.format(
            inc.get('topology_mode', 'reference_qhull_rebuild'),
            inc.get('persistent_tin_updates', 0),
            inc.get('locator_mode', 'unknown'), inc.get('location_shortcuts', False)))
        log('[GROUND INCREMENTAL] calls={} queries={:,} cached_cert={:,} grid_cert={:,} '
            'ordered_walk={:,} brute={:,} tests_reused={:,} tests_new={:,} '
            'transform={:.3f}s map={:.3f}s locate={:.3f}s tests={:.3f}s select={:.3f}s '
            'audited={}'.format(
                inc.get('calls', 0), inc.get('queries', 0), inc.get('cache_cert', 0),
                inc.get('grid_cert', 0), inc.get('walked', 0), inc.get('brute', 0),
                inc.get('eval_reused', 0), inc.get('eval_new', 0),
                inc.get('transform_s', 0.0), inc.get('map_s', 0.0), inc.get('locate_s', 0.0),
                inc.get('evaluate_s', 0.0), inc.get('select_s', 0.0),
                inc.get('audit_checks', 0)))
    if backend.get('stage4_fallback'):
        log('[GROUND STAGE4] unavailable; Stage 3 fallback={}'.format(
            backend['stage4_fallback']))
    if backend.get('grid_fallback'):
        log('[GROUND GRID] inactive; fallback={}'.format(backend['grid_fallback']))
    if backend.get('cpp_fallback') or backend.get('fallback') or gpu.get('failed_calls'):
        log('[GROUND FALLBACK] cpp={} locator={} gpu_failed_calls={}'.format(
            backend.get('cpp_fallback') or 'none',
            backend.get('fallback') or 'none',
            gpu.get('failed_calls', 0)))
    destination = os.environ.get('NAKSHA_GROUND_PROFILE_FILE', '').strip()
    if destination:
        # The destination is optional; a logging error must never stop or alter
        # a classification run. Avoid creating unrelated directories implicitly.
        try:
            output = Path(destination).expanduser()
            if output.parent.is_dir():
                with _PRINT_LOCK:
                    with output.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(payload, allow_nan=False) + '\n')
        except (OSError, ValueError, TypeError) as exc:
            log('[GROUND PROFILE] JSONL write skipped: {}'.format(type(exc).__name__))

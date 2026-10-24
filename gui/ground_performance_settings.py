"""Persistent Ground runtime options for both Python and packaged Naksha LiDAR.

Only this application's process environment is updated; no system-wide
variables, point-cloud records, or PTD calculation parameters are modified.
QSettings uses the same NakshaAI/LidarApp identity as Global Settings.
"""
from __future__ import annotations

import os
from functools import lru_cache

WORKER_KEY = "performance/ground_workers"
GPU_KEY = "performance/ground_gpu"
DEFAULT_WORKERS = 2
ALLOWED_WORKERS = (1, 2, 4)


def normalize_workers(value):
    """Avoid unbounded concurrent TIN allocations and invalid saved values."""
    try:
        number = int(str(value).strip())
    except (ValueError, TypeError, OverflowError):
        return DEFAULT_WORKERS
    return number if number in ALLOWED_WORKERS else DEFAULT_WORKERS


def normalize_gpu_flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return value == 1


def _settings_or_default(settings):
    if settings is not None:
        return settings
    from PySide6.QtCore import QSettings
    return QSettings("NakshaAI", "LidarApp")


def read_ground_preferences(settings=None):
    settings = _settings_or_default(settings)
    return {
        "workers": normalize_workers(settings.value(WORKER_KEY, DEFAULT_WORKERS)),
        "gpu": normalize_gpu_flag(settings.value(GPU_KEY, False)),
    }


def save_ground_preferences(settings, workers, gpu):
    """Persist preferences. GPU Vulkan/CUDA readiness is checked by runtime."""
    requested = normalize_workers(workers)
    if requested != int(workers):
        raise ValueError("Ground worker count must be 1, 2, or 4")
    settings.setValue(WORKER_KEY, requested)
    settings.setValue(GPU_KEY, bool(gpu))


@lru_cache(maxsize=1)
def cuda_readiness():
    """Report actual CUDA compute readiness, not merely a Vulkan-capable GPU."""
    try:
        import torch
    except (ImportError, OSError) as exc:
        return False, "CUDA unavailable: PyTorch CUDA runtime is not bundled/installed."
    try:
        if not torch.cuda.is_available():
            return False, "CUDA unavailable: no usable CUDA-enabled PyTorch device."
        device = torch.device("cuda:0")
        # A small operation also checks the driver and bundled CUDA libraries.
        tensor = torch.ones((1,), device=device, dtype=torch.float64)
        torch.cuda.synchronize(device)
        del tensor
        return True, "CUDA ready: " + torch.cuda.get_device_name(0)
    except Exception as exc:
        return False, "CUDA unavailable: runtime/device check failed (" + type(exc).__name__ + ")."


def configure_ground_runtime(settings=None, environ=None, *,
                             respect_environment=True, cuda_probe=None):
    """Apply Ground options for future runs; NEVER change global Windows env.

    Explicit process variables are supported for developer launches. Saving
    Global Settings calls with respect_environment=False so the user's new
    selection takes effect in the current application session as well.
    """
    settings = _settings_or_default(settings)
    environ = os.environ if environ is None else environ
    prefs = read_ground_preferences(settings)
    workers = prefs["workers"]
    gpu_requested = prefs["gpu"]
    if respect_environment:
        if "NAKSHA_GROUND_WORKERS" in environ:
            workers = normalize_workers(environ["NAKSHA_GROUND_WORKERS"])
        if "NAKSHA_GROUND_GPU" in environ:
            gpu_requested = normalize_gpu_flag(environ["NAKSHA_GROUND_GPU"])
        if normalize_gpu_flag(environ.get('NAKSHA_GROUND_VULKAN_COMPUTE', '0')):
            # Explicit developer opt-in enables Vulkan even when saved UI
            # preferences were CPU-only. Does not touch system env settings.
            gpu_requested = True

    gpu_ready = False
    backend = 'cpu'
    reason = 'GPU compute disabled; original CPU Ground remains available.'
    vk_explicit = 'NAKSHA_GROUND_VULKAN_COMPUTE' in environ
    vk_preference = environ.get('NAKSHA_GROUND_VULKAN_COMPUTE', '1')
    vk_allowed = str(vk_preference).strip().lower() not in ('0', 'false', 'no', 'off')
    if gpu_requested and vk_allowed and cuda_probe is None:
        try:
            from gui.ground_ptd_vulkan import readiness as probe_vulkan
            gpu_ready, reason = probe_vulkan()
            if gpu_ready:
                backend = 'vulkan'
        except Exception as exc:
            reason = 'Vulkan Ground probe failed (' + type(exc).__name__ + ').'
    if gpu_requested and not gpu_ready:
        probe = cuda_readiness if cuda_probe is None else cuda_probe
        try:
            gpu_ready, cuda_reason = probe()
            reason = cuda_reason if gpu_ready else reason + ' ' + cuda_reason
            if gpu_ready:
                backend = 'cuda'
        except Exception as exc:
            reason = 'GPU check failed (' + type(exc).__name__ + '); original CPU path retained.'
    environ['NAKSHA_GROUND_WORKERS'] = str(workers)
    environ['NAKSHA_GROUND_GPU'] = '1' if gpu_requested and gpu_ready else '0'
    # When explicitly overridden, do not rewrite the developer's selection.
    # Otherwise this flag follows the successful Vulkan readiness check.
    if not vk_explicit:
        environ['NAKSHA_GROUND_VULKAN_COMPUTE'] = '1' if backend == 'vulkan' else '0'
    return {
        'workers': workers,
        'gpu_requested': gpu_requested,
        'gpu_active': bool(gpu_requested and gpu_ready),
        'gpu_backend': backend,
        'gpu_status': reason,
        'cuda_status': reason,  # existing UI consumers expect this legacy key
    }

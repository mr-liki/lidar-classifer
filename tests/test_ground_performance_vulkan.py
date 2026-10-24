from gui import ground_performance_settings as prefs


class Settings:
    def value(self, k, default):
        return {'performance/ground_workers': 2, 'performance/ground_gpu': True}.get(k, default)


def test_vulkan_preferred_over_cuda(monkeypatch):
    from gui import ground_ptd_vulkan as vk
    monkeypatch.setattr(vk, 'readiness', lambda: (True, 'Vulkan GPU dispatch OK'))
    monkeypatch.setattr(prefs, 'cuda_readiness', lambda: (_ for _ in ()).throw(AssertionError('CUDA unnecessary')))
    env = {}
    r = prefs.configure_ground_runtime(settings=Settings(), environ=env)
    assert r['gpu_backend'] == 'vulkan'
    assert r['gpu_active'] is True
    assert env['NAKSHA_GROUND_GPU'] == '1'
    assert env['NAKSHA_GROUND_VULKAN_COMPUTE'] == '1'


def test_vulkan_fallback_to_cuda(monkeypatch):
    from gui import ground_ptd_vulkan as vk
    monkeypatch.setattr(vk, 'readiness', lambda: (False, 'No Vulkan FP64 device'))
    monkeypatch.setattr(prefs, 'cuda_readiness', lambda: (True, 'CUDA ready'))
    env = {}
    r = prefs.configure_ground_runtime(settings=Settings(), environ=env)
    assert r['gpu_backend'] == 'cuda'
    assert env['NAKSHA_GROUND_VULKAN_COMPUTE'] == '0'


def test_both_gpu_backends_absent_cpu_preserved(monkeypatch):
    from gui import ground_ptd_vulkan as vk
    monkeypatch.setattr(vk, 'readiness', lambda: (False, 'No Vulkan FP64 device'))
    monkeypatch.setattr(prefs, 'cuda_readiness', lambda: (False, 'CUDA not available'))
    env = {}
    r = prefs.configure_ground_runtime(settings=Settings(), environ=env)
    assert r['gpu_backend'] == 'cpu'
    assert env['NAKSHA_GROUND_GPU'] == '0'


def test_explicit_vulkan_disable_respected(monkeypatch):
    monkeypatch.setattr(prefs, 'cuda_readiness', lambda: (False, 'CUDA unavailable'))
    env = {'NAKSHA_GROUND_VULKAN_COMPUTE': '0'}
    r = prefs.configure_ground_runtime(settings=Settings(), environ=env)
    assert r['gpu_backend'] == 'cpu'
    assert env['NAKSHA_GROUND_VULKAN_COMPUTE'] == '0'

"""Actual compressed LAZ batch regressions using the native converter."""
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch
import laspy
import numpy as np
import pytest
from gui.lnt_batch import build_batch, BatchCancelled


def write_cloud(path, number, modern=False):
    header = laspy.LasHeader(point_format=8 if modern else 3,
                             version='1.4' if modern else '1.2')
    header.offsets = [500000., 4000000., 0.]
    header.scales = [.001, .002, .001]
    header.add_extra_dim(laspy.ExtraBytesParams(name='echo_rank', type=np.uint16))
    data = laspy.LasData(header)
    n = 1234 + number
    data.x = 500000. + np.arange(n) * .01 + number * 20
    data.y = 4000000. + np.arange(n) * .004
    data.z = np.arange(n) * .001 + 10
    data.classification = np.full(n, number + 2, dtype=np.uint8)
    data.point_source_id = np.full(n, number, dtype=np.uint16)
    data.echo_rank = np.arange(n, dtype=np.uint16)
    data.intensity = np.arange(n, dtype=np.uint16)
    data.red, data.green, data.blue = [np.full(n, c, np.uint16) for c in (10000, 20000, 30000)]
    if modern:
        data.scan_angle = np.full(n, 2000, dtype=np.int16)  # 12 degrees
    else:
        data.scan_angle_rank = np.full(n, 12, dtype=np.int8)
    data.write(path)
    return data


def test_multiple_compressed_laz_and_las_preserve_individual_caches(tmp_path):
    from gui.lnt_collection import read_collection
    from gui.lnt_class_counts import scan_class_histogram
    from gui.lnt_open import build_lnt, lnt_is_current
    paths=[tmp_path/'a.laz',tmp_path/'b.laz',tmp_path/'c.las']
    clouds=[write_cloud(p,i+1,modern=i==1) for i,p in enumerate(paths)]
    before=[p.read_bytes() for p in paths]
    result=build_batch(paths,cache_root=tmp_path/'index',chunk_size=117)
    group=read_collection(result[1])
    assert result[0]==result[1] and result[1].endswith('.lntset')
    assert group['paths']==[str(p.with_suffix('.lnt')) for p in paths]
    assert group['total']==sum(len(c.points) for c in clouds)
    assert Path(result[1]).stat().st_size<10000
    assert not list(tmp_path.rglob('combined.*'))
    assert all(lnt_is_current(str(p)) for p in paths)
    assert before==[p.read_bytes() for p in paths]
    hist=scan_class_histogram(result[1])
    for i,c in enumerate(clouds): assert hist[i+3]==len(c.points)
    with patch('gui.lnt_native.convert_las',side_effect=AssertionError('individual cache not reused')):
        assert build_batch(paths,cache_root=tmp_path/'index')==result
        for p in paths:assert build_lnt(str(p),lambda *a:None,lambda:False)==str(p.with_suffix('.lnt'))
        # A new selection references the same individual caches as well.
        build_batch(paths[1:],cache_root=tmp_path/'index')


def test_cancel_mid_laz_batch_leaves_no_partial_cache(tmp_path):
    paths=[tmp_path/'a.laz',tmp_path/'b.laz']
    for i,p in enumerate(paths):write_cloud(p,i)
    state={'cancel':False}
    def progress(value,text):
        if 'Decompressing' in text:state['cancel']=True
    with pytest.raises(BatchCancelled):
        build_batch(paths,progress,lambda:state['cancel'],cache_root=tmp_path/'index',chunk_size=117)
    assert not list(tmp_path.rglob('*.lntset'))
    assert not list(tmp_path.glob('naksha_lnt_*'))
    assert not paths[0].with_suffix('.lnt').exists()


def test_low_disk_space_fails_beside_source_without_temp_fallback(tmp_path):
    paths=[tmp_path/'a.laz',tmp_path/'b.laz']
    for i,p in enumerate(paths):write_cloud(p,i)
    usage=namedtuple('usage','total used free')(100,99,1)
    with patch('gui.lnt_open.shutil.disk_usage',return_value=usage):
        with pytest.raises((OSError,RuntimeError),match='Not enough free space'):
            build_batch(paths,cache_root=tmp_path/'index')
    assert not list(tmp_path.glob('*.lnt'))
    assert not list(tmp_path.glob('naksha_lnt_*'))


def test_scratch_is_always_in_each_source_folder(tmp_path):
    folders=[tmp_path/'first',tmp_path/'second']
    for p in folders:p.mkdir()
    paths=[p/'tile.laz' for p in folders]
    for i,p in enumerate(paths):write_cloud(p,i)
    import tempfile
    real=tempfile.TemporaryDirectory
    seen=[]
    def scratch(*args,**kwargs):
        assert kwargs.get('dir') is not None,'attempted system TEMP fallback'
        seen.append(Path(kwargs['dir']))
        return real(*args,**kwargs)
    with patch('tempfile.TemporaryDirectory',side_effect=scratch),patch('tempfile.gettempdir',side_effect=AssertionError('system TEMP used')):
        build_batch(paths)
    assert all(p.with_suffix('.lnt').is_file() for p in paths)
    assert all(folder in seen for folder in folders)
    assert all(any(p==f or f in p.parents for f in folders) for p in seen)


def test_unwritable_source_reports_error_without_system_temp(tmp_path):
    paths=[tmp_path/'a.laz',tmp_path/'b.laz']
    for i,p in enumerate(paths):write_cloud(p,i)
    with patch('gui.lnt_open.tempfile.mkstemp',side_effect=PermissionError('source is read-only')):
        with pytest.raises(OSError,match='Choose a writable source folder'):
            build_batch(paths)
    assert not list(tmp_path.rglob('*.lnt'))


def test_repeated_input_is_reported(tmp_path):
    path = tmp_path / 'a.laz'
    write_cloud(path, 1)
    with pytest.raises(ValueError, match='more than once'):
        build_batch([path, path], cache_root=tmp_path / 'cache')


def test_only_changed_source_is_rebuilt_even_after_cache_edit(tmp_path):
    from gui.lnt_open import lnt_is_current
    from gui import lnt_native
    import os
    paths=[tmp_path/'a.laz',tmp_path/'b.laz']
    for i,p in enumerate(paths):write_cloud(p,i)
    build_batch(paths)
    preserved=paths[1].with_suffix('.lnt').read_bytes()
    write_cloud(paths[0],7)
    # Classification changes cache mtime; source fingerprints must still
    # recognize the changed LAZ instead of accepting an obsolete cache.
    os.utime(paths[0].with_suffix('.lnt'),None)
    assert not lnt_is_current(str(paths[0]))
    convert=lnt_native.convert_las
    with patch('gui.lnt_native.convert_las',wraps=convert) as spy:
        _,selection=build_batch(paths)
        assert spy.call_count==1
    assert paths[1].with_suffix('.lnt').read_bytes()==preserved
    from gui.lnt_collection import read_collection
    assert read_collection(selection)['total']==sum(len(laspy.read(p).points) for p in paths)


def test_truncated_individual_cache_is_rebuilt(tmp_path):
    from gui.lnt_open import build_lnt,lnt_is_current
    path=tmp_path/'a.laz';write_cloud(path,1)
    cache=Path(build_lnt(str(path),lambda *a:None,lambda:False))
    cache.write_bytes(cache.read_bytes()[:64])
    assert not lnt_is_current(str(path))
    build_lnt(str(path),lambda *a:None,lambda:False)
    assert lnt_is_current(str(path))


def test_failed_rebuild_preserves_previous_cache_and_cleans_scratch(tmp_path):
    from gui.lnt_open import build_lnt
    path=tmp_path/'a.laz';write_cloud(path,1)
    cache=Path(build_lnt(str(path),lambda *a:None,lambda:False))
    previous=cache.read_bytes()
    write_cloud(path,2)
    with patch('gui.lnt_native.convert_las',side_effect=RuntimeError('converter failed')):
        with pytest.raises(RuntimeError,match='converter failed'):
            build_lnt(str(path),lambda *a:None,lambda:False)
    assert cache.read_bytes()==previous
    assert not list(tmp_path.glob('naksha_lnt_*'))

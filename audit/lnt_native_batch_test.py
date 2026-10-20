"""Native multi-file LAS/LAZ merge: attributes, CRS, cache and cancellation.

Run: venv/Scripts/python.exe audit/lnt_native_batch_test.py
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import laspy
import numpy as np
from pyproj import CRS

from gui.lnt_batch import BatchCancelled, build_batch
from gui.lnt_open import LNT_FORMAT_VERSION, _BuildWorker, lnt_file_version


def write_las(path, count, *, epsg=None, rgb=True, line_id=1, offset=(0., 0.)):
    header = laspy.LasHeader(point_format=3 if rgb else 1, version="1.2")
    header.scales = [.001] * 3
    header.offsets = [500000., 4000000., 0.]
    if epsg is not None:
        header.add_crs(CRS.from_epsg(epsg))
    header.add_extra_dim(laspy.ExtraBytesParams(name="echo_rank", type=np.uint8))
    rng = np.random.default_rng(line_id)
    with laspy.open(path, mode="w", header=header) as writer:
        for start in range(0, count, 1000):
            n = min(1000, count - start)
            record = laspy.ScaleAwarePointRecord.zeros(n, header=header)
            record.x = rng.uniform(0., 50., n) + 500000. + offset[0]
            record.y = rng.uniform(0., 50., n) + 4000000. + offset[1]
            record.z = rng.uniform(10., 20., n)
            record.classification = np.full(n, 2 + line_id, np.uint8)
            record.intensity = rng.integers(1, 60000, n, dtype=np.uint16)
            record.point_source_id = np.full(n, line_id, np.uint16)
            record.echo_rank = np.full(n, line_id, np.uint8)
            if rgb:
                record.red = np.full(n, 30000, np.uint16)
                record.green = np.full(n, 20000, np.uint16)
                record.blue = np.full(n, 10000, np.uint16)
            writer.write_points(record)


from gui.lnt_collection import read_collection
from gui.lnt_class_counts import scan_class_histogram
from gui.lnt_open import build_lnt, lnt_is_current
from gui.lnt_batch import cached_render_cache
from unittest.mock import patch

with tempfile.TemporaryDirectory(prefix="native_batch_") as folder:
    folder=Path(folder)
    first,second=folder/'a.laz',folder/'b.laz'
    write_las(first,3000,epsg=32643,line_id=1)
    write_las(second,2000,epsg=32644,line_id=2)
    target=CRS.from_epsg(32643).to_wkt()
    source,index=build_batch([first,second],target_crs_wkt=target)
    group=read_collection(index)
    assert source==index and index.endswith('.lntset')
    assert group['total']==5000 and len(group['paths'])==2
    assert all(p.with_suffix('.lnt').is_file() for p in (first,second))
    assert group['paths'][0]==str(first.with_suffix('.lnt'))
    assert group['paths'][1]!=str(second.with_suffix('.lnt')) # individual projected variant
    assert Path(group['paths'][1]).parent==second.parent
    assert cached_render_cache(second,target)==group['paths'][1]
    assert not list(folder.rglob('combined.*'))
    np.testing.assert_array_equal(scan_class_histogram(index)[[3,4]],[3000,2000])
    print('PASS mixed CRS: individual reusable LNTs, per-file projected variant, no combined payload',flush=True)
    with patch('gui.lnt_native.convert_las',side_effect=AssertionError('cached source reconverted')):
        assert build_batch([first,second],target_crs_wkt=target)==(source,index)
        for p in (first,second):assert build_lnt(str(p),lambda *a:None,lambda:False)==str(p.with_suffix('.lnt'))
    print('PASS individual files and a cached selection reuse their own LNTs',flush=True)
    worker=_BuildWorker([str(first),str(second)])
    worker.target_crs_wkt=target
    finished=[];worker.finished.connect(lambda p,e:finished.append((p,e)))
    worker.run()
    assert finished and not finished[0][1],finished
    assert worker.render_source.endswith('.lntset')
    print('PASS File > Open worker uses metadata-only individual LNT collection',flush=True)
print('ALL NATIVE INDIVIDUAL CACHE CHECKS PASSED',flush=True)

"""Real raw extra-byte fields must survive LAS/LAZ and native batch loading."""
import hashlib

import laspy
import numpy as np
import pytest

from gui.laspy_compat import install_opaque_extra_bytes_fix
from gui.lnt_open import build_lnt, _laz_to_las, lnt_is_current
from gui.lnt_batch import build_batch
from gui.lnt_collection import read_collection


def write_cloud(path, width):
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.add_extra_dim(laspy.ExtraBytesParams(name='opaque', type=f'{width}u1'))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name='scaled', type='i2', scales=[.25], offsets=[10.]))
    cloud = laspy.LasData(header)
    cloud.x = np.arange(19) + 500000.
    cloud.y = np.arange(19) + 4000000.
    cloud.z = np.arange(19) * .1
    cloud.classification = np.full(19, 2, np.uint8)
    cloud.synthetic = np.ones(19, np.uint8)
    cloud['opaque'] = np.arange(19 * width, dtype=np.uint8).reshape(19, width)
    cloud['scaled'] = np.arange(19) * .25 + 10.
    cloud.write(path)
    return cloud


@pytest.mark.parametrize('width', [5, 9, 17, 25])
@pytest.mark.parametrize('suffix', ['.las', '.laz'])
def test_raw_bytes_preserved_and_native_cache_builds(tmp_path, width, suffix):
    install_opaque_extra_bytes_fix()  # Safe to call more than once.
    source = tmp_path / ('cloud' + suffix)
    expected = write_cloud(source, width)
    before = hashlib.sha256(source.read_bytes()).digest()
    actual = laspy.read(source)
    np.testing.assert_array_equal(actual['opaque'], expected['opaque'])
    np.testing.assert_allclose(actual['scaled'], expected['scaled'])
    opaque = actual.point_format.dimension_by_name('opaque')
    assert opaque.num_elements == width and not opaque.is_scaled
    converted = tmp_path / 'decompressed.las'
    _laz_to_las(str(source), str(converted), lambda *a: None, lambda: False)
    promoted = laspy.read(converted)
    np.testing.assert_array_equal(promoted['opaque'], expected['opaque'])
    np.testing.assert_allclose(promoted['scaled'], expected['scaled'])
    np.testing.assert_array_equal(promoted.classification, np.full(19, 34))
    cache = build_lnt(str(source), lambda *a: None, lambda: False)
    assert lnt_is_current(str(source)) and cache == str(source.with_suffix('.lnt'))
    assert hashlib.sha256(source.read_bytes()).digest() == before


def test_opaque_fields_in_individual_multi_file_caches(tmp_path):
    paths = [tmp_path / 'first.las', tmp_path / 'second.laz']
    for path in paths:
        write_cloud(path, 9)
    source, index = build_batch(paths)
    assert source == index
    group = read_collection(index)
    assert group['paths'] == [str(p.with_suffix('.lnt')) for p in paths]
    assert all(lnt_is_current(str(p)) for p in paths)

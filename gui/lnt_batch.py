"""Build per-source LNT caches and a metadata-only multi-file viewing index."""
from __future__ import annotations
import hashlib
import io
import os
from pathlib import Path
import tempfile
import numpy as np

CACHE_FOLDER_NAME = 'naksha_native_batches'
class BatchCancelled(Exception):
    pass


def cache_candidate_dirs(paths, sizes, folder_name=CACHE_FOLDER_NAME):
    """Source folders only. Never allocate conversion/cache data in system TEMP."""
    totals = {}
    for path, size in zip(paths,sizes):
        parent = Path(path).parent
        totals[parent] = totals.get(parent,0) + int(size)
    return [p/folder_name for p in sorted(totals,key=totals.get,reverse=True)]


def folder_is_writable(folder):
    try:
        Path(folder).mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=folder) as probe:
            probe.write(b'ok');probe.flush()
        return True
    except OSError:
        return False


def cached_render_cache(source, target_crs_wkt=None):
    """The individual cache to reopen, including a reusable project-CRS variant."""
    from gui.lnt_open import lnt_is_current, lnt_path_for, _source_cache_current
    source=os.fspath(source)
    if not lnt_is_current(source): return None
    if not target_crs_wkt or Path(source).suffix.lower()=='.lnt': return lnt_path_for(source)
    import laspy
    from pyproj import CRS
    target=CRS.from_user_input(target_crs_wkt)
    with laspy.open(source) as reader: crs=reader.header.parse_crs()
    if crs is None or crs==target: return lnt_path_for(source)
    tag=hashlib.sha256(target.to_wkt().encode()).hexdigest()[:16]
    variant=str(Path(source).with_name(Path(source).stem+'.'+tag+'.lnt'))
    return variant if _source_cache_current(source,variant) else None


def _projected_cache(source, target, progress, cancelled, chunk_size):
    """Optional per-file project-CRS variant; scratch remains beside this source."""
    import laspy
    from pyproj import CRS, Transformer
    from gui import lnt_native
    from gui.lnt_open import _source_cache_current, _source_signature, _publish_source_cache, _scratch_dir, Cancelled
    with laspy.open(source) as reader:
        header=reader.header.copy()
    crs=header.parse_crs()
    if crs is None or target is None or crs == target:
        if crs is None and target is not None:
            progress(100,f'{Path(source).name} declares no CRS; assuming {target.name}')
        return None
    tag=hashlib.sha256(target.to_wkt().encode()).hexdigest()[:16]
    destination=Path(source).with_name(Path(source).stem+'.'+tag+'.lnt')
    stamp=_source_signature(source)
    if _source_cache_current(source,str(destination)):
        return str(destination)
    transform=Transformer.from_crs(crs,target,always_xy=True)
    lo,hi=np.asarray(header.mins),np.asarray(header.maxs)
    corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    bounds=transform.transform_bounds(lo[0],lo[1],hi[0],hi[1],densify_pts=21,errcheck=True)
    z=np.asarray(transform.transform(*corners.T,errcheck=True)[2])
    low=np.array([bounds[0],bounds[1],z.min()]);high=np.array([bounds[2],bounds[3],z.max()])
    header.offsets=(low+high)*.5
    header.scales=np.maximum(header.scales,np.maximum((high-low)/(2*(2**31-1024)),1e-9))
    header.add_crs(target)
    folder=_scratch_dir(source)
    import shutil
    needed=header.point_count*(header.point_format.size+22)+8*1024**2
    if shutil.disk_usage(folder).free<needed:
        raise OSError(f'Not enough free space beside {source} for its project-CRS cache')
    with tempfile.TemporaryDirectory(prefix='naksha_projected_',dir=folder) as staging:
        las,lnt=Path(staging)/'projected.las',Path(staging)/'cache.lnt'
        with laspy.open(source) as reader,laspy.open(las,mode='w',header=header) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                if cancelled():raise Cancelled()
                output=laspy.ScaleAwarePointRecord.zeros(len(chunk),header=header)
                output.array[:]=chunk.array
                output.x,output.y,output.z=transform.transform(chunk.x,chunk.y,chunk.z,errcheck=True)
                writer.write_points(output)
        if cancelled():raise Cancelled()
        progress(90,f'Building individual project-CRS cache for {Path(source).name}')
        lnt_native.convert_las(str(las),str(lnt))
        if cancelled():raise Cancelled()
        _publish_source_cache(source,lnt,destination,stamp)
    return str(destination)


def build_batch(sources, progress=lambda p,t:None, cancelled=lambda:False,
                target_crs_wkt=None, cache_root=None, chunk_size=500_000):
    """Return (selection index, selection index); no combined LAS/LNT is built.

    Individual caches remain next to every source and are reusable by the
    single-file loader. The small .lntset only routes reads/writes to them.
    """
    import laspy
    from pyproj import CRS
    from gui.lnt_open import build_lnt, Cancelled
    from gui.lnt_collection import write_collection
    paths=[os.path.realpath(os.fspath(p)) for p in sources]
    if len(paths)<2 or any(Path(p).suffix.lower() not in ('.las','.laz','.lnt') for p in paths):
        raise ValueError('Select at least two LAS/LAZ/LNT files')
    if len({os.path.normcase(p) for p in paths})!=len(paths):
        raise ValueError('The same LAS/LAZ file was selected more than once')
    if chunk_size<1:raise ValueError('chunk_size must be positive')
    if len(paths)>4096:raise ValueError('Select at most 4096 individual LNT sources at once')
    destinations=[os.path.normcase(str(Path(p).with_suffix('.lnt'))) for p in paths]
    if len(set(destinations))!=len(destinations):
        raise ValueError('Two selected sources would use the same individual LNT name; rename one source')
    try:
        target=CRS.from_user_input(target_crs_wkt) if target_crs_wkt else None
        if target is None:
            for p in paths:
                if Path(p).suffix.lower()=='.lnt':continue
                with laspy.open(p) as r:target=r.header.parse_crs()
                if target is not None:break
        lnts=[]
        for i,path in enumerate(paths):
            if cancelled():raise Cancelled()
            report=lambda pct,text,i=i:progress((i+pct/100)*90/len(paths),f'[{i+1}/{len(paths)}] {text}')
            lnt=build_lnt(path,report,cancelled)
            if Path(path).suffix.lower()!='.lnt':
                lnt=_projected_cache(path,target,report,cancelled,chunk_size) or lnt
            lnts.append(lnt)
        if cancelled():raise Cancelled()
        key=hashlib.sha256(('\0'.join(paths)+'\0'+(target.to_wkt() if target else '')).encode()).hexdigest()[:24]
        roots=([Path(cache_root)] if cache_root else cache_candidate_dirs(paths,[os.stat(p).st_size for p in paths]))
        encoded=io.BytesIO()
        write_collection(encoded,lnts,paths,target.to_wkt() if target else None)
        payload=encoded.getvalue()
        for root in roots:
            selection=root/key/'selection.lntset'
            try:
                if selection.read_bytes()==payload:
                    progress(100,f'Opening {len(lnts)} cached individual LNTs')
                    return str(selection),str(selection)
            except OSError:
                pass
        folder=next((r/key for r in roots if folder_is_writable(r/key)),None)
        if folder is None:raise OSError('Cannot write the selection index in a source folder; system TEMP is not used')
        selection=folder/'selection.lntset'
        with tempfile.TemporaryDirectory(prefix='index_',dir=folder) as staging:
            staged=Path(staging)/'selection.lntset'
            staged.write_bytes(payload)
            if cancelled():raise Cancelled()
            os.replace(staged,selection)
        progress(100,f'Opening {len(lnts)} individual LNT caches (no combined point file)')
        return str(selection),str(selection)
    except Cancelled:
        raise BatchCancelled() from None

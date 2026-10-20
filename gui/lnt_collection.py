"""A metadata-only view of individual LNTs; point payloads stay in their files."""
from __future__ import annotations
import json
from contextlib import nullcontext
import os
from pathlib import Path
import struct
import numpy as np

MAGIC = b'LNTSET1\0'
HEADER = struct.Struct('<4sHHQ6d')
DIRECTORY = struct.Struct('<QQ')
ENTRY = np.dtype([('node_id','<u8'),('byte_offset','<u8'),('byte_length','<u4'),
    ('point_count','<u4'),('center','<f4',3),('half_size','<f4'),
    ('children_mask','u1'),('tree_level','u1'),('reserved','u1',10)])
ROUTE = np.dtype([('file','<u4'),('offset','<u8'),('root','u1')])


def is_collection(path):
    return os.fspath(path).lower().endswith('.lntset')


def read_plain_directory(path):
    with open(path,'rb') as f:
        raw = f.read(HEADER.size)
        header = HEADER.unpack(raw)
        if header[0] != b'LNT\0' or header[1] != 2:
            raise ValueError(f'{path}: unsupported LNT header')
        f.seek(header[3])
        total, count = DIRECTORY.unpack(f.read(DIRECTORY.size))
        entries = np.frombuffer(f.read(count*ENTRY.itemsize),dtype=ENTRY).copy()
        if len(entries) != count:
            raise ValueError(f'{path}: truncated node directory')
    return header, int(total), entries


def write_collection(path, lnts, sources=None, target_crs=None):
    """Write only paths, directory entries and routing metadata, never point data."""
    parts = [read_plain_directory(p) for p in lnts]
    low = np.min([p[0][4:7] for p in parts],axis=0)
    high = np.max([p[0][7:10] for p in parts],axis=0)
    high = low + max(float(np.max(high-low)),1e-9)
    prefix_depth = 1
    while 8**prefix_depth < len(parts):
        prefix_depth += 1
    blocks, routes = [], []
    offset, total = 64, 0
    for index, (header, points, original) in enumerate(parts):
        entries = original.copy()
        levels = original['tree_level'].astype(np.uint64)
        if len(levels) and int(levels.max()) + prefix_depth > 21:
            raise ValueError('Too many files/tree levels for a 64-bit LNT forest')
        shifts = 3*levels
        prefix = np.uint64(8**prefix_depth + index)
        entries['node_id'] = (prefix << shifts) | (original['node_id'] & ((np.uint64(1)<<shifts)-np.uint64(1)))
        entries['tree_level'] = levels + prefix_depth
        entries['center'] = original['center'].astype(np.float64) + np.asarray(header[4:7])-low
        entries['byte_offset'] = offset + np.r_[np.uint64(0),np.cumsum(entries['byte_length'].astype(np.uint64))[:-1]]
        offset += int(entries['byte_length'].astype(np.uint64).sum())
        route = np.zeros(len(entries),dtype=ROUTE)
        route['file'],route['offset'],route['root'] = index,original['byte_offset'],original['tree_level']==0
        blocks.append(entries); routes.append(route); total += points
    entries = np.concatenate(blocks); route = np.concatenate(routes)
    header = HEADER.pack(b'LNT\0',2,0,offset,*low,*high)
    metadata = json.dumps(dict(sources=list(map(os.fspath,sources or lnts)),target_crs=target_crs)).encode('utf-8')
    with (nullcontext(path) if hasattr(path,'write') else open(path,'wb')) as f:
        f.write(MAGIC);f.write(struct.pack('<I',len(lnts)))
        for name in lnts:
            encoded = os.path.realpath(name).encode('utf-8')
            f.write(struct.pack('<I',len(encoded)));f.write(encoded)
        f.write(header);f.write(DIRECTORY.pack(total,len(entries)))
        f.write(entries.tobytes());f.write(route.tobytes())
        f.write(struct.pack('<I',len(metadata)));f.write(metadata)


def read_collection(path):
    with open(path,'rb') as f:
        if f.read(8) != MAGIC:
            raise ValueError(f'{path}: invalid LNT collection')
        count, = struct.unpack('<I',f.read(4))
        if not 1 <= count <= 4096:
            raise ValueError('Invalid LNT collection file count')
        paths=[]
        for _ in range(count):
            size,=struct.unpack('<I',f.read(4))
            if not 0 < size <= 32768:
                raise ValueError('Invalid LNT collection path')
            paths.append(f.read(size).decode('utf-8'))
        header=HEADER.unpack(f.read(64))
        total,nodes=DIRECTORY.unpack(f.read(16))
        if not 0 < nodes <= (os.fstat(f.fileno()).st_size-f.tell())//(ENTRY.itemsize+ROUTE.itemsize):
            raise ValueError('Invalid LNT collection directory size')
        entries=np.frombuffer(f.read(nodes*ENTRY.itemsize),dtype=ENTRY).copy()
        routes=np.frombuffer(f.read(nodes*ROUTE.itemsize),dtype=ROUTE).copy()
        if len(entries)!=nodes or len(routes)!=nodes:
            raise ValueError('Truncated LNT collection')
        size,=struct.unpack('<I',f.read(4))
        meta=json.loads(f.read(size))
    return dict(paths=paths,header=header,total=total,entries=entries,routes=routes,**meta)


def payload_bytes(path, offsets, collection=None):
    """Read arbitrary virtual byte addresses (used for exact class-count deltas)."""
    offsets=np.asarray(offsets,dtype=np.int64)
    if not is_collection(path):
        mm=np.memmap(path,dtype=np.uint8,mode='r')
        try:return np.array(mm[offsets])
        finally:del mm
    group=collection if collection is not None else read_collection(path)
    starts=group['entries']['byte_offset'].astype(np.int64)
    flat=offsets.ravel()
    slots=np.searchsorted(starts,flat,side='right')-1
    if np.any(slots<0):raise ValueError('Address outside collection payload')
    rows=group['entries'][slots]; routes=group['routes'][slots]
    delta=flat-starts[slots]
    if np.any(delta>=rows['byte_length']):raise ValueError('Address outside collection payload')
    out=np.empty(len(flat),np.uint8)
    for i,p in enumerate(group['paths']):
        hit=routes['file']==i
        if not hit.any():continue
        mm=np.memmap(p,dtype=np.uint8,mode='r')
        try:out[hit]=mm[routes['offset'][hit].astype(np.int64)+delta[hit]]
        finally:del mm
    return out.reshape(offsets.shape)

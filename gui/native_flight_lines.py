"""Persistent flight-line IDs for individual native caches and collections."""
from pathlib import Path
import json
import os
import struct
import tempfile
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QMessageBox
from gui.lnt_collection import is_collection, read_collection, read_plain_directory


class FlightLines:
    def __init__(self, path):
        self.group = read_collection(path) if is_collection(path) else None
        self.paths = self.group['paths'] if self.group else [str(path)]
        self.maps = []
        self.sources = []
        for item in self.paths:
            try:
                with open(str(item)+'.source.json',encoding='utf-8') as f: original=json.load(f)['source']
            except (OSError,ValueError,KeyError): original=str(item)
            self.sources.append(original)
            header, total, entries = read_plain_directory(item)
            count = (header[3]-64)//11
            side = str(item)+'.lines'
            with open(side,'rb') as f:
                magic, stored = struct.unpack('<8sQ',f.read(16))
            if magic != b'LNTLINE1' or stored != count or os.path.getsize(side) != 16+count*2:
                raise ValueError('Invalid flight-line sidecar: '+side)
            self.maps.append(np.memmap(side, dtype='<u2', mode='r', offset=16, shape=(count,)))

    def ids(self, nodes, local, entries):
        offsets = entries['byte_offset'][nodes].astype(np.int64)
        files = np.zeros(len(nodes),dtype=np.int64)
        if self.group:
            route = self.group['routes'][nodes]
            files = route['file']
            offsets = route['offset'].astype(np.int64)
        out = np.empty(len(nodes),np.uint16)
        for i, mapping in enumerate(self.maps):
            hit = files == i
            out[hit] = mapping[(offsets[hit]-64)//11+local[hit]]
        return out

    def unique(self):
        return np.unique(np.concatenate([np.unique(m) for m in self.maps]))

    def close(self):
        for mapping in self.maps: mapping._mmap.close()
        self.maps=[]

    def source_files(self, nodes):
        paths = self.group.get('sources',self.sources) if self.group else self.sources
        files = self.group['routes']['file'][nodes] if self.group else np.zeros(len(nodes),int)
        names = np.array([str(p) for p in paths],dtype=object)
        return names[files]


def ensure_sidecars(view, cancelled=lambda:False):
    """Backfill older caches without rebuilding or changing classified points."""
    try:
        return FlightLines(view.lnt_path)
    except (OSError,ValueError,struct.error):
        pass
    group = read_collection(view.lnt_path) if is_collection(view.lnt_path) else None
    paths = group['paths'] if group else [view.lnt_path]
    sources = group['sources'] if group else [getattr(view.app,'_lnt_source_path','')]
    for path, source in zip(paths,sources):
        if cancelled(): raise RuntimeError('Flight-line preparation cancelled')
        try:
            FlightLines(path)
            continue
        except (OSError,ValueError,struct.error):
            pass
        try:
            with open(path+'.source.json',encoding='utf-8') as f: source=json.load(f)['source']
        except (OSError,ValueError,KeyError): pass
        if not str(source).lower().endswith(('.las','.laz')):
            candidates=[Path(path).with_suffix(ext) for ext in ('.laz','.las')]
            source=next((str(p) for p in candidates if p.is_file()),'')
        if not source or not os.path.isfile(source):
            raise ValueError('The original LAS/LAZ is needed to recover flight-line IDs for '+str(path))
        header, total, entries = read_plain_directory(path)
        count=(header[3]-64)//11
        origin=np.array(header[4:7])
        from gui.native_fields import POINT
        from gui.native_section_display import source_line_ids
        with tempfile.TemporaryDirectory(prefix='naksha_lines_',dir=Path(path).parent) as work:
            # Replaying the deterministic converter recovers exact IDs even for
            # coincident returns. Verify every geometry/attribute byte before
            # using its sidecar; the user's classified cache is never replaced.
            from gui import lnt_native
            from gui.lnt_open import _laz_to_las
            staged_las=source
            if str(source).lower().endswith('.laz'):
                staged_las=str(Path(work)/'source.las')
                _laz_to_las(source,staged_las,lambda *args:None,cancelled)
            replay=str(Path(work)/'replay.lnt')
            lnt_native.convert_las(str(staged_las),replay)
            if cancelled(): raise RuntimeError('Flight-line preparation cancelled')
            replay_header, replay_total, replay_entries=read_plain_directory(replay)
            matches=header==replay_header and np.array_equal(entries,replay_entries)
            if matches:
                old=np.memmap(path,dtype=np.uint8,mode='r',offset=64,shape=(count,11))
                new=np.memmap(replay,dtype=np.uint8,mode='r',offset=64,shape=(count,11))
                try:
                    for start in range(0,count,500000):
                        if not np.array_equal(old[start:start+500000,:10],new[start:start+500000,:10]):
                            matches=False; break
                finally:
                    old._mmap.close(); new._mmap.close()
            if matches:
                os.replace(replay+'.lines',str(path)+'.lines')
                continue
            xyz=np.memmap(Path(work)/'xyz',mode='w+',dtype='<f8',shape=(count,3))
            data=np.memmap(path,mode='r',dtype=np.uint8)
            tolerance=0.
            for node in entries:
                start=int(node['byte_offset']); n=int(node['point_count'])
                raw=data[start:start+n*11].view(POINT)
                at=(start-64)//11; half=float(node['half_size'])
                xyz[at:at+n]=origin+node['center']-half+raw['rel']*(2*half/65535.)
                tolerance=max(tolerance,2*half/65535.*np.sqrt(3)*.501)
            del data
            ids=source_line_ids(source,xyz,tolerance,getattr(view.app,'_lnt_render_crs_wkt',None))
            del xyz
            staged=Path(work)/'lines'
            with open(staged,'wb') as f:
                f.write(struct.pack('<8sQ',b'LNTLINE1',count)); f.write(ids.astype('<u2').tobytes())
            os.replace(staged,str(path)+'.lines')
    return FlightLines(view.lnt_path)


class _Worker(QThread):
    ready=Signal(object,str)
    def __init__(self,view):
        super().__init__(view.app); self.view=view; self.callbacks=[]
    def run(self):
        try: self.ready.emit(ensure_sidecars(self.view,self.isInterruptionRequested),'')
        except Exception as exc: self.ready.emit(None,str(exc))


def prepare(view, callback, required=True):
    if getattr(view,'flight_lines',None) is not None:
        callback(); return
    worker=getattr(view,'_flight_worker',None)
    if worker is not None:
        worker.callbacks.append(callback); return
    worker=_Worker(view); worker.callbacks.append(callback); view._flight_worker=worker
    def done(result,error):
        view._flight_worker=None
        if getattr(view.app,'_lnt_main_view',None) is not view or getattr(view,'_flight_closing',False):
            if result is not None: result.close()
            return
        if error:
            view._flight_unavailable=True
            view.app.statusBar().showMessage(error,15000)
            if required: QMessageBox.warning(view.app,'Flight lines',error)
            else:
                for fn in worker.callbacks: fn()
            return
        view.flight_lines=result
        for fn in worker.callbacks: fn()
    worker.ready.connect(done)
    worker.finished.connect(worker.deleteLater)
    view.app.statusBar().showMessage('Preparing flight-line information...',5000)
    worker.start()

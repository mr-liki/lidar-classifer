"""Compare native boundary flags with the shipped VTK numpy implementation.
Build structural_boundary_probe.cpp as a DLL with the native include directory,
then run: python audit/check_structural_boundary.py path/to/probe.dll
"""
import ast
import contextlib
import ctypes
import io
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT/'gui/unified_actor_manager.py').read_text(encoding='utf-8'))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_boundary_flags')
scope = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-vtk-boundary>', 'exec'), scope)
reference = scope['_compute_boundary_flags']
dll = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
fn = dll.boundary_probe
fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
fn.restype = ctypes.c_int
rng = np.random.default_rng(42)
x,y = np.meshgrid(np.linspace(0,40,301), np.linspace(0,30,201))
base = np.column_stack([x.ravel(), y.ravel(), rng.normal(0,.04,x.size)]).astype(np.float32)
classes = np.where(base[:,0]<20, 2, 5).astype(np.uint8)
roof = (base[:,0]>6)&(base[:,0]<15)&(base[:,1]>8)&(base[:,1]<23)
stepped = base.copy(); stepped[roof,2] += 6
fixtures = [('plane',base,None), ('classes',base,classes), ('roof',stepped,classes),
            ('gaps',stepped[::3],classes[::3]),
            ('random',rng.uniform([0,0,-3],[50,30,10],(100000,3)).astype(np.float32),rng.integers(0,10,100000,dtype=np.uint8)),
            ('empty',np.empty((0,3),np.float32),None)]
for name,xyz,cls in fixtures:
    xyz = np.ascontiguousarray(xyz)
    cls = None if cls is None else np.ascontiguousarray(cls)
    for mode in range(6):
        actual = np.zeros(len(xyz),np.uint8)
        assert fn(xyz.ctypes.data, None if cls is None else cls.ctypes.data, len(xyz), actual.ctypes.data, mode)
        pts=xyz.copy(); cc=None if cls is None else cls.copy(); n=len(pts)
        if mode==2 and n>30 and cc is not None: cc[n//4:n//4+n//2]=9
        # The class-patch probe deliberately adds a class channel when absent.
        if mode==2 and n>30 and cc is None: continue
        if mode==3 and n>30: pts[n//4:n//4+n//2,2]+=4
        with contextlib.redirect_stdout(io.StringIO()):
            if mode in (4,5):
                first=0 if mode==4 else n//4
                expected=np.zeros(n,bool)
                expected[first:first+n//2]=reference(pts[first:first+n//2].astype(np.float64),None if cc is None else cc[first:first+n//2])>0
            else:
                expected=reference(pts.astype(np.float64),cc)>0
        assert np.array_equal(actual>0,expected), (name,mode,int(np.count_nonzero((actual>0)!=expected)))
        print('PASS',name,'mode',mode, 'edges',int(expected.sum()))
print('Native flags match VTK: full clouds, tile seams, class edits, slot reuse, scan gaps and empty data.')


# Uneven LOD density must not turn isolated samples on a flat roof into rings.
x,y=np.meshgrid(np.linspace(0,10,151),np.linspace(0,10,151))
dense=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,5))).astype(np.float32)
x,y=np.meshgrid(np.arange(20,40,.8),np.arange(0,20,.8))
sparse=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,5))).astype(np.float32)
pts=np.ascontiguousarray(np.concatenate((dense,sparse)));cc=np.full(len(pts),6,np.uint8)
legacy=np.zeros(len(pts),np.uint8);lod=legacy.copy()
assert fn(pts.ctypes.data,cc.ctypes.data,len(pts),legacy.ctypes.data,0)
assert fn(pts.ctypes.data,cc.ctypes.data,len(pts),lod.ctypes.data,6)
interior=(pts[:,0]>21)&(pts[:,0]<38)&(pts[:,1]>1)&(pts[:,1]<18)
assert np.count_nonzero(lod[interior])==0
assert np.count_nonzero(legacy[interior])>100
print('PASS sparse flat-roof LOD samples do not become interior scan edges')
# A same-class low roof step remains a boundary after local-plane correction.
pts=stepped.copy();pts[roof,2]=6;pts[roof&(pts[:,0]>10),2]+=.7
lod=np.zeros(len(pts),np.uint8);cc=np.full(len(pts),6,np.uint8)
assert fn(pts.ctypes.data,cc.ctypes.data,len(pts),lod.ctypes.data,6)
near_step=roof&(np.abs(pts[:,0]-10)<.4)&(pts[:,1]>10)&(pts[:,1]<20)
assert np.count_nonzero(lod[near_step])>20
print('PASS 70-centimeter roof step remains visible in streamed boundary flags')

# A lower ground return in a missing roof-sample cell must not outline the
# surrounding upper roof interior. The lower return itself may remain flagged.
x,y=np.meshgrid(np.linspace(0,6,61),np.linspace(0,6,61))
pts=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,6))).astype(np.float32)
res=max(.05,np.sqrt(36/len(pts))*1.5)
grid=np.floor(pts[:,:2].astype(np.float64)/res).astype(int)
hole=(grid[:,0]==15)&(grid[:,1]==15)
cc=np.full(len(pts),6,np.uint8);cc[hole]=2;pts[hole,2]=0
lod=np.zeros(len(pts),np.uint8)
assert fn(pts.ctypes.data,cc.ctypes.data,len(pts),lod.ctypes.data,6)
near=(np.abs(grid[:,0]-15)<=1)&(np.abs(grid[:,1]-15)<=1)&~hole
assert not np.any(lod[near])
print('PASS lower-class LOD sampling holes do not outline upper roof interiors')

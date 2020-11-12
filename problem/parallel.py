# -*- coding: utf-8 -*-
from __future__ import division, print_function
import multiprocessing as mp
import time

from dolfin import Mesh, UnitSquareMesh, File
import tempfile # fenics cannot create or write meshes from or to other things than files...
import types
try:
    import copy_reg as copyreg  # python2
except ImportError:
    import copyreg              # python3

# def _pickle_method(method):
#     func_name = method.im_func.__name__
#     obj = method.im_self
#     cls = method.im_class
#     return _unpickle_method, (func_name, obj, cls)
# def _unpickle_method(func_name, obj, cls):
#     for cls in cls.mro():
#         try: func = cls.__dict__[func_name]
#         except KeyError: pass
#         else: break
#     return func.__get__(obj, cls)

def _pickle_method(method):
    obj = method.im_self
    name = method.im_func.__name__
    return _unpickle_method, (obj, name)

def _unpickle_method(obj, name):
    return getattr(obj, name)

def _pickle_mesh(mesh):
    with tempfile.NamedTemporaryFile(prefix='mesh_', suffix='.xml') as f:
        File(f.name) << mesh
        xml = f.read()
    return _unpickle_mesh, (xml,)

def _unpickle_mesh(xml):
    with tempfile.NamedTemporaryFile(prefix='mesh_', suffix='.xml') as f:
        f.write(xml)
        f.flush()
        mesh = Mesh(f.name)
    return mesh

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
copyreg.pickle(Mesh, _pickle_mesh, _unpickle_mesh)

ts = lambda: time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
_print = lambda *args, **kwargs: print(ts(), *args, **kwargs)


class ParallelizableProblem(object):
    def __getstate__(self):
        return self.info

    def __setstate__(self, state):
        assert isinstance(state, dict)
        self.__dict__.update(self.__class__(state).__dict__)

    @property
    def mesh(self):
        mesh = self.info['fe']['mesh']
        if isinstance(mesh, int):
            mesh = Mesh(UnitSquareMesh(mesh, mesh))
            self.info['fe']['mesh'] = mesh
        elif not isinstance(mesh, Mesh):
            mesh = Mesh(mesh)
            self.info['fe']['mesh'] = mesh
        return mesh

    @property
    def name(self):
        return self.info['problem']['name']

    @property
    def degree(self):
        return self.info['fe']['degree']

    def dofs(self): return self.space.dim()  # len(Function(self.space).vector())


class Parallel(object):
    def __init__(self, function, cpucount=None, maxchunksize=float('inf')):
        self.function = function
        self.cpucount = cpucount or mp.cpu_count()
        self.pool = mp.Pool(self.cpucount)
        self.batch_num = 0
        self.maxchunksize = maxchunksize

    def __call__(self, iterable):
        chunksize = min(max(len(iterable) // self.cpucount, 1), self.maxchunksize)
        self.batch_num += 1
        _print("Computing batch {} of '{}' (batchsize: {} | chunksize: {})".format(self.batch_num, self.function.__name__, len(iterable), chunksize))
        result = self.pool.map_async(self.function, iterable, chunksize)
        return result.get()

    def __del__(self):
        self.pool.terminate()


class Sequential(object):
    def __init__(self, function, cpucount=None):
        self.function = function
        self.batch_num = 0

    def __call__(self, iterable):
        chunksize = len(iterable)
        self.batch_num += 1
        _print("Computing batch {} of '{}' (batchsize: {} | chunksize: {})".format(self.batch_num, self.function.__name__, len(iterable), chunksize))
        result = map(self.function, iterable)
        return list(result)

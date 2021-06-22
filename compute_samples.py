# coding: utf-8
import argparse, os, json, time
# import multiprocessing as mp

import numpy as np
import sobol
from joblib import Parallel, delayed


N_JOBS = -1


def log(*args, **kwargs):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()), *args, **kwargs)

class Sampler(object):
    def __init__(self, _info):
        dimension    = _info['expansion']['size']
        distribution = _info['sampling']['distribution']
        strategy     = _info['sampling']['strategy']

        if distribution not in ['normal', 'uniform']:
            raise ValueError("distribution must be 'uniform' or 'normal'")
        if strategy not in ['random', 'sobol']:
            raise ValueError("strategy must be 'random' or 'sobol'")

        if distribution == 'normal' and strategy == 'random':
            def generator(_numSamples):
                return np.random.randn(_numSamples, dimension)
        elif distribution == 'normal' and strategy == 'sobol':
            self.offset = 1
            def generator(_numSamples):
                samples = sobol.i4_sobol_generate_std_normal(dim_num=dimension, n=_numSamples, skip=self.offset)
                self.offset += _numSamples
                return samples
        elif distribution == 'uniform' and strategy == 'random':
            def generator(_numSamples):
                return 2*np.random.rand(_numSamples, dimension)-1
        else:
            self.offset = 1
            def generator(_numSamples):
                samples = 2*sobol.i4_sobol_generate(dim_num=dimension, n=_numSamples, skip=self.offset)-1
                self.offset += _numSamples
                return samples
        self.generator = generator

    def __call__(self, _numSamples):
        return self.generator(_numSamples)

def load_problem_and_sampler(info):
    problemName = info['problem']['name']
    log(f"Loading problem: {problemName}")
    Problem = __import__(f"problem.{problemName}", fromlist=["Problem"]).Problem
    problem = Problem(info)

    log(f"Loading sampler: {info['sampling']['strategy']}-{info['sampling']['distribution']} (dimension: {info['expansion']['size']})")
    sampler = Sampler(info)

    return problem, sampler

class NPZStorageDirectory(object):
    def __init__(self, _directory):
        self.filePath = os.path.join(_directory, '{fileCount}.npz')
        self.fileCount = 0

    def __lshift__(self, _keysAndValues):
        fileName = self.filePath.format(fileCount=self.fileCount)
        log(f"Saving samples: '{fileName}'")
        np.savez_compressed(fileName, **_keysAndValues)
        self.fileCount += 1

def compute_batch(_batchSize, _storage):
    ys = sampler(_batchSize)
    if not np.all(np.isfinite(ys)):
        raise RuntimeError(f"Invalid value encountered in output of sampler: {np.count_nonzero(~np.all(np.isfinite(ys), axis=1))} samples are not finite.")


    ks = Parallel(n_jobs=N_JOBS)(map(delayed(problem.coefficient_vector), ys))
    us = Parallel(n_jobs=N_JOBS)(map(delayed(problem.solution), ys))

    # pool = mp.Pool()
    # ks = pool.map_async(problem.coefficient_vector, ys)
    # us = pool.map_async(problem.solution, ys)
    # ks = ks.get()
    # us = us.get()
    # pool.close()
    # pool.join()

    ks = np.array(ks)
    if not np.all(np.isfinite(ks)):
        raise RuntimeError(f"Invalid value encountered in output of problem.solution: {np.count_nonzero(~np.all(np.isfinite(ks), axis=1))} samples are not finite.")
    if len(ys) != len(ks):
        raise RuntimeError(f"Number of Samples and number of Solutions differ: {len(ys)} != {len(ks)}")

    us = np.array(us)
    if not np.all(np.isfinite(us)):
        raise RuntimeError(f"Invalid value encountered in output of problem.solution: {np.count_nonzero(~np.all(np.isfinite(us), axis=1))} samples are not finite.")
    if len(ys) != len(us):
        raise RuntimeError(f"Number of Samples and number of Solutions differ: {len(ys)} != {len(us)}")

    _storage << dict(ys=ys, ks=ks, us=us)

def isInt(s):
    try:
        int(s)
        return True
    except:
        return False

def load_parameters(problemDir):
    problemFile = f"{problemDir}/parameters.json"
    try:
        with open(problemFile, 'r') as f:
            problemInfo = json.load(f)
    except FileNotFoundError:
        raise IOError(f"Can not read file '{problemFile}'")
    except json.JSONDecodeError:
        raise IOError(f"'{problemFile}' is not a valid JSON file")
    return problemInfo


if __name__=='__main__':
    descr = """Sample solutions for the given problem."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('PROBLEM', help='path to the directory where the samples will be stored. The problem specification is assumed to lie in (PROBLEM/parameters.json)')
    parser.add_argument('SAMPLES', type=int, help='the number of samples to compute')
    parser.add_argument('-b', '--batch-size', dest='BATCH_SIZE', type=int, default=100, help='the size of each batch (default: 100)')
    args = parser.parse_args()

    problemDir = args.PROBLEM
    if not os.path.isdir(problemDir):
        raise IOError(f"PROBLEM '{problemDir}' is not a directory")

    problemDirContents = os.listdir(problemDir)
    if not 'parameters.json' in problemDirContents:
        raise IOError(f"'{problemDir}' does not contain a 'parameters.json'")
    for fileName in problemDirContents:
        if fileName.endswith('.npz') and isInt(fileName[:-4]):
            raise IOError(f"'{problemDir}' already contains other data ('{problemDir}/{fileName}')")

    problemInfo = load_parameters(problemDir)
    problem, sampler = load_problem_and_sampler(problemInfo)
    storage = NPZStorageDirectory(problemDir)

    number = args.SAMPLES
    batchSize = args.BATCH_SIZE
    batchNumber = 0
    while number > 0:
        log(f"Computing batch: {batchNumber}")
        compute_batch(min(batchSize, number), storage)
        number -= batchSize
        batchNumber += 1

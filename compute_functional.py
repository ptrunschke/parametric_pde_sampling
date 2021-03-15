# coding: utf-8
import argparse, os, json, time

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.hermite_e import hermeval
from scipy.special import factorial

def log(*args, **kwargs):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()), *args, **kwargs)

def isInt(s):
    try:
        int(s)
        return True
    except:
        return False

def isFunctional(s):
    if s not in ["mean"]:#, "max"]:
        raise ValueError(f"Unknown functional '{s}'")
    return s

if __name__=='__main__':
    descr = """Compute linear functional for the samples of the given problem."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('PROBLEM', help='Path to the directory where the results will be stored. The problem specification is assumed to lie in (PROBLEM/parameters.json)')
    parser.add_argument('-f', '--functional', dest='FUNCTIONAL', type=isFunctional, default='mean', choices=['mean'], help='the functional to apply')
    args = parser.parse_args()

    problemDir = args.PROBLEM
    if not os.path.isdir(problemDir):
        raise IOError(f"PROBLEM '{problemDir}' is not a directory")

    problemDirContents = os.listdir(problemDir)
    if not 'parameters.json' in problemDirContents:
        raise IOError(f"'{problemDir}' does not contain a 'parameters.json'")
    functionalFile = f"functional_{args.FUNCTIONAL}.npz"
    if functionalFile in problemDirContents:
        raise IOError(f"'{problemDir}' already contains other data ('{problemDir}/{functionalFile}')")

    if args.FUNCTIONAL == "mean":
        from compute_orthogonalization import load_problem, get_mass_matrix
        problemFile = f"{problemDir}/parameters.json"
        try:
            with open(problemFile, 'r') as f:
                problemInfo = json.load(f)
        except FileNotFoundError:
            raise IOError(f"Can not read file '{problemFile}'")
        except json.JSONDecodeError:
            raise IOError(f"'{problemFile}' is not a valid JSON file")
        problem = load_problem(problemInfo)
        V = problem.space
        M = get_mass_matrix(V)
        oM = np.ones(M.shape[0]) @ M
        def functional(us):
            return us @ oM

    log("Loading precomputed samples")
    samples = []
    values  = []
    problemDirContents = os.listdir(problemDir)
    for fileName in problemDirContents:
        if not (fileName.endswith('.npz') and isInt(fileName[:-4])):
            continue
        z = np.load(os.path.join(problemDir, fileName))
        samples.append(z['ys'])
        values.append(functional(z['us']))
        if len(samples[-1]) != len(values[-1]):
            raise RuntimeError(f"Number of Samples and number of Solutions differ in file '{problemDir}/{fileName}' ({len(samples[-1])} != {len(values[-1])})")

    samples = np.concatenate(samples, axis=0)
    values = np.concatenate(values, axis=0)
    assert len(samples) == len(values), "IE"
    if len(samples) == 0:
        raise IOError(f"'{problemDir}' does not contain any samples")

    np.savez_compressed(f"{problemDir}/{functionalFile}", samples=samples, values=values)

# coding: utf-8
import argparse, os, json, time

import numpy as np
import scipy.sparse as sps
import sksparse.cholmod as cholmod

from dolfin import *
set_log_level(LogLevel.WARNING)

def log(*args, **kwargs):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()), *args, **kwargs)

def get_mass_matrix(fs):
    backend = parameters['linear_algebra_backend']
    parameters['linear_algebra_backend'] = "Eigen"
    u = TrialFunction(fs)
    v = TestFunction(fs)
    mass = inner(u, v) * dx
    mass = assemble(mass)

    M = sps.csr_matrix(as_backend_type(mass).sparray())
    parameters['linear_algebra_backend'] = backend
    return M

def get_stiffness_matrix(fs):
    backend = parameters['linear_algebra_backend']
    parameters['linear_algebra_backend'] = "Eigen"
    u = TrialFunction(fs)
    v = TestFunction(fs)
    stiffness = inner(grad(u), grad(v)) * dx
    stiffness = assemble(stiffness)

    S = sps.csr_matrix(as_backend_type(stiffness).sparray())
    parameters['linear_algebra_backend'] = backend
    return S

def cholesky(S):
    eps = np.finfo(S.dtype).eps

    m = sps.diags(1/np.sqrt(S.diagonal()))
    mI = sps.diags(np.sqrt(S.diagonal()))
    T = m.dot(S).dot(m)
    T.eliminate_zeros()

    factor = cholmod.cholesky(T.tocsc(), eps)
    PL,DD = factor.L_D()
    DD_diag = DD.diagonal()
    DD_diag[DD_diag<0] = 0
    D = sps.diags(np.sqrt(DD_diag))
    L = mI.dot(factor.apply_Pt(PL.tocsc()).dot(D))
    P = factor.apply_P(sps.eye(S.shape[0], format='csc'))

    assert np.linalg.norm((L.dot(L.T)-S).data) < 1e-12
    assert len(sps.triu(P.dot(L), 1).data) == 0

    return P, L

def load_problem(info):
    problemName = info['problem']['name']
    log(f"Loading problem: {problemName}")
    Problem = __import__(f"problem.{problemName}", fromlist=["Problem"]).Problem
    problem = Problem(info)
    return problem

if __name__=='__main__':
    descr = """Compute the stiffness matrix `S`, its sparse Cholesky factorization `L L^T = P S P^T` for the given problem."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('PROBLEM', help='path to the directory where the results will be stored. The problem specification is assumed to lie in (PROBLEM/parameters.json)path to the directory where the samples will be stored. The problem specification is assumed to lie in (PROBLEM/parameters.json)')
    args = parser.parse_args()

    problemDir = args.PROBLEM
    if not os.path.isdir(problemDir):
        raise IOError(f"PROBLEM '{problemDir}' is not a directory")

    problemDirContents = os.listdir(problemDir)
    if not 'parameters.json' in problemDirContents:
        raise IOError(f"'{problemDir}' does not contain a 'parameters.json'")
    if 'orthogonalization.npz' in problemDirContents:
        raise IOError(f"'{problemDir}' already contains other data ('{problemDir}/orthogonalization.npz')")

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

    orthogonalizationFile = f"{problemDir}/orthogonalization.npz"

    log(f"Computing gramian")
    S = get_stiffness_matrix(V)
    log(f"Computing Cholesky factorization")
    P,L = cholesky(S)
    log(f"Saving orthogonalization: '{orthogonalizationFile}'")
    np.savez_compressed(orthogonalizationFile, gramian=S, choleskyFactor=L, permutation=P)

# coding: utf-8
import argparse, os, json, time

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.hermite_e import hermeval
from scipy.special import factorial
import xerus as xe

def log(*args, **kwargs):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()), *args, **kwargs)

def isInt(s):
    try:
        int(s)
        return True
    except:
        return False

tensor = lambda arr: xe.Tensor.from_buffer(np.ascontiguousarray(arr))

if __name__=='__main__':
    descr = """Compute the reconstruction for the given problem."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('PROBLEM', help='path to the directory where the results will be stored. The problem specification is assumed to lie in (PROBLEM/parameters.json)path to the directory where the samples will be stored. The problem specification is assumed to lie in (PROBLEM/parameters.json)')
    parser.add_argument('SAMPLES', type=int, help='the number of samples to use')
    parser.add_argument('-m', '--modes', dest='MODES', type=int, default=0, help='the number of modes to use (default: all)')
    parser.add_argument('-s', '--solver', dest='SOLVER', type=str, choices=["uq_ra_adf", "uqSALSA"], default="uq_ra_adf", help='the solver to use (default: uq_ra_adf)')
    args = parser.parse_args()

    # basis = "fourier"
    basis = "hermite"
    # basis = "unboundedFourier"

    problemDir = args.PROBLEM
    if not os.path.isdir(problemDir):
        raise IOError(f"PROBLEM '{problemDir}' is not a directory")

    problemDirContents = os.listdir(problemDir)
    if not 'parameters.json' in problemDirContents:
        raise IOError(f"'{problemDir}' does not contain a 'parameters.json'")
    reconstructionFile = f"reconstruction_N{args.SAMPLES}_M{args.MODES}_{basis}_{args.SOLVER}.xrs"
    if reconstructionFile in problemDirContents:
        raise IOError(f"'{problemDir}' already contains other data ('{problemDir}/{reconstructionFile}')")

    log("Loading precomputed samples")
    samples = []
    values  = []
    numSamples = 0
    problemDirContents = os.listdir(problemDir)
    for fileName in problemDirContents:
        if not (fileName.endswith('.npz') and isInt(fileName[:-4])):
            continue
        z = np.load(os.path.join(problemDir, fileName))
        samples.append(z['ys'])
        values.append(z['us'])
        if len(samples[-1]) != len(values[-1]):
            raise RuntimeError(f"Number of Samples and number of Solutions differ in file '{problemDir}/{fileName}' ({len(samples[-1])} != {len(values[-1])})")
        numSamples += len(samples[-1])
        if numSamples >= args.SAMPLES:
            break
    if numSamples == 0:
        raise IOError(f"'{problemDir}' does not contain any samples")
    elif numSamples < args.SAMPLES:
        raise IOError(f"Not enough samples: {args.SAMPLES - numSamples} missing")
    elif numSamples > args.SAMPLES:
        samples[-1] = samples[-1][:-(numSamples-args.SAMPLES)]
        values[-1]  = values[-1][:-(numSamples-args.SAMPLES)]

    samples = np.concatenate(samples, axis=0)
    values = np.concatenate(values, axis=0)
    assert len(samples) == args.SAMPLES and len(values) == args.SAMPLES, f"{samples.shape} vs {values.shape}"

    if args.MODES > 0:
        if args.MODES > samples.shape[1]:
            raise IOError(f"Not enough modes: {args.MODES} expected but only {samples.shape[1]} are given")
        samples = samples[:,:args.MODES]

    log("Loading Cholesky factorization of Gramian")
    if "orthogonalization.npz" not in problemDirContents:
        raise IOError(f"'{problemDir}' does not contain an orthogonalization file")
    z = np.load(os.path.join(problemDir, 'orthogonalization.npz'), allow_pickle=True)
    L = z['choleskyFactor'][()]  # z['choleskyFactor'] == array(SPARSE, dtype=object, shape=())

    log("Orthogonalize samples")
    values = L.dot(values.T).T

    N,x_dim = values.shape  # number of samples and physical dimension
    M = samples.shape[1] + 1

    # y_dims = [10]*(M-1)
    y_dims = [7]*(M-1)

    # renormalization factors to obtain orthonormal polynomials
    # factors = np.sqrt(2*np.arange(M))             # Legendre
    # factors = np.sqrt(2*np.arange(M))             # Fourier (1)
    if basis == "fourier":
        from fbasis import fourier_basis
        # Phi is the CDF of the probability measure rho that is used for integration.
        # For the uniform distribution on the interval [-1,1] this CDF is given by
        Phi = lambda x: (x+1)/2
        factors = np.empty(max(y_dims), dtype=float)
        factors[0] = 1
        factors[1:] = np.sqrt(2)
        basis_measures     = lambda m: fourier_basis(y_dims[m], samples[:,m], Phi).T
        basis_basisWeights = lambda m: np.diag(factors[:y_dims[m]])
    elif basis == "hermite":
        factors = 1/np.sqrt(factorial(np.arange(max(y_dims))))  # Hermite
        basis_measures     = lambda m: hermeval(samples[:,m], np.diag(factors[:y_dims[m]])).T
        basis_basisWeights = lambda m: np.diag(1/factors[:y_dims[m]]**2)
    elif basis == "unboundedFourier":
        from fbasis import fourier_basis, Phi
        # Phi is the CDF of the probability measure rho that is used for integration.
        # For the uniform distribution on the interval [-1,1] this CDF is given by
        factors = np.empty(max(y_dims), dtype=float)
        factors[0] = 1
        factors[1:] = np.sqrt(2)
        basis_measures     = lambda m: fourier_basis(y_dims[m], samples[:,m], Phi).T
        basis_basisWeights = lambda m: np.diag(factors[:y_dims[m]])
    else:
        raise RuntimeError(f"Basis '{basis}' is not defined")


    log(f"Compute rank-one measures (basis: {basis})")
    meas = []
    for m in range(M-1):
        measm = basis_measures(m)
        assert measm.shape == (N, y_dims[m])
        meas.append(tensor(measm))
    assert len(meas) == M-1

    vals = tensor(values)

    # For Legendre polynomials the supremum norm of the normalized polynomials is equal to the normalization factor.
    # Hermite polynomials are unbounded. Nevertheless we use this factor for weighted sparsity.
    basisWeights = [tensor(np.eye(x_dim))]
    for m in range(M-1):
        basisWeights.append(tensor(basis_basisWeights(m)))
        # basisWeights.append(tensor(np.diag(factors[:y_dims[m]])))       # Legendre
        # d = (m+1)*np.arange(1, d+1, dtype=float)
        # d *= xe.frob_norm(vals)**(1/(M-1)) / np.linalg.norm(d)
        # basisWeights.append(tensor(np.diag(d)))       # Fourier (2)
    assert len(basisWeights) == M
    assert basisWeights[0].dimensions == [x_dim]*2
    for m in range(1,M):
        assert basisWeights[m].dimensions == [y_dims[m-1]]*2



    if args.SOLVER == "uq_ra_adf":
        # =============
        #  Use UqRaADF
        # =============

        # assert np.all(np.asarray(y_dims[1:]) == y_dims[0])
        # meas = []
        # for m in range(M-1):
        #     meas.append(basis_measures(m))
        # meas = np.transpose(meas, (1,0,2))
        # assert np.shape(meas) == (numSamples, M-1, y_dims[0]), f"NOT {np.shape(meas)} == {(numSamples, M-1, y_dims[0])}"
        # meas = [[tensor(cmp_m) for cmp_m in m] for m in meas]
        # assert values.shape == (numSamples, x_dim)
        # vals = [tensor(v) for v in values]
        # log(f"Run reconstruction", flush=True)
        # reco = xe.uq_ra_adf(meas, vals, [x_dim]+y_dims, targeteps=1e-6, maxitr=5000)

        _meas = []
        for m in range(M-1):
            _meas.append(basis_measures(m))
        # meas = np.transpose(meas, (1,0,2))
        # assert np.shape(meas) == (numSamples, M-1, y_dims[0]), f"NOT {np.shape(meas)} == {(numSamples, M-1, y_dims[0])}"
        meas = [list() for _ in range(numSamples)]
        for mode_meas in _meas:
            for e,sample_meas in enumerate(mode_meas):
                meas[e].append(tensor(sample_meas))
        # meas = [[tensor(cmp_m) for cmp_m in m] for m in meas]
        assert values.shape == (numSamples, x_dim)
        vals = [tensor(v) for v in values]
        log(f"Run reconstruction", flush=True)
        reco = xe.uq_ra_adf(meas, vals, [x_dim]+y_dims, targeteps=1e-6, maxitr=5000)

    elif args.SOLVER == "uqSALSA":
        # ===========
        #  Use SALSA
        # ===========

        log(f"Compute initial value")
        init = xe.TTTensor.random([x_dim]+y_dims, [1]*len(y_dims))
        init.set_component(0, tensor(np.mean(values, axis=0).reshape(1,x_dim,1)))
        for m in range(len(y_dims)):
            init.set_component(m+1, xe.Tensor.dirac([1,y_dims[m],1], [0]*3))

        log(f"Run reconstruction", flush=True)
        solver = xe.uqSALSA(init, meas, vals)

        # Parameters from salsa-uniform-fourier
        solver.targetResidual = 1e-6
        solver.maxSweeps = 5000
        solver.maxStagnatingEpochs = 500
        solver.maxIRsteps = 10
        solver.basisWeights = basisWeights
        solver.maxRanks = [15] + [7]*(M-2)


        # solver.targetResidual = 1e-6
        # solver.trackingPeriodLength = 30
        # solver.maxSweeps = 5000
        # solver.maxStagnatingEpochs = 500
        # # solver.maxIRsteps = 10
        # # solver.alphaFactor = 10
        # # solver.omegaFactor = 5
        # # solver.maxIRsteps = 0
        # # solver.alphaFactor = 0
        # solver.falpha = 1.1
        # solver.fomega = 1.1
        # solver.basisWeights = basisWeights
        # solver.maxRanks = [15] + [7]*(M-2)

        solver.run()
        reco = solver.bestState.x
    else:
        raise NotImplementedError(f"Unknown solver {args.SOLVER}")

    log(f"Reconstruction dimensions: {reco.dimensions}")
    log(f"Reconstruction ranks:      {reco.ranks()}")

    xe.save_to_file(reco, f"{problemDir}/{reconstructionFile}", xe.FileFormat.BINARY)

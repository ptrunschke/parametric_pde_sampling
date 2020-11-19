# -*- coding: utf-8 -*-
from __future__ import division
from dolfin import interpolate, Expression, Function, Constant
import numpy as np
from scipy.special import zeta


class TestField:
    """
    Artificial M-term KL (plane wave Fourier modes similar to those in EGSZ).
    Diffusion coefficient reads (in linear case)

        a(x,y) = mean + scale * \sum_{m=1}^M (m+1)^decay y_m \sin(\pi b_1 x_1) \sin(\pi b_2 x_2)

    for b_1 = floor( (m+2)/2 ) and b_2 = ceil( (m+2)/2 ). In exponential case,

        b(x,y) = exp(a(x,y)).

    The field is log-normal for expfield=True.

    Parameters:
    -----------
    M : int
        Number of terms in expansion.
    k : int
        Modes to differentiate.
    mean : float, optional
        Mean value of the field.
        (defaults to 1.0)
    scale : float, optional
        Scaling coefficient for the centered terms of the expansion.
        (defaults to 1.0)
    decay : float, optional
        Decay rate for the terms.
        (defaults to 0.0)
    expfield : bool, optional
        Switch to choose lognormal field.
        (defaults to False)
    """
    def __init__(self, M, k=[], mean=1.0, scale=1.0, decay=0.0, expfield=False):
        assert M > 0, 'number of terms in expansion has to be positive'
        assert isinstance(k, list)
        assert decay <= 0
        if not expfield:
            assert mean >= scale*sum(m**decay for m in range(1,M+1)), "{} vs {}".format(mean, scale*sum(m**decay for m in range(1,M+1)))
        # create a Fenics expression of the affine field
        self.a = Expression('sin(pi*F1*x[0]) * sin(pi*F2*x[1])', F1=0, F2=0, degree=5)
        self.M = M
        self.k = k
        self.mean = mean
        self.scale = scale
        self.decay = decay
        self.expfield = expfield

    def realisation(self, y, V):  # type: (List[float], FunctionSpace) -> Function
        """
        Compute an interpolation of the random field on the FEM space V for the given sample y.

        Parameters:
        -----------
        y : list of floats
            Parameters.
        V : FunctionSpace

        Returns
        -------
        out : ndarray
            FEM coefficients for the field realisation.
        """
        assert len(y) == self.M, 'number of parameters differs from the number of terms in the expansion'
        a = self.a
        k = self.k
        decay = self.decay
        scale = self.scale
        mean = self.mean

        def indexer(i):
            m1 = np.floor(i/2)
            m2 = np.ceil(i/2)
            return m1, m2

        #@cache  #TODO: klepto
        def summand(f1, f2):
            a.F1, a.F2 = f1, f2
            return interpolate(a, V).vector().get_local()

        def linear_part(ys, ms):
            x = Function(V).vector().get_local()  # zero
            assert np.all(x == 0)
            for m in ms:
                x += (m+1)**decay * ys[m] * summand(*indexer(m+2))
            x *= scale
            return x

        if not self.expfield:
            if len(k) == 0:
                x = mean + linear_part(y, range(self.M))
            elif len(k) == 1:
                x = linear_part(np.ones(self.M), k)
            else:
                x = linear_part(np.ones(self.M), [])
        else:
            x = np.exp(mean + linear_part(y, range(self.M)))
            for ki in k:
                x *= linear_part(np.ones(self.M), [ki])

        f = Function(V)
        f.vector().set_local(x)
        return f

class TestFieldEGSZ13:
    """
    The diffusion coefficient as in [EGSZ13] reads

        a(x,y) = a_0(x) + \sum_{m=1}^M y_m a_m(x)

    for a_0(x) = 1 and

        a_m(x) = \bar(alpha) m^{-\tilde{sigma}} \cos(2\pi\beta_1(m) x_1) \cos(2\pi\beta_2(m) x_2).

    Here for k(m) = floor( -1/2 + \sqrt{1/4 +2m} ), we set

        \beta_1(m) = m - k(m) (k(m)+1)/2         and        \beta_2(m) = k(m) - \beta_1(m)

    We choose for \gamma=0.9, that \bar{\alpha} = \gamma / \zeta{\tilde{sigma}}
    where \zeta is the Riemann zeta function.

    Parameters:
    -----------
    M : int
        Number of terms in expansion.
    decay : float
        Decay rate for the terms.
    mean : float, optional
        Mean value of the field. (defaults to 1.0)
    expfield : bool
        If true, return exp( a(x,y) ) instead of a(x,y)
    """
    def __init__(self, M, decay, mean=1.0, expfield=False):
        assert M > 0, 'number of terms in expansion has to be positive'
        assert decay >= 0
        # TODO assert that return is positive
        # create a Fenics expression of the affine field
        self.a = Expression('cos(2*pi*F1*x[0]) * cos(2*pi*F2*x[1])', F1=0, F2=0, degree=10)
        self.M = M
        self.mean = mean # a_0(x)
        self.decay = decay # \tilde{sigma}
        self.gamma = 0.9
        self.a_bar = self.gamma/zeta(self.decay)
        self.expfield = expfield

    def realisation(self, y, V):  # type: (List[float], FunctionSpace) -> Function
        """
        Compute an interpolation of the random field on the FEM space V for the given sample y.

        Parameters:
        -----------
        y : list of floats
            Parameters.
        V : FunctionSpace

        Returns
        -------
        out : ndarray
            FEM coefficients for the field realisation.
        """
        assert len(y) == self.M, 'number of parameters differs from the number of terms in the expansion'
        a = self.a

        def k(m): return np.floor(-0.5+np.sqrt(0.25 + 2*m))
        def beta1(m): return m - k(m) * (k(m)+1)/2
        def beta2(m): return k(m) - beta1(m)

        def summand(f1, f2): # this is: cos(1\pi\beta1(m) x_1) cos(2\pi\beta2(m) x_2)
            a.F1, a.F2 = f1, f2
            return interpolate(a, V).vector().get_local()

        def linear_part(ys, ms): # this is: sum_{m=1}^M y_m * a_m(x)
            x = Function(V).vector().get_local()  # zero
            assert np.all(x == 0)
            for m in ms:
                # NOTE index ys[m-1] requires m-1, since m=1,...,M
                x += ys[m-1] * self.a_bar * m**(-self.decay) * summand(beta1(m),beta2(m))
            return x

        a_val = self.mean + linear_part(y, range(1,self.M+1))
        if self.expfield: a_val = np.exp(a_val)
        f = Function(V)
        f.vector().set_local(a_val)
        return f

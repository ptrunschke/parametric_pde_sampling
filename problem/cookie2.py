# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from dolfin import *
set_log_level(WARNING)
from field.cookiefield import CookieField
from parallel import ParallelizableProblem


class Problem(ParallelizableProblem):
    def __init__(self, info):
        assert info['problem']['name'] == "cookie2"
        self.info = info

        # setup fe space
        self.space = FunctionSpace(self.mesh, 'CG', self.degree)

        self.D = info['problem']['subdomains per axis']
        self.kappa = info['problem']['kappa']

        # setup random field
        assert info['expansion']['size'] == 2*self.D**2
        self.field = CookieField(self.D, prepare_cookie_subdomain_data, kappa=self.kappa)

        # define forcing term
        self.forcing = Constant(1)

        # define boundary condition
        self.bc = DirichletBC(self.space, Constant(0.0), 'on_boundary')


    def solution(self, y):
        V = self.space
        f = self.forcing
        bc = self.bc

        mu = self.field.realisation(y, V)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = mu * inner(grad(u), grad(v)) * dx
        L = f * v * dx

        u = Function(V)
        solve(a == L, u, bc)

        return u.vector().get_local(), y


def cart(*args):
    "Cartesian product"
    return np.array(np.meshgrid(*args)).T.reshape(-1, len(args))


def prepare_cookie_subdomain_data(fyc, cookie_D):
    # prepare cookie subdomain data
    fy1 = (fyc+1)/2    # rescale `fyc` to be in [0,1]
    cx = 1 / cookie_D  # subdomain size
    rx = cx / 2        # maximal cookie radius

    coeff_eps = 1e-1
    idcs = np.arange(cookie_D)
    centers = rx + cx*cart(idcs, idcs).T        # centers of the cookies
    radii   = rx * (3 + fyc[cookie_D ** 2:])/4  # radii of the cookies
    coeffs  = coeff_eps + fy1[:cookie_D ** 2]   # coefficients inside the cookies

    # data format: (center[0], center[1], coefficient, radius)
    subdomain_data = np.stack([centers[0], centers[1], coeffs, radii]).T
    return subdomain_data

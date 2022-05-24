# -*- coding: utf-8 -*-
from __future__ import division, print_function
from dolfin import *

set_log_level(LogLevel.WARNING)
from parametric_pde_sampling.field.testfield import TestField, TestFieldEGSZ13
from .parallel import ParallelizableProblem


class Problem(ParallelizableProblem):
    def __init__(self, info):
        assert info["problem"]["name"] == "darcy"
        self.info = info

        # setup fe space
        self.space = FunctionSpace(self.mesh, "CG", self.degree)
        self.space_coef = FunctionSpace(self.mesh, "DG", self.degree - 1)

        # setup random field
        M = self.info["expansion"]["size"]
        mean = self.info["expansion"]["mean"]
        scale = self.info["expansion"]["scale"]
        decay = -self.info["expansion"]["decay rate"]
        expfield = self.info["sampling"]["distribution"] == "normal"
        self.field = TestField(M, mean=mean, scale=scale, decay=decay, expfield=expfield)
        # self.field = TestFieldEGSZ13(M, decay=-decay, mean=mean, expfield=expfield)

        # define forcing term
        self.forcing = Constant(1)

        # define boundary condition
        self.bc = DirichletBC(self.space, Constant(0.0), "on_boundary")

    def coefficient(self, y):
        Vc = self.space_coef
        return self.field.realisation(y, Vc)

    def coefficient_vector(self, y):
        Vc = self.space_coef
        kappa = self.coefficient(y)
        return interpolate(kappa, Vc).vector().get_local()

    def solution(self, y):
        """
        Return solution of Darcy problem for given parameter realization y.

        Parameter
        ---------
        y   :   array_like
                Sample for realization of the problem.

        Returns
        -------
        u   :   solution vector (numpy array)
        """
        V = self.space
        f = self.forcing
        bc = self.bc
        kappa = self.coefficient(y)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(grad(u), grad(v)) * dx
        L = f * v * dx

        u = Function(V)
        solve(a == L, u, bc)

        return u.vector().get_local()

    def application(self, y_u):
        y, u_vec = y_u
        M = self.info["expansion"]["size"]
        assert y.shape == (M,)

        V = self.space
        f = self.forcing
        bc = self.bc
        kappa = self.coefficient(y)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(grad(u), grad(v)) * dx
        L = f * v * dx

        A, b = assemble_system(a, L, bc)
        u = Function(V).vector()
        u[:] = u_vec
        return (A * u).get_local()

    def residuum(self, y_u):
        y, u_vec = y_u
        M = self.info["expansion"]["size"]
        assert y.shape == (M,)

        V = self.space
        Vc = self.space_coef
        f = self.forcing
        kappa = self.field.realisation(y, Vc)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(grad(u), grad(v)) * dx
        L = f * v * dx

        bc = DirichletBC(V, Constant(0.0), "on_boundary")

        A, b = assemble_system(a, L, bc)
        u = Function(V).vector()
        u[:] = u_vec
        res = (A * u - b).get_local()
        assert res.shape == u_vec.shape
        return res

    def P1_residual_estimator(self, y_u):
        V = self.space
        Vc = self.space_coef

        mesh = V.mesh()
        try:
            h = CellDiameter(V.mesh())
        except NameError:
            h = CellSize(V.mesh())
        DG0 = FunctionSpace(mesh, "DG", 0)
        dg0 = TestFunction(DG0)

        y, u_vec = y_u
        u = Function(V)
        u.vector().set_local(u_vec)
        f = self.forcing
        kappa = self.field.realisation(y, Vc)

        R_T = -(f + div(kappa * grad(u)))
        R_dT = kappa * grad(u)
        J = jump(R_dT)
        indicator = h**2 * (1 / kappa) * R_T**2 * dg0 * dx + avg(h) * avg(1 / kappa) * J**2 * avg(dg0) * 2 * dS

        eta_res_local = assemble(indicator, form_compiler_parameters={"quadrature_degree": -1})
        return eta_res_local.get_local()

    def refine_mesh(self, marked_cells):
        marker = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
        marker.set_all(False)
        marker.array()[marked_cells] = True  # for idx in marked_cells: marker[idx] = True
        self.mesh = refine(self.mesh, marker)
        self.space = FunctionSpace(self.mesh, "CG", self.degree)

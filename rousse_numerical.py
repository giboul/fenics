# %%
import ufl
import numpy

import pyvista
from dolfinx.plot import vtk_mesh

from mpi4py import MPI

from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem


def q(u):
    return 1 + u**2

domain = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2 * x[1]
f = -ufl.div(q(u_ufl) * ufl.grad(u_ufl))

V = fem.functionspace(domain, ("Lagrange", 1))
def u_exact(x): return eval(str(u_ufl))
u_D = fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

uh = fem.Function(V)
v = ufl.TestFunction(V)
F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_monitor": None,
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_monitor": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}

problem = NonlinearProblem(
    F,
    uh,
    bcs=[bc],
    petsc_options=petsc_options,
    petsc_options_prefix="nonlinpoisson",
)

problem.solve()
converged = problem.solver.getConvergedReason()
num_iter = problem.solver.getIterationNumber()
assert converged > 0, f"Solver did not converge, got {converged}."
print(
    f"Solver converged after {num_iter} iterations with converged reason {converged}."
)

# %%
p = 4 * ufl.exp((x[0] ** 2 + (x[1]) ** 2))
Q = fem.functionspace(domain, ("Lagrange", 5))
expr = fem.Expression(p, Q.element.interpolation_points)
F_eval = fem.Function(Q)
F_eval.interpolate(expr)

topology, cell_types, x = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.point_data["u"] = uh.x.array
warped = grid.warp_by_scalar("u", factor=25)

plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
plotter.show()

if False:
    from mpi4py import MPI
    from petsc4py.PETSc import ScalarType  # type: ignore

    import numpy as np
    from matplotlib import pyplot as plt
    import pyvista

    import ufl
    from dolfinx import fem, mesh, plot
    from dolfinx.fem.petsc import LinearProblem

    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (2.0, 1.0)),
        n=(32, 16),
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    def darcy_face(x):
        return np.isclose(x[0], 0.0)

    facets = mesh.locate_entities_boundary(
        msh,
        dim=msh.topology.dim-1,
        marker=darcy_face,
    )

    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(1), dofs=dofs, V=V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    ###
    alpha = 1.0
    beta  = 0.5
    gamma = 0.1
    eps   = 0.2   # ε  (avoid shadowing Python built-in)
    phi   = 0.3
    zeta  = 0.05
    eta   = 1.0

    # Shorthand for spatial derivative
    Du = u.dx(0)
    Dv = v.dx(0)
    
    # Coefficient groups
    A = alpha + beta * u + gamma * u**2      # multiplies  u''  →  -A * u'v'
    B = eps   + phi   * u + zeta  * u**2    # multiplies  u'   →  +B * u'v
    C = eta                                  # multiplies  u    →  +C * u v

    # Residual F(u; v) = 0
    F = (
        - A * Du * Dv           # integration-by-parts of A u''
        + B * Du * v            # first-order term
        + C * u  * v            # zeroth-order term
    ) * ufl.dx
    ###

    f = 10 * ufl.exp(-((x[0] - 0.5) **2 + (x[1] - 0.5) ** 2) / 0.02)
    g = ufl.sin(5 * x[0])

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="demo_poisson_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True
        },
    )
    uh = problem.solve()
    assert isinstance(uh, fem.Function)

    cells, types, x = plot.vtk_mesh(V)
    plt.plot(x[:, :2], uh.x.array.real, '.')
    plt.show()

    # exit()

    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    plotter.show()
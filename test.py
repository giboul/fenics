"""
Solver for the nonlinear ODE using FEniCS/DOLFINx 0.10.0:

  α u'' + β u u'' + γ u² u'' + ε u' + φ u u' + ζ u² u' + η u = 0

Weak (variational) form after integration by parts:

  ∫ [ -(α + βu + γu²) u'v'  +  (ε + φu + ζu²) u'v  +  η u v ] dx = 0
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, log
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
    create_vector,
)
import ufl

# ── 1. Coefficients ──────────────────────────────────────────────────────────

alpha = 1.0
beta  = 0.5
gamma = 0.1
eps   = 0.2
phi   = 0.3
zeta  = 0.05
eta   = 1.0

# ── 2. Mesh & function space ─────────────────────────────────────────────────

L = 10.0
N = 200

domain = mesh.create_interval(MPI.COMM_WORLD, N, [0.0, L])
V = fem.functionspace(domain, ("CG", 2))

# ── 3. Boundary conditions ───────────────────────────────────────────────────

def left(x):
    return np.isclose(x[0], 0.0)

def right(x):
    return np.isclose(x[0], L)

dofs_left  = fem.locate_dofs_geometrical(V, left)
dofs_right = fem.locate_dofs_geometrical(V, right)

bc_left  = fem.dirichletbc(PETSc.ScalarType(1.0), dofs_left,  V)
bc_right = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_right, V)
bcs = [bc_left, bc_right]

# ── 4. Variational problem ───────────────────────────────────────────────────

u = fem.Function(V)
v = ufl.TestFunction(V)

Du = u.dx(0)
Dv = v.dx(0)

A_coeff = alpha + beta * u + gamma * u**2
B_coeff = eps   + phi   * u + zeta  * u**2

F_form = (
    - A_coeff * Du * Dv
    + B_coeff * Du * v
    + eta * u  * v
) * ufl.dx

J_form = ufl.derivative(F_form, u)

# Compile forms
F_compiled = fem.form(F_form)
J_compiled = fem.form(J_form)

# ── 5. Initial guess ─────────────────────────────────────────────────────────

x_coords = V.tabulate_dof_coordinates()[:, 0]
u.x.array[:] = 1.0 - x_coords / L   # linear from u(0)=1 to u(L)=0

# ── 6. Newton solver (manual loop) ───────────────────────────────────────────

A_mat = create_matrix(J_compiled)
b_vec = create_vector(F_compiled)

ksp = PETSc.KSP().create(domain.comm)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)

du = fem.Function(V)   # Newton increment

max_iter  = 50
rtol      = 1e-8
atol      = 1e-10

print("Newton iterations:")
print(f"{'Iter':>5}  {'|F|':>14}  {'|du|':>14}")
print("-" * 40)

for iteration in range(max_iter):

    # Assemble residual
    with b_vec.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector(b_vec, F_compiled)
    apply_lifting(b_vec, [J_compiled], [bcs], x0=[u.x.petsc_vec], alpha=-1.0)
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_vec, bcs, x0=u.x.petsc_vec, alpha=-1.0)

    res_norm = b_vec.norm(PETSc.NormType.NORM_2)

    # Assemble Jacobian
    A_mat.zeroEntries()
    assemble_matrix(A_mat, J_compiled, bcs=bcs)
    A_mat.assemble()

    # Solve J du = -F
    ksp.setOperators(A_mat)
    ksp.solve(b_vec, du.x.petsc_vec)
    du.x.scatter_forward()

    du_norm = du.x.petsc_vec.norm(PETSc.NormType.NORM_2)
    print(f"{iteration:>5}  {res_norm:>14.6e}  {du_norm:>14.6e}")

    # Update solution
    u.x.array[:] += du.x.array
    u.x.scatter_forward()

    if du_norm < atol + rtol * u.x.petsc_vec.norm():
        print(f"\n✓ Converged in {iteration + 1} iterations.")
        break
else:
    print("\n✗ Newton did not converge. Try a better initial guess.")

# ── 7. Post-processing ───────────────────────────────────────────────────────

sort_idx = np.argsort(x_coords)
x_sorted = x_coords[sort_idx]
u_sorted = u.x.array[sort_idx]

print("\nx\t\tu(x)")
print("-" * 30)
for xi, ui in zip(x_sorted[::20], u_sorted[::20]):
    print(f"{xi:.4f}\t\t{ui:.6f}")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(x_sorted, u_sorted, lw=2, color="steelblue")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Nonlinear ODE solution (DOLFINx 0.10.0)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("solution.png", dpi=150)
    plt.show()
    print("Plot saved to solution.png")
except ImportError:
    pass

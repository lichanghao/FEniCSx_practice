import os

import numpy as np
import ufl
from dolfinx.fem import Function, FunctionSpace
from dolfinx.fem import assemble_scalar, form
from dolfinx.fem import locate_dofs_topological, dirichletbc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from ufl import dx, grad, inner

from dolfinx import log, plot

try:
    import pyvista as pv
    import pyvistaqt as pvqt
    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

# Save all logging to file
log.set_output_file("log.txt")

eps1 = 3.0e-02  # surface parameter
M1 = 500
M2 = 50
dt = 5.0e-06  # time step
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson

msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
ME = FunctionSpace(msh, P1 * P1)
q, v = ufl.TestFunctions(ME)
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)

# Zero u
u.x.array[:] = 0.0

# Interpolate initial condition
u.sub(0).interpolate(lambda x: 0.63 + 0.2 * (0.5 - np.random.rand(x.shape[1])))
u.x.scatter_forward()

# Compute the chemical potential df/dc
# c = ufl.variable(c)
# f = 100 * c**2 * (1 - c)**2
# dfdc = ufl.diff(f, c)

# mu_(n+theta)
mu_mid = (1.0 - theta) * mu0 + theta * mu
c_mid = (1.0 - theta) * c0 + theta * c

# IMPORTANT (Qiang Du, 2004)
# Weak statement of the equations
F0 = inner(c, q) * dx - inner(c0, q) * dx - dt * eps1 * inner(grad(mu_mid), grad(q)) * dx - dt * 1 / eps1 * inner(mu_mid * (3 * c_mid**2 - 1), q) * dx
F1 = inner(mu + 1 / eps1**2 * (c**2 - 1) * c, v) * dx + inner(grad(c), grad(v)) * dx
F2 = dt * M1 * inner((assemble_scalar(form(c_mid * dx)) - (-0.3)), q) * dx
F = F0 + F1 + F2

# Dirichlet boundary conditions
def boundary_marker(x):
    x_marker = np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    y_marker = np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    return np.logical_or(x_marker, y_marker)

facets = locate_entities_boundary(msh, dim=1,
                                       marker=boundary_marker)
dofs = locate_dofs_topological(V=ME.sub(0), entity_dim=1, entities=facets)
bc = dirichletbc(value=ScalarType(-1), dofs=dofs, V=ME.sub(0))


# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u, bcs=[bc])
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6

# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

# Output file
file = XDMFFile(MPI.COMM_WORLD, "demo_ch/output.xdmf", "w")
file.write_mesh(msh)

# Step in time
t = 0.0

#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 3000 * dt

# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
V0, dofs = ME.sub(0).collapse()

# Prepare viewer for plotting the solution during the computation
if have_pyvista:
    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.create_vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    grid.point_data["c"] = u.x.array[dofs].real
    grid.set_active_scalars("c")

    p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    p.add_mesh(grid, clim=[-1, 1])
    p.view_xy(True)
    p.add_text(f"time: {t}", font_size=12, name="timelabel")

c = u.sub(0)
u0.x.array[:] = u.x.array
while (t < T):
    t += dt
    r = solver.solve(u)
    print(f"Step {int(t/dt)}: num iterations: {r[0]}")
    print(f"Concentration integral: {assemble_scalar(form(c * dx))}")
    u0.x.array[:] = u.x.array
    file.write_function(c, t)

    # Update the plot window
    if have_pyvista:
        p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        grid.point_data["c"] = u.x.array[dofs].real
        p.app.processEvents()

file.close()

# Update ghost entries and plot
if have_pyvista:
    u.x.scatter_forward()
    grid.point_data["c"] = u.x.array[dofs].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = "c.png"
    pv.plot(grid, show_edges=True, screenshot=screenshot)
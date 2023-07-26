import os
import numpy as np
import ufl
from dolfinx.fem import Function, FunctionSpace
from dolfinx.fem import assemble_scalar, form, assemble
from dolfinx.fem import locate_dofs_topological, dirichletbc
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from ufl import dx, grad, inner

from dolfinx import log, plot

# detect pyvista for visulization
try:
    import pyvista as pv
    import pyvistaqt as pvqt
    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visuzalize the solution")
    have_pyvista = False

# Save all logging to file
log.set_output_file("log.txt")

# General parameters (900K) (Units: SI, um)
F = 96485                               # Faradic constant
R = 8.314                               # Gas constant
T = 900 + 273                           # Absolute temperature
a = F / R / T                           # F/(RT)
j0 = 2.14e-10                           # Exchange current density
drt = 4.11e-5 / R / T                   # D/(RT)
rho_TPB = 10                            # Density of triple phase boundaries
s_io = 8.4e-6                           # Ionic conductivity
s_el = 2.17                             # Electronic conductivity
f_Ni = 0.33                             # Fraction for Ni metal
f_YSC = 0.46                            # Fraction for YSC ceramics
f_pore = 1 - f_Ni - f_YSC               # Fraction for pores

# Geometry, mesh, function space
msh = create_unit_square(MPI.COMM_WORLD, 50, 50, CellType.triangle)
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
ME = FunctionSpace(msh, P1 * P1 * P1 * P1)
u_test = ufl.TestFunction(ME)
u = Function(ME)  # current solution

# Split mixed function spaces
V_Phyd = ME.sub(1)                      # Hydrogen gas pressure function space
V_Pwater = ME.sub(0).sub(1)             # Water vapor pressure function space
V_Vel = ME.sub(0).sub(0).sub(1)         # Electronic potential function space
V_Vio = ME.sub(0).sub(0).sub(0)         # Ionic potential function space

# Split mixed functions
remain, Phyd = ufl.split(u)             # Phyd: Hydrogen gas pressure
remain, Pwater = ufl.split(remain)      # Pwater: Water vapor pressure
Vio, Vel = ufl.split(remain)            # Vel / Vio: Electronic / Ionic potential
remain, Phyd_t = ufl.split(u_test)      # Phyd_t: test function for Phyd
remain, Pwater_t = ufl.split(remain)    # Pwater_t: test function for Pwater
Vio_t, Vel_t = ufl.split(remain)        # Vel_t / Vio_t: test function for Vel / Vio

# Initialization
u.x.array[:] = 0.1                      # can be any number except for zero

# Boundary conditions
def left_boundary_marker(x):
    x_marker = np.isclose(x[0], 0.0)
    return x_marker

def right_boundary_marker(x):
    x_marker = np.isclose(x[0], 1.0)
    return x_marker

def top_and_bottom_boundary_marker(x):
    return np.logical_or(np.isclose(x[1], 1.0), np.isclose(x[1], 0.0))

def create_dirichlet_boundary_condition(value, msh, V_space, marker):
    facets = locate_entities_boundary(msh, dim=msh.topology.dim-1,
                                       marker=marker)
    dofs = locate_dofs_topological(V=V_space, entity_dim=msh.topology.dim-1, entities=facets)
    bc = dirichletbc(value=ScalarType(value), dofs=dofs, V=V_space)
    return bc

bc1 = create_dirichlet_boundary_condition(1.00, msh, V_Vio, right_boundary_marker)
bc2 = create_dirichlet_boundary_condition(0, msh, V_Vel, left_boundary_marker)
bc3 = create_dirichlet_boundary_condition(0.97, msh, V_Phyd, left_boundary_marker)
bc4 = create_dirichlet_boundary_condition(0.03, msh, V_Pwater, left_boundary_marker)

bcs = [bc1, bc2, bc3, bc4]

# Weak statement of the equations (already including Newmann boundary conditions)
Voe = 1.028
# Bulter-Volmer equation as a source term
j = j0 * (ufl.exp(a*(Voe + Vel - Vio)) * ufl.sqrt(Phyd/Pwater) - ufl.exp(-a*(Voe + Vel - Vio)) * ufl.sqrt(Pwater/Phyd))
# Ionic potential (Poisson-like equation, nonlinear)
F1 = -inner(s_io * f_YSC * grad(Vio), grad(Vio_t)) * dx + inner(j * rho_TPB, Vio_t) * dx
# Electronic potential (Poisson-like equation, nonlinear)
F2 = -inner(s_io * f_Ni * grad(Vel), grad(Vel_t)) * dx - 2 * inner(j * 2*rho_TPB, Vel_t) * dx
# Water vapor pressure (Poisson-like equation, nonlinear)
F3 = -inner(drt * f_pore * grad(Pwater), grad(Pwater_t)) * dx - inner(j * rho_TPB / 2 / F, Pwater_t) * dx
# Hydrogen gas pressure (Poisson-like equation, nonlinear)
F4 = -inner(drt * f_pore * grad(Phyd), grad(Phyd_t)) * dx + 2 * inner(j  * rho_TPB / 2 / F, Phyd_t) * dx
# The full bilinear form
F_all = F1 + F2 + F3 + F4


# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F_all, u, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-7
solver.maximum_iterations = 10000

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

r = solver.solve(u)
current = assemble_scalar(form(j * rho_TPB * dx(msh))) / assemble_scalar(form(1.0 * dx(msh)))
overpotential = assemble_scalar(form((Voe + Vel - Vio + R*T/2/F*ufl.ln(Phyd/Pwater)) * dx(msh))) / assemble_scalar(form(1.0 * dx(msh)))
print(current)
print(overpotential)

# Get the sub-space for c and the corresponding dofs in the mixed space
V0, dofs = V_Vel.collapse()

# Prepare viewer for plotting the solution during the computation
if have_pyvista:
    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.create_vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # u.x.scatter_forward()
    grid.point_data["c"] = u.x.array[dofs].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = "c.png"
    pv.plot(grid, show_edges=True, screenshot=screenshot)
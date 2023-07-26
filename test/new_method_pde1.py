#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc
import time 
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix 
import sys

filename = "data_pde1_direct_formulation.txt"
myfile = open(filename,'w')
myfile.write("mesh_num\t\tL2\t\tH10\n")
myfile.close()

error_list_L2 = []
error_list_H10 = []
error_list_L2_b = []


mesh_num_list = [10,20,40,80]
for mesh_num in mesh_num_list:
    u_e = Expression('2.0/3*x[0]*x[1]*x[1]*x[1]-x[0]*x[1]*x[1]+ 5.0/6',degree = 4)

    f = Expression('-4.0*x[0]*x[1]+2.0*x[0]', degree = 4)
    mesh = UnitSquareMesh(mesh_num,mesh_num)
    V = FunctionSpace(mesh, "CG",1)

    #Define different boundary parts
    boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(99)
    class boundary_D1(SubDomain):
        def inside(self,x,on_boundary):
            return near(x[0],0,DOLFIN_EPS) and on_boundary #and x[1]!=1 and x[1]!=0
    class boundary_D2(SubDomain):
        def inside(self,x,on_boundary):
            return near(x[0],1,DOLFIN_EPS) and on_boundary 
    bc_d1 = boundary_D1()
    bc_d2 = boundary_D2()
    bc_d1.mark(boundary_markers,0)      # 0 marks x = 0 boundary
    bc_d2.mark(boundary_markers,1)      # 1 marks x = 1 boundary 
    bc_u = DirichletBC(V,u_e,boundary_markers,1)
    ds = Measure('ds',domain = mesh, subdomain_data = boundary_markers)
    bcs = [bc_u]

    """
    Get the dof list on the integral boundary 
    """
    bc_u_0 = DirichletBC(V,Constant(0),boundary_markers,0)
    bc0_dof_array = []
    for dof in bc_u_0.get_boundary_values():
        bc0_dof_array.append(dof)
    bc0_dof_array = np.array(bc0_dof_array)
    i_min = min(bc0_dof_array)
    # print(bc0_dof_array)

    # Define variational problem
    phi = TrialFunction(V)
    v = TestFunction(V)

    F = dot(grad(phi),grad(v))*dx - f * v * dx         + (phi-1)*v*ds(0)
    a = lhs(F)
    L= rhs(F)

    """
    Assemble the matrix and the right side vector
    apply bcs
    assemble_system preserves symmetry
    """
    A,b = assemble_system(a,L,bcs)

    """
    Get the row of the matrix corresponding to the bc_u_0 dof
    add them together to get the row 
    """
    start_time = time.time()

    dim = V.dim()
    # print(row_i(2,4,dim))

    """
    Modify the matrix 
    """
    A_np = A.array()
    b_np = b.get_local()

    for dof in bc0_dof_array: 
        if dof != i_min: 
            A_np[i_min,:] += A_np[dof,:]
            b_np[i_min] += b_np[dof]
    for dof in bc0_dof_array: 
        if dof != i_min: 
            A_np[:,i_min] += A_np[:,dof]
    # extract submatrix and sub-vector using index slicing
    index_list = []
    bc0_dof_list = list(bc0_dof_array)
    for i in range(dim):
        if i ==i_min or i not in bc0_dof_list:
            index_list.append(i)
    
    A_np = A_np[index_list,:][:,index_list]

#     print("Memory utilised (bytes): ", sys.getsizeof(A_np))
#     print("Type of the object", type(A_np))
    b_np = b_np[index_list]
    b_petsc = PETSc.Vec().createSeq(len(b_np)) # creating a vector
    b_petsc.setValues(range(len(b_np)), b_np) # assigning values to the vector
#     print('\n Vector b: ')
    # print(b_petsc.getArray()) # printing the vector 
    
    
#     print(b_petsc.getArray() - b_np)
    
    x = PETSc.Vec().createSeq(len(b_np)) # create the solution vector x

    S = csr_matrix(A_np)
#     print("Sparse 'row' matrix: \n",S)
#     print("Memory utilised (bytes): ", sys.getsizeof(S))
#     print("Type of the object", type(S))

    
    petsc_mat = PETSc.Mat().createAIJ(size=S.shape,
                                       csr=(S.indptr, S.indices,
                                          S.data))
#     print(petsc_mat.getValues(range(5), range(5)) - A_np[:5,:5])
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-10 
    opts["ksp_atol"] = 1e-10 
    opts["ksp_max_it"] = 1000 
    # opts["pc_type"] = "none"
    opts["pc_type"] = "hypre"
    opts["pc_hypre_type"] = 'boomeramg'
#     opts["pc_sor_omega"] = 1.5
    opts["ksp_monitor_true_residual"] = None
#     opts["ksp_view"] = None # List progress of solver

    ksp = PETSc.KSP().create() # creating a KSP object named ksp
    ksp.setOperators(petsc_mat)
    ksp.setFromOptions()
    print ('Solving with:', ksp.getType()) # prints the type of solver
    # Solve!
    ksp.solve(b_petsc, x) 

#     print('\\n Solution vector x: ')
    x_np = x.getArray()
    x_np_big = np.zeros(dim)
    x_np_big[index_list] = x_np
    x_np_big[bc0_dof_array] = x_np_big[i_min]
    
    u_e_FEM = interpolate(u_e,V)
    u_e_vec = u_e_FEM.vector().get_local()

    for dof in bc0_dof_array: 
        if dof != i_min:
            u_e_vec = np.delete(u_e_vec,dof,0)
    

    phi = Function(V) # solution 
    phi.vector()[:] = x_np_big

    print("mesh number: ",mesh_num)
    error_L2 = errornorm(u_e,phi,'L2', degree_rise=4)
    error_H1 = errornorm(u_e,phi,'H1', degree_rise=4)
    error_H10 = errornorm(u_e,phi,'H10', degree_rise=4)
    error_L2_b = sqrt(assemble((u_e-phi)*(u_e-phi)*ds(0)))
    error_list_L2.append(error_L2)
    error_list_H10.append(error_H10)
    error_list_L2_b.append(error_L2_b)

    s_error_L2 = str(round(error_L2,8))
    # error_H1 = errornorm(u_e,u.sub(0),'H1', degree_rise=4)
    s_error_H10 = str(round(error_H10,8))
    s_error_L2_b = str(round(error_L2_b,8))

    print("L2 error \t\tH1 error")
    print(s_error_L2 + "\t\t"+s_error_H10)
    myfile = open(filename,'a')
    myfile.write(str(mesh_num)+"\t"+s_error_L2+"\t"+s_error_H10+"\t"+s_error_L2_b+"\n")
    myfile.close()

#Compute convergence order 
def compute_orders(error_list):
    order_list = [] 
    for i in range(len(error_list)-1):
        order = np.log(error_list[i]/error_list[i+1])/np.log(2)
        order_list.append(order)
        # print(order, end = "; ")
    # print()
    return order_list


l2_order_list = compute_orders(error_list_L2)
myfile = open(filename,'a')
myfile.write("l2 order: \n")
for item in l2_order_list: 
    myfile.write(str(item)+"\t")
myfile.write("\n")
myfile.close()
print("L2 convergence order:")
for item in l2_order_list: 
    print(str(item)+"\t")

h1_order_list = compute_orders(error_list_H10)
myfile = open(filename,'a')
myfile.write("h1 order: \n")
for item in h1_order_list: 
    myfile.write(str(item)+"\t")
myfile.write("\n")
myfile.close()

print("H1 convergence order:")
for item in h1_order_list: 
    print(str(item)+"\t")




# In[ ]:





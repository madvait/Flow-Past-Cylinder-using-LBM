import numpy as np
import matplotlib.pyplot as plt

#global variables used for the D2Q9 scheme
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
c = np.array([[0,0],[1, 0],[0, 1],[-1, 0],[0, -1],[1, 1],[-1, 1],[-1, -1],[1, -1]])
inflow_velocity = 0.04

#the function draws a circle filled with dry nodes
def circle(A,x,y,r):
    
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            distance= np.sqrt((i-x)**2+(j-y)**2)
            if distance<=r:
                A[i,j] = 1
    return A

#just a quick function to plot numpy array as an image
def plotarray(A):
    return plt.imshow(np.transpose(A))

#calculate density for the equilibrium distribution
def calc_density(f):
    rho = np.sum(f, axis = -1)
    return rho

''' this function uses the einstein summation convention in np to contract the indices along the required axis and calculate the velocity as a tensor
quick comment on einsum. XYi can be seen as shape of array f=(nx,ny,9) and id as shape of c=(9,2) and contract it to XYd a (nx,ny,2)
 by adding the contracted subscripts over their range '''
def calc_velocity_vectorspace(f, rho):
    velocity = np.einsum("XYi,id->XYd", f,c)/rho[...,np.newaxis]
    return velocity
'''the density matrix is broadcasted to for a (nx,ny,1) matrix for element wise operation
one could also use a nested for loop for the same, but this is much faster'''


'''' A similar approach is taken  for calculating f_i^{eq}....first we calculate the u.ci tensor using einsum'''
def calc_feq_tensor(u,rho):
    # calculate the tensor containing dot products of u with ci for i in range(9) and store them in its space tensor form (size: Nx,Ny,9)
    dot_tensor = np.einsum('id,XYd->XYi', c,u)
    #calculate the u.u value as |u|*|u| and store it in the scalar space form (size:Nx,Ny)
    velocity_magnitude = np.linalg.norm(u, axis = -1, ord = 2)
    #similar to the previous function, the shapes need to be the same: rho(Nx,Ny)->(Nx,Ny,9); w(9)->w(Nx,Ny,9);velmag(Nx,Ny)->(Nx,Ny,9)
    feq = rho[..., np.newaxis] * w[np.newaxis, np.newaxis, :] * (
        1 + 3 * dot_tensor + 4.5 * (dot_tensor**2) - 1.5 * (velocity_magnitude[..., np.newaxis]**2)
    )
    return feq

#this function applies the left outlet boundary condition to a post stream distribution function (use the 3D f array as input)
def left_outlet(f):
        nx = np.shape(f)[0]
        f[nx-1,:,3] = f[nx-2,:,3]
        f[nx-1,:,6] = f[nx-2,:,6]
        f[nx-1,:,7] = f[nx-2,:,7]
        return f
#note:also returns the entire array after oulet correction
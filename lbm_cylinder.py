import numpy as np
import matplotlib.pyplot as plt
from lbm import *
from tqdm import tqdm

#constants of the problem

inflow_velocity = 0.04
# lattice constants
nx,ny = 300,50
tau = 0.525           
iterationtime = 10000

#setting up the scene:
canvas = np.zeros([nx,ny])
float_map = circle(canvas,nx/4,ny/2,ny/9)
boolean_mask = float_map.astype(bool)

#lattice weights and velocities
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
c = np.array([[0,0],[1, 0],[0, 1],[-1, 0],[0, -1],[1, 1],[-1, 1],[-1, -1],[1, -1]])
reverse_indices = np.array([0,3,4,1,2,7,8,5,6])

# initializing the distribution functions and macroscopic quantities
rho = np.ones([nx,ny]) #rho is a 2D array containing  the value of densities at (x,y)
u = np.zeros([nx,ny,2]) #u is a 3D array containing  the ux,uy at (x,y): u[i,j,0] = ux and u[i,j,1] = uy 
f = np.zeros([nx,ny,9]) #f is the distribution function array containing the 9 distrubion values for each (x,y)

# before the time iteration, some conditions would be applied throughout the domain:
u[:,:,0]= inflow_velocity

# we set the initial distribution function to be the equilibrium distribution at the initialized macroscopic conditions
f = calc_feq_tensor(u,rho)

#begin loop
def update(f):
    #1.) Applying the Outlflow boundary condition to f
    f_prev = left_outlet(f)

    #2.) compute Macroscopic density and velocities rho and u
    rho_prev = calc_density(f)
    u_prev = calc_velocity_vectorspace(f_prev,rho_prev)

    #3.) apply the inlet velocity boundary condition:
    u_prev[0,1:-1,0] = inflow_velocity # enforce inlet wall velocity
    rho_prev[0,:] = (f_prev[0,:,0]+f_prev[0,:,2]+f_prev[0,:,4]+2*(f_prev[0,:,3]+f_prev[0,:,6]+f_prev[0,:,7]))/(1-inflow_velocity) #update wall density
    
    #4.) Compute new equilibrium distribution
    f_eq = calc_feq_tensor(u_prev,rho_prev)

    #part of enforcing boundary condition
    f_prev[0,:,:] = calc_feq_tensor(u_prev,rho_prev)[0,:,:] #assign inlet distribution funcs as equilibrium solutions for the case (provides stable solution)

    #5.) Perform collision using the BGK Operator
    f_star = (f_prev - (1/tau)*(f_prev-f_eq) )

    #6.) Perform the bounceback operation
    for i in range(9):
        f_star[boolean_mask, i] = f_prev[boolean_mask,reverse_indices[i]]
    
    #7.) Stream using np.roll
    f_stream = np.copy(f_star)
    for i in range(9):
        f_stream[:,:,i] = np.roll(f_star[:,:,i], c[i], axis = (0,1))
    
    #going back to the previous loop
    return f_stream

x = np.arange(nx)
y = np.arange(ny)
x,y = np.meshgrid(x, y)

plt.figure(figsize=(15,4), dpi = 100)

for time in tqdm(range(iterationtime)):
    fnext = update(f)
    f = np.copy(fnext)
    rho = calc_density(fnext)
    u = calc_velocity_vectorspace(fnext,rho)
    if time%100==0 and time>0:
        velocity_magnitude = np.linalg.norm(u, axis=-1)
        plt.subplot(111)
        plt.contourf(x,y,np.transpose(velocity_magnitude), cmap='magma')
        plt.colorbar(label='Velocity Magnitude')
        plt.title('Velocity Magnitude Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
    
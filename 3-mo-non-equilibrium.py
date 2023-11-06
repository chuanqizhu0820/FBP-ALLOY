
    # vii = (phi - phi0 + 1.0e-10) / dt * delta / np.pi / np.sqrt(phi * (1.0 - phi)+ 1.0e-10)
    # kapii = (kap+vii)/(1.0+vii)
     
import numpy as np
import matplotlib.pyplot as plt

# cooling conditions
Tm = 931.0
temp0 = 910.0

# materials properties
Gamma = 2.4e-7
Mu = 0.1
ml = -600.0
kap = 0.14
cl = 0.013
cle = (temp0 - Tm) / ml
cse = (temp0 - Tm) / ml * kap
Dl = 2.4e-9
Ds = 2.4e-12

# numerical parameters
Nt = 10000
Nx = 128   
dx = 4.0e-9 
delta = dx*8.0
dt = 0.12*dx*dx/Mu/Gamma

# array allocation
dphi = np.zeros(Nx)
phi = np.zeros(Nx)
phi0 = np.zeros(Nx)
conl = np.zeros(Nx)
cons = np.zeros(Nx)
con = np.zeros(Nx)
dcon = np.zeros(Nx)
coni = np.zeros(Nx)

# Initial conditions
phi[:Nx//8*1] = 1.0
conl = np.ones(Nx)*cl
cons = np.ones(Nx)*cse

con = phi*cons + (1.0-phi)*conl

def laplacian(arr):
    return (np.append(arr[1:], arr[-1]) + np.insert(arr[:-1], 0, arr[0]) - 2*arr)/dx**2

def gradient(arr):
    return (np.insert(arr[:-1], 0, arr[0]) - np.append(arr[1:], arr[-1]))/dx/2.0

for it in range(Nt):

    phi0.fill(0.0)
    phi0 += phi
  
    phi += dt*Mu*(Gamma*(laplacian(phi) - np.pi*np.pi/delta/delta *
                  (1.0/2.0-phi))-(cle-coni)*ml*np.sqrt(phi*(1.0-phi))*np.pi/delta)
    
    phi[phi > 1.0] = 1.0
    phi[phi < 0.0] = 0.0
    
    # solute capturing
    for ix in range(Nx):
        if(phi[ix]>0.00 and phi[ix]<1.00):
            vii = (phi[ix] - phi0[ix]) / dt * delta / np.pi / np.sqrt(phi[ix] * (1.0 - phi[ix]))
            cap = (kap+vii)/(1.0+vii)
            cons[ix] = (cons[ix]*phi0[ix] + conl[ix]*cap*(phi[ix]-phi0[ix]))/phi[ix]

    dcon.fill(0.0)
    dcon = con-cons*phi -conl*(1-phi)

    coni.fill(0.0)
    for ix in range(Nx):
        # zero-flux condition at solid-interface boundary
        if(phi[ix-1]==1.0):
            conl[ix-1] = conl[ix]

        # Search the representative liquid concentration near the solid-interface boundary
        di = 0
        while(phi[ix+di]<1.0 and phi[ix+di]>0.0):
            coni[ix] = conl[ix+di]
            di -= 1

        # Cast rejection ahead of the middle obstacle
        di = 0
        while(phi[ix+di]<1.0 and phi[ix+di]>0.5):
            di += 1
            if(phi[ix+di]<=0.5):
                dcon[ix+di] += dcon[ix]
                dcon[ix] = 0.0
                break
     
    # update liquid concentration after cast operation
    for ix in range(Nx):
        if(phi[ix]<1.0):
            conl[ix] += dcon[ix]/(1.0-phi[ix])

    con = cons*phi + conl*(1-phi)
    
    con += dt*Dl*((1.0-phi)*laplacian(conl) - gradient(phi)*gradient(conl))

    # update liquid concentration after diffusion
    for ix in range(Nx):
        if(phi[ix]<1.0):
            conl[ix] = (con[ix]-cons[ix]*phi[ix])/(1.0-phi[ix])

    if (it % 2000 == 0):
        print(np.sum(con)/Nx)
    
plt.plot(con)
plt.plot(conl)
# plt.plot(coni)
plt.xlabel('Position')
plt.ylabel('Con')
plt.show()
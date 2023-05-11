import numba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter



def boundary(f): 
    f[0], f[-1] = 0,0
    return f

def get_fR(x,t): return boundary(C*np.cos(k*x-w*t)*np.exp(-(x-x0)**2/(2*s**2)))

def get_fI(x,t): return boundary(C*np.sin(k*x-w*t)*np.exp(-(x-x0)**2/(2*s**2)))

@numba.jit(nopython=True, nogil=True, cache=True)
def V(x,t): return 0

def module(fR,fI): return fR**2 + fI**2

def get_norm(f): return np.sum(np.abs(f)**2)*dx
    

@numba.jit(nopython=True, nogil=True, cache=True)
def evolve(fR,fI):
    """Evolves the real and imaginary part of the Wavefunction. Uses numba to accelerate the process.

    Parameters
    ----------
    `fR` : Real part of the wavefunction.
    `fI` : Imaginary part of the wavefunction.

    Returns
    -------
    `fR` : Updated real part of the wavefunction.
    `fI` : Updated imaginary part of the wavefunction.

    """
    for i in range(1,Nx-1):
        fR[i] = fR[i] - dt*h/(2*m*dx**2)*(fI[i+1]-2*fI[i]+fI[i-1]) - dt*V(0,0)*fI[i]

    for i in range(1,Nx-1):
        fI[i] = fI[i] + dt*h/(2*m*dx**2)*(fR[i+1]-2*fR[i]+fR[i-1]) - dt*V(0,0)*fR[i]

    # fR[0], fR[-1] = 0,0
    # fI[0], fI[-1] = 0,0

    return fR, fI

def Animation(frame):
    """Function that creates a frame for the GIF."""
    ax1.clear();ax2.clear()

    # Real Part
    fR0, = ax1.plot(x,fR_frames[0],c="blue",alpha=0.3,label="$\psi_R(t_0))$")
    fRt, = ax1.plot(x,fR_frames[frame],c="red",label="$\psi_R(t))$")

    # Imaginary Part
    fI0, = ax2.plot(x,fI_frames[0],c="blue",alpha=0.3,label="$\psi_I(t_0))$")
    fIt, = ax2.plot(x,fI_frames[frame],c="red",label="$\psi_I(t))$")


    # Walls
    wall1 = ax1.axvline(x[0],ymin=0,c="k",alpha=0.5)
    wall2 = ax1.axvline(x[-1],ymin=0,c="k",alpha=0.5)

    wall1 = ax2.axvline(x[0],ymin=0,c="k",alpha=0.5)
    wall2 = ax2.axvline(x[-1],ymin=0,c="k",alpha=0.5)

    ax1.set_xlabel("x");ax1.set_ylabel("$\psi$")
    ax2.set_xlabel("x");ax2.set_ylabel("$\psi$")

    ax1.set_ylim(-1,1)
    ax2.set_ylim(-1,1)

    # ax1.legend(loc="upper right")
    # fig.tight_layout()

    return fRt,fIt

def write(line,file):
	"""Prints and writes line in the terminal and output file."""
	print(line,flush=True)
	with open(file,"a",encoding="utf-8") as outFile: outFile.write(line+"\n")



# Constants
pi = np.pi
hbar = 1.054571817e-34
uma = 1.6605402e-27
me = 9.1093837015e-31
a0 = 5.29177210903e-11
e = 1.602176634e-19
Eh = hbar**2/(me*a0**2)

# Spatial Inputs
L = 50
dx = 0.01
x0 = L/2
x = np.arange(0,L+dx,dx)
Nx = len(x)

# Time Inputs
dt = 0.0001                       # Time step (au)
Nt = 500000                        # Time points
duration = Nt*dt                 # Total time
t = np.arange(0,duration+dt,dt)

# Animation Inputs
animation_frames = 200
animation_name = "dq.gif"

# Wavefunction inputs:
C = 1
k = 3
w = 2
s = 2
h = 1
m = 1


# Initialization
fR = boundary(get_fR(x,0))
fI = boundary(get_fI(x,0))

# Integration Loop
print("\n Starting integration...")

fR_frames, fI_frames = [],[]
for i,t_i in enumerate(t):
    fR,fI = evolve(fR,fI)
    # fI = evolveI(fI,fR)

    # print(fR)

    if i%int(Nt/animation_frames) == 0 : 
        fR_frames.append(fR.copy())
        fI_frames.append(fI.copy())
        
fR_frames, fI_frames = np.array(fR_frames), np.array(fI_frames)


# Creating GIF animation of the evolution of concentrations
print("\nStarting animation...")
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))

animation = FuncAnimation(fig,Animation,frames=animation_frames,interval=20,blit=False,repeat=True, )
animation.save(animation_name,dpi=120,writer=PillowWriter(fps=25))
# fig.tight_layout()
plt.show()
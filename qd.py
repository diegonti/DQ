"""
Python program that propagates a wave packet under a specified potential.
Considers a simple Euler integration method and a more robust Runge-Kutta 4 method.
Diego Ontiveros
"""

from time import time as cpu_time
import warnings

import numba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
warnings.filterwarnings("ignore", category=numba.errors.NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=numba.errors.NumbaWarning)


######################## Wavefunctions and Potential expressions ###########################

def boundary(f): 
    """Sets extremes of array to 0."""
    f[0], f[-1] = 0,0
    return f

def get_fR(x,t): return boundary(C*np.cos(k*x-w*t)*np.exp(-(x-x0)**2/(2*s**2)))

def get_fI(x,t): return boundary(C*np.sin(k*x-w*t)*np.exp(-(x-x0)**2/(2*s**2)))

def get_V(x,type:str=None,t=None,**params): 
    """Function to choose and calculate the potential array.

    Parameters
    ----------
    `x` : Position array.\n
    `t` : Time point. Optional, by default None.\n
    `type` : The type of potential to use. Optional, by default 0. Available are:\n
        `barrier` : Potential barrier of value V0 and centered at xe.\n
        `harmonic` : Harmonic potential centered at xe with force constant k. V = 0.5*k*(x-xe)**2\n
        `morse` : Morse potential centerded at xe with dissociation energy D and force constant k. V = D*(1-np.exp(-alpha*(x-xe)))**2\n
    `**params` : Extra keyword parameters needed for the potential function. For each case, minding the order:
        `barrier` : xe, V0.\n
        `harmonic` : xe, k.\n
        `morse` : D, xe, k.\n
        
    Returns
    -------
    `V` : Potential array.
    """
    if type is None: type = "zero"

    type = type.lower()
    if type == "barrier":
        xe = params.get("xe",L/2)
        V0 = params.get("V0",1)
        V = barrier(x,xe,V0)
    elif type == "harmonic":
        xe = params.get("xe",L/2)
        k = params.get("k",3)
        V = harmonic(x,xe,k)
    elif type == "morse":
        D = params.get("D",4)
        xe = params.get("xe",L/2)
        k = params.get("k",3)
        V = morse(x,D,xe,k)
    elif type.startswith("zero"):
        V = np.zeros(len(x))
    else: raise ValueError("Potential type not detected. Pleas select one from the following: zero, barrier, harmonic, morse.")

    return V

def morse(x,D,xe,k):
    """Morse potential of the form V(x) = D*(1-exp(-a*(x-xe)))**2"""
    alpha = np.sqrt(k/(2*D))
    return D*(1-np.exp(-alpha*(x-xe)))**2

def harmonic(x,xe,k):
    """Harmonic potential of the form V(x) = 0.5*k*(x-xe)**2"""
    return 0.5*k*(x-xe)**2

def barrier(x,xe,V0):
    """Barrier potential centered at xe and value V0"""
    V = np.zeros(len(x))
    sites = np.logical_and( x>0.9*xe , x<1.1*xe)
    V[sites] = V0
    return V



@numba.jit(nopython=True, nogil=True, cache=True)
def get_norm(fR,fI): return np.sum(module(fR,fI))*dx
    
@numba.jit(nopython=True, nogil=True, cache=True)
def module(fR,fI): return fR**2 + fI**2


################################ Animation Functions ################################

def Animation_split(frame):
    """Function that creates a mpl frame for the GIF visualization. Separates real and momentum space."""
    ax1.clear();ax2.clear()

    # Real Part
    fRt, = ax1.plot(x,fR_frames[frame],c="red",label="$\psi_R(t)$", alpha=0.5)
    fIt, = ax1.plot(x,fI_frames[frame],c="royalblue",label="$\psi_I(t)$", alpha=0.5)
    f2t, = ax1.plot(x,f2_frames[frame],c="k",label="$|\psi|^2(t)$")

    # Momentum Part
    fRt_K, = ax2.plot(p,fR_K_frames[frame],c="red",label="$\psi_R(t)$", alpha=0.5)
    fIt_K, = ax2.plot(p,fI_K_frames[frame],c="royalblue",label="$\psi_I(t)$", alpha=0.5)
    f2t_K, = ax2.plot(p,f2_K_frames[frame],c="k",label="$|\psi|^2(t)$")



    # Walls
    V1 = ax1.plot(x,V, c="grey", alpha=0.5)
    # V2 = ax2.plot(x,V, c="grey", alpha=0.5)

    wall1 = ax1.axvline(x[0],ymin=0,c="k",alpha=0.5)
    wall2 = ax1.axvline(x[-1],ymin=0,c="k",alpha=0.5)

    # wall1 = ax2.axvline(p[0],ymin=0,c="k",alpha=0.5)
    # wall2 = ax2.axvline(p[-1],ymin=0,c="k",alpha=0.5)

    # Norm Text
    ax1.text(0.5, 0.7, f"{norm_frames[frame]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.1, 0.08, f"{norm_split_frames[frame][0]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.9, 0.08, f"{norm_split_frames[frame][1]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    # ax2.text(0.5, 0.7, f"{norm_frames[frame]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    # ax2.text(0.1, 0.08, f"{norm_split_frames[frame][0]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    # ax2.text(0.9, 0.08, f"{norm_split_frames[frame][1]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)


    ax1.set_xlabel("x");ax1.set_ylabel("$\psi$")
    ax2.set_xlabel("k");ax2.set_ylabel("$\psi$")

    ax1.set_ylim(-1,1)
    ax2.set_ylim(-2,2)
    ax2.set_xlim(-5,5)

    ax1.legend(loc=(0.75,0.78))
    ax2.legend(loc=(0.75,0.78))

    fig.tight_layout()

    return fRt,fIt


def Animation(frame):
    """Function that creates a mpl frame for the GIF visualization. All in one plot"""
    ax1.clear()

    # Wavefunctions
    fRt, = ax1.plot(x,fR_frames[frame],c="red",label="$\psi_R(t)$", alpha=0.5)
    fIt, = ax1.plot(x,fI_frames[frame],c="royalblue",label="$\psi_I(t)$", alpha=0.5)
    f2t, = ax1.plot(x,f2_frames[frame],c="k",label="$|\psi|^2(t)$")


    # Walls
    V1 = ax1.plot(x,V, c="grey", alpha=0.5)

    wall1 = ax1.axvline(x[0],ymin=0,c="k",alpha=0.5)
    wall2 = ax1.axvline(x[-1],ymin=0,c="k",alpha=0.5)

    # Norm Text
    ax1.text(0.5, 0.7, f"{norm_frames[frame]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.1, 0.08, f"{norm_split_frames[frame][0]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.9, 0.08, f"{norm_split_frames[frame][1]:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)


    ax1.set_xlabel("x");ax1.set_ylabel("$\psi$")

    ax1.set_ylim(-1,1)

    ax1.legend(loc=(0.75,0.78))
    fig.tight_layout()

    return fRt,fIt


########################### Evolution Functions #################################33
@numba.jit(nopython=True, nogil=True, cache=True)
def get_k(fR,fI,V):
    """Gets the k arrays for the RK4 method."""
    fR_c = fR.copy()
    for i in range(1,Nx-1):
        fR[i] = -Ck*(fI[i+1]-2*fI[i]+fI[i-1]) + V[i]*fI[i]/h

    for i in range(1,Nx-1):
        fI[i] = Ck*(fR_c[i+1]-2*fR_c[i]+fR_c[i-1]) - V[i]*fR_c[i]/h

    return fR, fI
    

@numba.jit(nopython=True, nogil=True, cache=True)
def evolve(fR,fI,V):
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
        fR[i] = fR[i] - dt*Ck*(fI[i+1]-2*fI[i]+fI[i-1]) + dt*V[i]*fI[i]/h

    for i in range(1,Nx-1):
        fI[i] = fI[i] + dt*Ck*(fR[i+1]-2*fR[i]+fR[i-1]) - dt*V[i]*fR[i]/h

    return fR, fI


# @numba.jit(nogil=True, cache=True)
def evolveRK(fR,fI,V):
    """Evolves the real and imaginary part of the Wavefunctionm using the RK4 method.

    Parameters
    ----------
    `fR` : Real part of the wavefunction.
    `fI` : Imaginary part of the wavefunction.
    `V`  : Potential.

    Returns
    -------
    `fR_f` : Updated real part of the wavefunction.
    `fI_f` : Updated imaginary part of the wavefunction.

    """

    fR_0 = fR.copy()
    fI_0 = fI.copy()

    sKR = 0.
    sKI = 0.
    coefs = [[0.5,1],[0.5,2],[1,2],[1,1]]

    for i in range(4):
        kiR,kiI = get_k(fR.copy(),fI.copy(),V)
        fRi = fR_0 + dt*kiR * coefs[i][0]
        fIi = fI_0 + dt*kiI * coefs[i][0]
       
        if i != 3:
            fR = fRi.copy()
            fI = fIi.copy()

        sKI = sKI + coefs[i][1]*kiI
        sKR = sKR + coefs[i][1]*kiR


    fR_f = fR_0 + dt/6*sKR
    fI_f = fI_0 + dt/6*sKI

    return fR_f, fI_f


def write(line,file):
	"""Prints and writes line in the terminal and output file."""
	print(line,flush=True)
	with open(file,"a",encoding="utf-8") as outFile: outFile.write(line+"\n")


############################### MAIN PROGRAM ###############################

initial_time = cpu_time()

# Constants
pi = np.pi
hbar = 1.054571817e-34
uma = 1.6605402e-27
me = 9.1093837015e-31
a0 = 5.29177210903e-11
e = 1.602176634e-19
Eh = hbar**2/(me*a0**2)

# Spatial Inputs
L = 100
dx = 0.1
x0 = 40
center = L/2
x = np.arange(0,L+dx,dx)
Nx = len(x)

# Time Inputs
dt = 1e-3                      # Time step (au)
Nt = 500000                    # Time points
duration = Nt*dt               # Total time
t = np.arange(0,duration+dt,dt)

# Animation Inputs
animation_frames = 50
animation_name = "dq.gif"
Animator = Animation_split

# Wavefunction inputs:
C = 1
k = 3
w = 1
s = 2
h = 1
m = 5

C = 1/(1*pi*s**2)**(1/4)
Ck = h/(2*m*dx**2)
Ek = (h*k)**2/(2*m)

# Initialization
fR = boundary(get_fR(x,0))
fI = boundary(get_fI(x,0))
V = get_V(x,type="morse", xe=center, k=0.001*k)


psi_K = np.fft.fft(fR+1j*fI)
fR_K,fI_K = psi_K.real,psi_K.imag
conts = get_norm(fR_K,fI_K)
conts = np.sqrt(conts)
p = np.fft.fftfreq(len(psi_K))*1/dx*2*pi


# Integration Loop
print("\nStarting integration...")
print("Completed:", end=" ")

fR_frames, fI_frames, f2_frames, norm_frames, norm_split_frames = [],[],[],[],[]
fR_K_frames, fI_K_frames, f2_K_frames, norm_K_frames, norm_K_split_frames = [],[],[],[],[]

for i,t_i in enumerate(t):
    fR,fI = evolveRK(fR,fI,V)

    
    # Save frames for animation
    if i%int(Nt/animation_frames) == 0 :
        psi_K = np.fft.fft(fR+1j*fI)
        fR_K,fI_K = psi_K.real/conts,psi_K.imag/conts

        mod = module(fR,fI)
        fR_frames.append(fR.copy())
        fI_frames.append(fI.copy())
        f2_frames.append(mod.copy())

        mod_K = module(fR_K,fI_K)
        fR_K_frames.append(fR_K.copy())
        fI_K_frames.append(fI_K.copy())
        f2_K_frames.append(mod_K.copy())

        norm1 = get_norm(fR[:int(Nx/2)],fI[:int(Nx/2)])
        norm2 = get_norm(fR[int(Nx/2):],fI[int(Nx/2):])     
        norm_split_frames.append([norm1,norm2])   
                                 
        norm = get_norm(fR,fI)
        norm_frames.append(norm)
    
    # % Completed
    if i%(Nt/10) == 0:
        print(f"{int(100*i/Nt)}% ",sep=" ",end="",flush=True)
        
fR_frames, fI_frames = np.array(fR_frames), np.array(fI_frames)
fR_K_frames, fI_K_frames = np.array(fR_K_frames), np.array(fI_K_frames)

finish_time = cpu_time()
print(f"\nProcess finished in {finish_time-initial_time:.2f}s")


# Creating GIF animation of the evolution of the wavefunction
print("\nStarting animation...")
initial_time2  = cpu_time()

if Animator == Animation_split: fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
else: fig, ax1 = plt.subplots()
fig.tight_layout()

animation = FuncAnimation(fig,Animator,frames=animation_frames,interval=20,blit=False,repeat=True, )
animation.save(animation_name,dpi=120,writer=PillowWriter(fps=25))

finish_time2  = cpu_time()
print(f"Process finished in {finish_time2-initial_time2:.2f}s")

# Test potencial Morse y harmonico con diferentes energias y distancias.


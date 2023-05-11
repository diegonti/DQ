import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def fR(x,t): return C*np.cos(k*x-w*t)*np.exp(-(x-x0)**2/(2*s**2))

def fI(x,t): return C*np.sin(k*x-w*t)*np.exp(-(x-x0)**2/(2*s**2))

def V(x,t): return 0

def module(fR,fI): return fR**2 + fI**2

def evolveR(x,t): 
    return fR(x,t) - dt*h/(2*m) * (fI(x+dx,t)-2*fI(x,t)+fI(x-dx,t))/dx**2 - dt*V(x,t)*fI(x,t)/h

def evolveI(x,t): 
    f = fI(x,t) + dt*h/(2*m) * (fR(x+dx,t)-2*fR(x,t)+fR(x-dx,t))/dx**2 - dt*V(x,t)*fR(x,t)/h




C = 1
k = 3
w = 1
L = 10
dx = 0.0001
dt = 0.1
x0 = L/2
s = 2
h = 1
m = 1
animation_frames = 1000
animation_name = "dq.gif"

x = np.arange(0,L+dx,dx)
fR_frames, fI_frames = [],[]

f_R = fR(x,0)
f_I = fI(x,0)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x,f_R, c="r", label="$\psi_R$", alpha=1)
ax.plot(x,f_I, c="b", label="$\psi_I$", alpha=1)
ax.plot(x,module(f_R,f_I), c="k", label="$|\psi|^2$", alpha=0.5)
ax.set_xlim(x[0],x[-1])
ax.set_xlabel("x");ax.set_ylabel("$\psi$")
ax.legend()

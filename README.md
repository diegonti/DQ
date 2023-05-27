# Quantum Dynamics - Wavefunction Evolution

QD Project for the Quantum Dynamics course of the Atomistic and Multiscale Computational Modelling Master Degree.

By combining principles from quantum mechanics, numerical methods, and programming, a well-developed wavefunction propagation program enables simulations and exploration of simple quantum systems. Here, a Python program that propagates a wave packet under a specified potential has been developed, considering a simple Euler integration method and a more robust Runge-Kutta 4 method. After the simulation, it provides an animation (GIF) of its evolution.

The program uses `numba` to accelerate the process.

<br>

# Mathematical Background

The considered wave packet was chosen as a product of a gaussian function and an imaginary phase, with the following expression:


$$ \Psi(x,t) = Ce^{-(x-x_0)^2/(2σ^2)} e^{i(kx-ωt)} $$

where $C$ is the normalization constant, $k$ the wave number, $\omega$ the angular frequency, and $x_0$ and $\sigma$ the gaussian centre and width, respectively. To ease the integration process, the above wavefunction as been separated into a real and imaginary part, with expressions as follows:

$$
\Psi(x,t)=\Psi_R+i\Psi_I \begin{cases}
    Ψ_R = Ccos(kx-ωt) e^{-(x-x_0)^2/(2σ^2)} \\
    Ψ_I=Csin(kx-ωt) e^{-(x-x_0)^2/(2σ^2)}
\end{cases}
$$

The simulations have been carried out in a box of $L=100$ a.u. Forcing the wave function to be zero at the extremes of the box, i.e. $Ψ(0,t)=Ψ(L,t)=0$. To simplify the simulation, atomic units have been used for all the variables.

Four different potentials are considered. First, the case of a free particle with no potential. A potential barrier of $V_0$ value centered at $x_e$, was also considered:

$$
V(x) = \begin{cases}
   V_0 & \text{if }\ 0.9x_e < x < 1.1x_e \\
    0  & \text{otherwise}
\end{cases}
$$


A harmonic potential, was also used, centered at $x_e$ and adjusted with the factor $\alpha$:

$$ V(x) = 0.5\alpha k(x-x_e)^2 $$

Finally, a Morse type potential has also been considered, see the below expression, centered at x_e and with D being the dissociation energy.

$$ V(x)=D(1-e^{-α(x-x_e)})^2  \ \ \   \alpha=\sqrt{k/2D} $$

Most of the parameters needed for the potentials can be set by the user.


For any doubts or questions, contact [Diego Ontiveros](https://github.com/diegonti) ([Mail](mailto:diegonti.doc@gmail.com)).

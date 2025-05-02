import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from example1_analytic import J_analytic, n_analytic

# initial condition!

def n_0(x):
    a = -25 * (x - 0.5)**2
    return np.exp(a)**2

def S_0(x):
    a = np.exp(5 * (x - 0.5))
    return -0.2 * np.log(a + 1/a)

def u0(x, eps):
    return np.sqrt(n_0(x)) * np.exp(1j * S_0(x) / eps)

# epsilon and h values for each plot.
eps_vals = [0.0064, 0.0001,  0.0000125]
h_vals = [1/64, 1/4096, 1/32768]

def V(x, t):
    return 10

plt.style.use("seaborn-v0_8-dark-palette")
plt.style.use("./stylesheet.mplstyle")
fig, axs = plt.subplots(len(eps_vals), 2, layout="tight")

T = 0.54
for i, (eps, h) in enumerate(zip(eps_vals, h_vals)):
    xarr = np.arange(0, 1, h)
    tarr = np.arange(0, 1.1*T, T)
    # k = T so it's only one step
    uarr = SP2(u0(xarr, eps), 0, 1, T, h, T, eps, V) 

    pd = pos_density(uarr)
    cd = curr_density(uarr, eps, h)

    #numerical plots
    axs[i, 0].scatter(xarr, pd, s=10, marker="+", color="C0")
    axs[i, 1].scatter(xarr, cd, s=10, marker="+", color="C0")

    axs[i, 0].set_ylabel(f"({romans[i]})", rotation=0, fontweight="heavy", labelpad=15)

x_analytic = np.linspace(0, 1, 500)
n_ana_arr = n_analytic(x_analytic, T)
J_ana_arr = J_analytic(x_analytic, T)


for i in range(len(eps_vals)):
    axs[i,0].plot(x_analytic, n_ana_arr, color="C1")
    axs[i,1].plot(x_analytic, J_ana_arr, color="C1")


axs[0, 0].set_title("position density")
axs[0, 1].set_title("current density")

plt.savefig('../figures/example1.pdf')
plt.show()

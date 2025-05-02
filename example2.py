import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

# initial condition!

def n_0(x):
    a = -25 * (x - 0.5)**2
    return np.exp(a)**2

def S_0(x):
    return x + 1
    # return 0.2*(x**2 - x)

def u0(x, eps):
    return np.sqrt(n_0(x)) * np.exp(1j * S_0(x) / eps)


# epsilon and h values for each plot.
eps_vals = [0.04, 0.0025, 0.00015625]
h_vals = [1/16, 1/256, 1/4096]


def V(x, t):
    return x**2 / 2

def n_analytic(x, t, eps):
    # u_I = dS/dx = 1
    arg = (x - np.sin(t)) / np.cos(t)
    return n_0(arg) / np.abs(np.cos(t))

def J_analytic(x, t, eps):
    factor = (1-x*np.sin(t)) / np.cos(t) 
    return n_analytic(x, t, eps) * factor



plt.style.use("seaborn-v0_8-dark-palette")
plt.style.use("./stylesheet.mplstyle")
fig, axs = plt.subplots(len(eps_vals), 2, layout="tight")


T = 3.6

k = 0.02

a, b = -2, 2

# this could be cleaner, but it works
plot_min, plot_max = -1.4, -0.4

for i, (eps, h) in enumerate(zip(eps_vals, h_vals)):
    xarr = np.arange(a, b, h)
    tarr = np.arange(0, (1+k)*T, k)
    uarr = SP2(u0(xarr, eps), a, b, T, h, k, eps, V) 

    pd = pos_density(uarr)
    cd = curr_density(uarr, eps, h)

    x_plot = [x for x in xarr if x > plot_min and x < plot_max]
    pd_plot = [p for x, p in zip(xarr, pd) if x > plot_min and x < plot_max]
    cd_plot = [c for x, c in zip(xarr, cd) if x > plot_min and x < plot_max]


    # numerical
    axs[i, 0].scatter(x_plot, pd_plot, s=10, marker="+", color="C0")

    # numerical
    axs[i, 1].scatter(x_plot, cd_plot, s=10, marker="+", color="C0")

    axs[i, 0].set_ylabel(f"({romans[i]})", rotation=0, fontweight="heavy", labelpad=15)

# analytical plots: always the highest resolution one
for i in range(len(eps_vals)):
    axs[i, 0].plot(x_plot, n_analytic(np.array(x_plot), T, eps), linestyle="solid", color="C1")
    axs[i, 1].plot(x_plot, J_analytic(np.array(x_plot), T, eps), linestyle="solid", color="C1")


axs[0, 0].set_title("position density")
axs[0, 1].set_title("current density")

plt.savefig('../figures/example2.pdf')
plt.show()


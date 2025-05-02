# this file is separate because the experiment takes a while.

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from example3_experiment import *


def n_analytic(x, y, t, eps):
    argx = (x - np.sin(t)) / np.cos(t)
    argy = (y - 0.5*np.sin(t)) / np.cos(t)

    return n_0(argx, argy) / np.abs(np.cos(t))**2

plt.style.use("seaborn-v0_8-dark-palette")
plt.style.use("./stylesheet.mplstyle")
fig, axs = plt.subplots(len(eps_vals), 2, layout="tight")

x_crosssection = 0
y_crosssection = -0.25

output_fn = "data/experiment3_out.pkl"

with open(output_fn, 'rb') as f:
    data = pickle.load(f)

xmin, xmax = -0.3, 0.3
ymin, ymax = -0.8, 0.2

for i, (eps, h) in enumerate(zip(eps_vals, h_vals)):
    u = data[i]
    xarr = np.arange(-2, 2, h)
    yarr = np.arange(-2, 2, h)
    xgrid, ygrid = np.meshgrid(xarr, yarr)
    pd = pos_density(u)
    
    # plot for y=-0.25
    xpd_plot = pd[np.where(ygrid == -0.25)]
    axs[i, 0].scatter(xarr, xpd_plot, s=10, marker="+", color="C0")
    axs[i, 0].set_xlim(-0.3, 0.3)
    axs[i, 0].set_ylim(0, 1.4)


    # plot for x=0
    ypd_plot = pd[np.where(xgrid == 0)]
    axs[i, 1].scatter(yarr, ypd_plot, s=10, marker="+", color="C0")
    axs[i, 1].set_xlim(-0.8, 0.2)
    axs[i, 1].set_ylim(0, 1.4)

    axs[i, 0].set_ylabel(f"({romans[i]})", rotation=0, fontweight="heavy", labelpad=15)

# weak limit plot
xarr = np.arange(-2, 2, 0.01)
for i in range(len(eps_vals)):
    axs[i, 0].plot(xarr, n_analytic(xarr, -0.25, T, eps), color="C1", linestyle="solid")
    axs[i, 1].plot(xarr, n_analytic(0, xarr, T, eps), color="C1")

axs[-1,0].set_xlabel("x")
axs[-1,1].set_xlabel("y")

axs[0, 0].set_title("y = -0.25")
axs[0, 1].set_title("x = 0")

plt.savefig('../figures/example3.pdf')
plt.show()

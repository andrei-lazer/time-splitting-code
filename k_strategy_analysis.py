import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from k_strategy_experiment import *

def monomial_fit(x, y):
    """
    Performs a fit of the form y = Ax^r.

    Params:
        x: numpy array of x values.
        y: numpy array of y values.

    Returns:
        r, A: np.float64 objects, such that y = Ax^r is the
        least-squares monomial fit.
    """

    logx = np.log(x)
    logy = np.log(y)
    r, logA = np.polyfit(logx, logy, 1)

    return r, np.exp(logA)


# outer = epsilon, inner = k
raw_data = np.load("data/k_experiment_out.npy")

print(raw_data)
sp1_data = raw_data[0,0]
sp2_data = raw_data[1,0]

# interpolation
klog = np.log(k_vals)
karr = np.arange(min(k_vals), max(k_vals), 100)

# plots!

plt.style.use("seaborn-v0_8-dark-palette")
plt.style.use("./stylesheet.mplstyle")

def custom_format(x, pos):
    return f'{x:g}'.replace('\u2212', '-')  # Replace Unicode minus with hyphen

fig, [sp1_ax, sp2_ax] = plt.subplots(1, 2, layout="tight")
fig.set_figheight(4)
fig.suptitle(r"$\ell^2$ error")

# SP1

sp1_ax.scatter(k_vals, sp1_data, marker="+", color="C0")
sp1_ax.set_title("SP1")
sp1_ax.set_xscale("log")
sp1_ax.set_yscale("log")

sp1_ax.set_ylabel("E(k)")
sp1_ax.set_xlabel("k")


r1, A1 = monomial_fit(k_vals, sp1_data)
sp1_fit = A1*k_vals**r1

sp1_ax.plot(k_vals, sp1_fit, label="$y = " + f"{A1:.2f}" + "x^{" + f"{r1:.2f}" + "}$", color="C1")


# SP2
sp2_ax.scatter(k_vals, sp2_data, marker="+", color="C0")
sp2_ax.set_title("SP2")
sp2_ax.set_xscale("log")
sp2_ax.set_yscale("log")

sp2_ax.set_xlabel("k")

sp1_ax.set_ylim(sp2_ax.get_ylim())


z = 5
r2, A2 = monomial_fit(k_vals[:-z], sp2_data[:-z])
sp2_fit = A2*k_vals**r2

sp2_ax.plot(k_vals, sp2_fit)
sp2_ax.plot(k_vals, sp2_fit, label="$y = " + f"{A2:.2f}" + "x^{" + f"{r2:.2f}" + "}$", color="C1")

print("Results")
print("SP1: $y = " + f"{A1:.2f}" + "x^{" + f"{r1:.2f}" + "}$")
print("SP2: y = $y = " + f"{A2:.2f}" + "x^{" + f"{r2:.2f}" + "}$")

for ax in [sp1_ax, sp2_ax]:
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_format))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_format))

plt.savefig("../figures/full_eoc.pdf")
plt.show()


# Computing the analytical solution for Example 1. See appendix for more details.

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from helper_functions import romans

# initial position density
def n_0(x):
    a = -25 * (x - 0.5)**2
    return np.exp(a)**2

# derivative of h w.r.t. v
def dh(x, v, t):
    sech2 = (np.cosh(5*(x - v*t - 0.5)))**(-2)
    return 1 - 5*t*sech2

# h function - see appendix
def h(x, v, t):
    return v + np.tanh(5*(x - v*t - 0.5))

def find_roots(x, t, v_range, num_guesses=100):
    """
    Finds maximum 3 roots of h over v (defined above)
    Params:
        x, t: floats. constant values of x and t
        v_range: 2-tuple. range of v values to search over.
        num_guesses: number of initial guessing over v_range to search over
    """
    # univariate function for root
    def univariate_h(v):
        return h(x, v, t)
    def univariate_dh(v):
        return dh(x, v, t)
    roots = []

    v_guesses = np.linspace(*v_range, num_guesses)
    
    for v0 in v_guesses:
        sol = root(univariate_h, v0)
        if sol.success:
            v_root = sol.x[0]
            if not any(np.isclose(v_root, r, rtol=1e-6, atol=1e-9) for r in roots):
                roots.append(v_root)
                if np.abs(v_root) > 1:
                    print("WARNING: abs(root) > 1")
        if len(roots) >= 3:
            break
    
    if len(roots) not in (1, 3):
        print(f"WARNING: only {len(roots)} roots")
    return roots

def compute_n(x, t, v_range=(-1, 1)):
    """
    computes the value of n, the position density, at (x, t)
    """
    roots = find_roots(x, t, v_range)
    n0_vals = []
    for v in roots:
        deriv = np.abs(dh(x, v, t))
        deriv = max(deriv, 1e-6) # avoid divide by 0

        n = n_0(x - v*t) / deriv
        n0_vals.append(n)
    
    n0 = sum(n0_vals)
    return n0

def compute_J(x, t, v_range=(-1, 2)):
    """
    computes the value of J, the current density, at (x, t)
    """
    roots = find_roots(x, t, v_range)
    n0_vals = []
    for v in roots:
        dh_0 = np.abs(dh(x, v, t))
        if np.abs(dh_0) < 1e-6:
            dh_0 = 1e-6
        n = v*n_0(x - v*t) / dh_0
        n0_vals.append(n)
    
    n0 = sum(n0_vals)
    return n0

def n_analytic(xarr, t):
    nvals = []
    for x in xarr:
        nvals.append(compute_n(x, t))
    return nvals

def J_analytic(xarr, t):
    Jvals = []
    for x in xarr:
        Jvals.append(compute_J(x, t))
    return Jvals


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-dark-palette")
    plt.style.use("./stylesheet.mplstyle")

    t_vals = [0.0, 0.1, 0.2, 0.3]
    x_vals = np.linspace(0, 1, 500)
    fig, axs = plt.subplots(len(t_vals), 2, layout="tight")
    for i, t in enumerate(t_vals):
        n = n_analytic(x_vals, t)
        J = J_analytic(x_vals, t)
        axs[i, 0].plot(x_vals, n, color="C0")
        axs[i, 1].plot(x_vals, J, color="C0")

        axs[i, 0].set_ylabel(f"({romans[i]})", rotation=0, fontweight="heavy", labelpad=15)

    axs[0, 0].set_title("position density")
    axs[0, 1].set_title("current density")
    plt.savefig("../figures/example1_weak_limit.pdf")
    plt.show()




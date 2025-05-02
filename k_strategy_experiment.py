# this file takes absolutely ages to run the first time, since 
# I'm using a very fine spatial and temporal mesh size for the 
# reference solution.

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *


# initial conditions
def n_0(x):
    a = -25 * (x - 0.5)**2
    return np.exp(a)**2

def S_0(x):
    return x + 1
    # return 0.2*(x**2 - x)

def u0(x, eps):
    return np.sqrt(n_0(x)) * np.exp(1j * S_0(x) / eps)

def V(x, t):
    return x**2 / 2

T = 3.072

eps_vals = [0.0256]
k_vals = [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.016, 0.024,
          0.032, 0.048, 0.064, 0.096, 0.128, 0.192, 0.256, 0.384, 0.512, 0.768, 1.024, 1.536, 3.072]


if __name__ == "__main__":

    results = np.zeros((2,len(eps_vals), len(k_vals)))

    h = 1./32768
    a, b = -2, 2


    xarr = np.arange(a, b, h)
    k_exact = 1e-5

    for i, eps in enumerate(eps_vals):
        fn = "data/eps_" + f"{eps:.6f}".replace(".", "") + "_exact.npy"
        try:
            exact_sol = np.load(fn, allow_pickle=True)
            print(f"eps = {eps:.5f}, loaded file successfully")
        except FileNotFoundError:
            print(f"eps = {eps:.5f}, calculating exact solution...")
            exact_sol = SP2(u0(xarr, eps), a, b, T, h, k_exact, eps, V)
            np.save(fn, exact_sol, allow_pickle=True)

        for j, k in enumerate(k_vals):
            print(f"eps = {eps:.5f}, k = {k:.5f}")
            print("computing SP1 approximation...")
            sp1_uarr = SP1(u0(xarr, eps), a, b, T, h, k, eps, V)
            print("computing SP2 approximation...")
            sp2_uarr = SP2(u0(xarr, eps), a, b, T, h, k, eps, V)

            # sp1_error = np.linalg.norm(sp1_uarr - exact_sol, ord=2)
            # sp2_error = np.linalg.norm(sp2_uarr - exact_sol, ord=2)
            results[0, i, j] = np.linalg.norm(sp1_uarr - exact_sol, ord=2)
            results[1, i, j] = np.linalg.norm(sp2_uarr - exact_sol, ord=2)

    output_fn = "data/k_experiment_out.npy"
    np.save(output_fn, results)
    print("results saved to", output_fn)







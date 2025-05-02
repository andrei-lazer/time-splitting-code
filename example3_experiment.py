# 2-D experiment. Takes a non-trivial amount of time to run, so in a seperate
# file to the plotting part.

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
import pickle

# initial conditions
def n_0(x, y):
    a = (x-0.5)**2
    b = (y-0.5)**2
    return np.exp(-50 * (a + 0.8*b))

def S_0(x, y):
    return x + 0.5*y

def u0(x, y, eps):
    return np.sqrt(n_0(x, y)) * np.exp(1j*S_0(x, y) / eps)

def V(x, y, t):
    return 0.5*(x**2 + y**2)

eps_vals = [0.04, 0.005, 0.000625]
h_vals = [1/16, 1/128, 1/1024]

T = 2.7
k = 0.05

ax, bx = -2, 2
ay, by = -2, 2


output_fn = "data/experiment3_out.pkl"

if __name__ == "__main__":
    results = []
    for i, (eps, h) in enumerate(zip(eps_vals, h_vals)):
        xarr = np.arange(ax, bx, h)
        yarr = np.arange(ay, by, h)
        X, Y = np.meshgrid(xarr, yarr)
        u_init = u0(X, Y, eps)
        print(f"eps: {eps}, h: {h}")
        u_arr = SP2_2D(u_init, X, Y, T, h, h, k, eps, V)
        results.append(u_arr)

    with open(output_fn, 'wb') as f:
        pickle.dump(results, f)

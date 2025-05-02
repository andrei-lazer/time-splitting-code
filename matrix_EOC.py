import numpy as np
from scipy.linalg import expm

def lie_splitting(x_0, A, B, t_0, T, k):
    def next(x_n):
        z = expm(k*A) @ expm(k*B) @ x_n
        return z

    t = t_0
    t_arr = [t_0]
    x_n = x_0
    x_arr = [x_0]
    niter = 0
    n = int(T / k)
    for _ in range(n):
        t += k
        niter += 1
        t_arr.append(t)
        x_n = next(x_n)
        x_arr.append(x_n)

    print(f"lie t final = {t}")

    return np.array(x_arr), t_arr

def strang_splitting(x_0, A, B, t_0, T, k):
    def next(x_n):
        z = expm(0.5*k*A) @ expm(k*B) @ expm(0.5*k*A) @ x_n
        return z

    t = t_0
    t_arr = [t_0]
    x_n = x_0
    x_arr = [x_0]

    n = int(T/k) + 1
    for _ in range(n):
        t += k
        t_arr.append(t)
        x_n = next(x_n)
        x_arr.append(x_n)

    print(f"strang t final = {t}")

    return np.array(x_arr), t_arr

if __name__ == "__main__":
    A = np.array([[3, 0],
         [0, -2]])
    B = np.array([[0, -2],
         [2, 0]])
    C = A+B

    x_0 = np.array([1, 1])
    t_0 = 0
    T = 1.2

    # exact solution
    true_soln = expm(T*C)@x_0

    k_vals = 0.1 * np.power(2., -np.arange(0, 9))

    # approximations for all k values
    lie_solns = np.array([lie_splitting(x_0, A, B, t_0, T, k_0)[0][-1] for k_0 in k_vals])
    strang_solns = np.array([strang_splitting(x_0, A, B, t_0, T, k_0)[0][-1] for k_0 in k_vals])

    # errors for all k values
    lie_errors = np.linalg.norm(lie_solns - true_soln, axis=1)
    strang_errors = np.linalg.norm(strang_solns - true_soln, axis=1)

    lie_eocs = np.log2(lie_errors[:-1]) - np.log2(lie_errors[1:])
    strang_eocs = np.log2(strang_errors[:-1]) - np.log2(strang_errors[1:])


    print("\nLie EOCs:")
    for l in lie_eocs:
        print("{:.5f}".format(l))


    print("\nStrang EOCS:")
    for s in strang_eocs:
        print("{:.5f}".format(s))

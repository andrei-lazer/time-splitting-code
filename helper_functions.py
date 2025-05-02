import numpy as np
import matplotlib.pyplot as plt

romans = ["i", "ii", "iii", "iv", "v", "vi"]

# fourier differentiation
def diff(y, dx, ord, axis=-1):
    n = y.shape[axis]
    # Scaled frequencies (angular frequency)
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi  

    # Compute the FFT of the function
    y_hat = np.fft.fft(y, axis=axis)
    # Scale by (i*k)^ord to compute the derivative in Fourier space
    dy_hat = (1j * k) ** ord * y_hat

    # Transform back to real space
    return np.fft.ifft(dy_hat, n=n, axis=axis)

def curr_density(u, eps, dx):
    grad_u = diff(u, dx, 1, -1)
    return eps*np.imag(np.conjugate(u)*grad_u)

def pos_density(u):
    return np.abs(u) ** 2

def SP1(u0, a, b, T, h, k, eps, V):
    xarr = np.arange(a, b, h)
    # T+k/2 is the end so that T is the last element
    tarr = np.arange(0, T + k/2, k)
    u = u0
    n = len(u)

    mu = np.fft.fftfreq(n, h) * 2 * np.pi
    for _ in tarr[1:]:
        uhat = np.fft.fft(u)

        # first step: finding U^* from U^n
        uhat *= np.exp(-0.5j*eps*k*mu**2)
        ustar = np.fft.ifft(uhat)

        # second step: finding U^{n+1} from U^*
        u = np.exp(-1j * k / eps * V(xarr, _)) * ustar

    return u # only returns the last element

def SP1_2D(u0, xgrid, ygrid, T, hx, hy, k, eps, V):
    tarr= np.arange(0, T+k/2, k)
    u = u0
    nx, ny = xgrid.shape

    # defining mu grid
    mux = np.fft.fftfreq(nx, hx) * 2 * np.pi
    muy = np.fft.fftfreq(ny, hy) * 2 * np.pi
    mux_grid, muy_grid = np.meshgrid(mux, muy)
    mu_laplace_grid = (mux_grid**2 + muy_grid**2)

    # multiplication factors
    laplace_mult = np.exp(-0.5j*eps*k*mu_laplace_grid)
    V_mult = np.exp(-1j*k / eps * V(xgrid, ygrid, 0))

    for _ in tarr[1:]:
        # getting U^* from U^n
        uhat = np.fft.fft2(u)
        uhat *= laplace_mult
        ustar = np.fft.ifft2(uhat)
        
        # second step
        ustar *= V_mult
        u = ustar

    return u

def SP2_2D(u0, xgrid, ygrid, T, hx, hy, k, eps, V):
    tarr= np.arange(0, T+k/2, k)
    u = u0
    nx, ny = xgrid.shape

    # precomputing laplace and potential multipliers
    mux = np.fft.fftfreq(nx, hx) * 2 * np.pi
    muy = np.fft.fftfreq(ny, hy) * 2 * np.pi
    mux_grid, muy_grid = np.meshgrid(mux, muy)
    mu_laplace_grid = (mux_grid**2 + muy_grid**2)

    laplace_mult = np.exp(-0.5j*eps*k*mu_laplace_grid)
    V_mult = np.exp(-0.5j * k / eps * V(xgrid, ygrid, 0))

    for _ in tarr[1:]:
        # U^n -> U^*
        ustar1 = V_mult * u

        # U^* -> U^**
        uhat = np.fft.fft2(ustar1)
        uhat *= laplace_mult
        ustar2 = np.fft.ifft2(uhat)
        
        # U^** -> U^{n+1}
        u = V_mult * ustar2

    return u

def SP2(u0, a, b, T, h, k, eps, V):
    xarr = np.arange(a, b, h)
    # T+k/2 is the end so that T is the last element
    tarr = np.arange(0, T + k/2, k)
    u = u0
    n = len(u)

    mu = np.fft.fftfreq(n, h) * 2 * np.pi
    V_mult = np.exp(-0.5j * k / eps * V(xarr, tarr))
    laplace_mult = np.exp(-0.5j*eps*k*mu**2)

    for _ in tarr[1:]:

        # first step: U^* from U^n
        ustar1 = V_mult * u

        # second step: U^** from U^*
        uhat = laplace_mult * np.fft.fft(ustar1)
        ustar2 = np.fft.ifft(uhat)

        # third step: finding U^{n+1} from U^**
        u = V_mult * ustar2

    return u

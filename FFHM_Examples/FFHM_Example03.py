# This program computes and plots the spectrum of L = -D_xx + 2q*cos(2x),
# the linear operator corresponding to the Mathieu equation L(y) = a*y.

# MODULES
import numpy as np
import scipy.integrate as integrate
import cmath
import matplotlib.pyplot as plt

# PARAMETERS
q = 2            # coefficient of the linear operator
N = 64           # number of Fourier modes
D = 256          # number of Floquet modes
f1 = lambda x: -1
f2 = lambda x: 2*q*np.cos(2*x)

# FUNCTIONS
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def fourier_coeffs(fun, modes):
    cosines = np.zeros(modes+1, dtype=np.float_)
    sines = np.zeros(modes+1, dtype=np.float)
    output = np.zeros(2*modes + 1, dtype=np.complex_)
    output[modes], err = integrate.quad(lambda x: fun(x), -1*cmath.pi/2, cmath.pi/2)
    output[modes] /= cmath.pi
    for k in range(1, modes+1):
        cosines[k], err = integrate.quad(lambda x: (fun(x) * np.cos(2*k*x)), -1*cmath.pi/2, cmath.pi/2)
        sines[k], err = integrate.quad(lambda x: (fun(x) * np.sin(2*k*x)), -1*cmath.pi/2, cmath.pi/2)
        output[modes-k] = (np.complex_(cosines[k]) + 1j * np.complex_(sines[k])) / cmath.pi
        output[modes+k] = (np.complex_(cosines[k]) - 1j * np.complex_(sines[k])) / cmath.pi
    return output

f1_vec = fourier_coeffs(f1, 2*N)
f2_vec = fourier_coeffs(f2, 2*N)
evals = np.array([], dtype=np.complex_)

for mu in frange(-1,1,2/D):
    L_matrix = np.zeros((2 * N + 1, 2 * N + 1), dtype=np.complex_)
    for n in range(-N, N+1):
        for m in range(-N, N+1):
            L_matrix[n+N, m+N] = f1_vec[2*N+(n-m)]*(1j*(mu + 2*m))**2 + f2_vec[2*N+(n-m)]
    evals = np.append(evals, np.linalg.eigvals(L_matrix))

plt.figure()
plt.scatter(evals.real, evals.imag)
plt.xlim([-3, 10])
plt.show()
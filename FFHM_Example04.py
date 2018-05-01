# FFHM_Example04.py
# This program computes and plots the spectrum of L = -D_xx + 2*(k^2)*cn(x|k)
# Improvements over FFHM_Example03.py:
# (1) the period 'L' now a parameter
# (2) matrix computation a bit cleaner
# Last edited by Ryan on 4/30/18 at 7:00 pm

# MODULES
import numpy as np
import scipy.integrate as integrate
import scipy.special as sp
import cmath
import matplotlib.pyplot as plt

# PARAMETERS
q = 2                    # coefficient of the linear operator
N = 64                   # number of Fourier modes
D = 256                  # number of Floquet modes
k = 0.3                  # only used for elliptic functions!
L = 2*sp.ellipk(k**2)    # period
f1 = lambda x: -1
f2 = lambda x: 6*(k**2)*(sp.ellipj(x, k**2)[0])**2     # sn(x|k); cf. docs for special.ellipj

# FUNCTIONS
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def fourier_coeffs(fun, modes, period):
    cosines = np.zeros(modes+1, dtype=np.float_)
    sines = np.zeros(modes+1, dtype=np.float)
    output = np.zeros(2*modes+1, dtype=np.complex_)
    output[modes], err = integrate.quad(lambda x: fun(x), -1*L/2, L/2)
    output[modes] /= period
    for k in range(1, modes+1):
        cosines[k], err = integrate.quad(lambda x: (fun(x) * np.cos((2*k*cmath.pi/period)*x)), -1*period/2, period/2)
        sines[k], err = integrate.quad(lambda x: (fun(x) * np.sin((2*k*cmath.pi/period)*x)), -1*period/2, period/2)
        output[modes-k] = (np.complex_(cosines[k]) + 1j * np.complex_(sines[k])) / period
        output[modes+k] = (np.complex_(cosines[k]) - 1j * np.complex_(sines[k])) / period
    return output

# PROGRAM
f1_vec = fourier_coeffs(f1, 2*N, L)
f2_vec = fourier_coeffs(f2, 2*N, L)
evals = np.array([], dtype=np.complex_)

print(f2_vec)

for mu in frange(-cmath.pi/L, cmath.pi/L, 2*cmath.pi/(L*D)):
    L_matrix = np.zeros((2*N + 1, 2*N + 1), dtype=np.complex_)
    for n in range(-N, N+1):
        for m in range(-N, N+1):
            factor = 1j*(mu + 2*cmath.pi*m/L)
            L_matrix[n+N, m+N] = f1_vec[2*N+(n-m)]*factor**2 + f2_vec[2*N+(n-m)]
    evals = np.append(evals, np.linalg.eigvals(L_matrix))

plt.figure()
plt.scatter(evals.real, evals.imag)
# plt.xlim([0, 10])
plt.show()
# mKdV_spectrum_01.py
# Last edited by Ryan on 5/8/18 at 11:30 am

# MODULES
import numpy as np
import scipy.integrate as integrate
import scipy.special as sp
import scipy.misc as msc
import cmath
import matplotlib.pyplot as plt

# PARAMETERS
N = 64                       # number of Fourier modes
D = 49                       # number of Floquet modes
k = 1.8                      # elliptic modulus
L = 2*sp.ellipk((1/k)**2)    # period
V = 10
U = lambda y: k*cmath.sqrt(V/(2*k**2-1))*sp.ellipj(np.real(k*cmath.sqrt(V/(2*k**2-1)))*y, (1/k)**2)[2]
U_prime = lambda y: msc.derivative(U,y)

f3 = lambda y: -1
f2 = lambda y: 0
f1 = lambda y: V + 6*U(y)**2
f0 = lambda y: -12*U(y)*U_prime(y)

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
    output[modes], err = integrate.quad(lambda x: np.real(fun(x)), -1*L/2, L/2)
    output[modes] /= period
    for k in range(1, modes+1):
        cosines[k], err = integrate.quad(lambda x: np.real(fun(x) * np.cos((2*k*cmath.pi/period)*x)), -1*period/2, period/2)
        sines[k], err = integrate.quad(lambda x: np.real(fun(x) * np.sin((2*k*cmath.pi/period)*x)), -1*period/2, period/2)
        output[modes-k] = (np.complex_(cosines[k]) + 1j * np.complex_(sines[k])) / period
        output[modes+k] = (np.complex_(cosines[k]) - 1j * np.complex_(sines[k])) / period
    print('sines: ' + str(sines))
    print('cosines: ' + str(cosines))
    print('return: ' + str(output))
    return output

# PROGRAM
f_hats = np.array([fourier_coeffs(f3, 2*N, L), fourier_coeffs(f2, 2*N, L), fourier_coeffs(f1, 2*N, L), fourier_coeffs(f0, 2*N, L)])
evals = np.array([], dtype=np.complex_)

for mu in frange(-cmath.pi/L, cmath.pi/L, 2*cmath.pi/(L*D)):
    L_matrix = np.zeros((2*N + 1, 2*N + 1), dtype=np.complex_)
    for n in range(-N, N+1):
        for m in range(-N, N+1):
            factor = np.array([(1j*(mu + 2*cmath.pi*m/L))**(3-p) for p in range(0,4)], dtype=np.complex_)
            idx = 2*N+(n-m)
            L_matrix[n+N, m+N] = np.dot(f_hats[:,idx], factor)
    evals = np.append(evals, np.linalg.eigvals(L_matrix))

plt.figure()
plt.scatter(evals.real, evals.imag)
# plt.xlim([0, 10])
plt.show()
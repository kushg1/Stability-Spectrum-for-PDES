# mKdV_spectrum_01.py

# MODULES
import numpy as np
import scipy.special
import scipy.misc
import cmath
import matplotlib.pyplot as plt
from mKdV.WeierstrassToJacobi import weierstrass_ellip_library as weier
import importlib

cmp = importlib.import_module('complex_integrator')
fs = importlib.import_module('fourier_series')

# PARAMETERS
N = 40                      # number of Fourier modes
D = 49                       # number of Floquet modes
V = 10.

sn = lambda y, k: scipy.special.ellipj(y, k**2)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2]

# Use this code to plot Figure 14
k = 0.8
mult_factor = np.sqrt(V/(2*k**2-1))
U = lambda y: k * mult_factor * cn(mult_factor*y, k)
U_prime = lambda y: -k * mult_factor**2 * sn(mult_factor*y, k) * dn(mult_factor*y, k)
L = 4*scipy.special.ellipk(k**2) / mult_factor

# k = 1.8
# mult_factor = np.sqrt(V/(2.*k**2.-1.))
# U = lambda y: k * mult_factor * dn(k*mult_factor*y, 1./k)
# U_prime = lambda y: -1 * mult_factor**2. * sn(k*mult_factor*y, 1./k) * cn(k*mult_factor*y, 1./k)
# L = 2.*scipy.special.ellipk(1./k**2.) / (k * mult_factor)

# Case of C=0 but U takes an imaginary argument
# k = 0.2
# mult_factor = np.sqrt(V/(2*k**2-1))
# U = lambda y: k * mult_factor / cn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2))
# U_prime = lambda y: -1j * k * mult_factor**2 * sn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2)) * \
#                     dn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2)) / cn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2))**2
# L = 4*scipy.special.ellipk(k**2)/mult_factor


# g2 = 1.1
# g3 = -0.1
# e1, e2, e3 = weier.weierstrass_Es(g2, g3)
# V = 10.
# jacobi_k = weier.weier_to_jacobi_k(e1, e2, e3)
# E = (jacobi_k**2 - jacobi_k**4) * V**2 / (2 * (2*jacobi_k**2 - 1)**2)
# omega1 = weier.omega1(g2, g3, e1)
# omega3 = weier.omega3(g2, g3, e3)
# y0 = 0.
# C = 0.
# fact = lambda y: weier.P(e1, e2, e3, 0.5*(y+y0)) - V/3
# denom = lambda y: (fact(y) - 2*np.sqrt(-2*E)) * (fact(y) + 2*np.sqrt(-2*E))
# PPrime = lambda y: weier.PPrime(e1, e2, e3, 0.5*(y+y0))
# PPrimePrime = lambda y: 6*(weier.P(e1, e2, e3, 0.5*(y+y0)))**2 - 0.5*g2
# U = lambda y: (np.sqrt(2*E) * PPrime(y) + C * 2 * fact(y)) / denom(y)
# U_prime = lambda y: (np.sqrt(2*E) * (0.5*PPrimePrime(y)*denom(y) - (PPrime(y)**2)*fact(y)) + C*PPrime(y)) / (denom(y))**2
# L = 2*omega1

# f(j) is the j-th term in the expression Sum[f(j)*\partial_y^j], i.e. it is
# the coefficient of the j-th derivative.
f3 = lambda y: -1
f2 = lambda y: 0
f1 = lambda y: V - 6*U(y)**2            # CHECK THE SIGN OF V!!!!!
f0 = lambda y: -12*U(y)*U_prime(y)

# FUNCTIONS
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

# PROGRAM
f_hats = np.array([fs.fourier_coeffs(f3, 2*N, L), fs.fourier_coeffs(f2, 2*N, L), fs.fourier_coeffs(f1, 2*N, L),
                   fs.fourier_coeffs(f0, 2*N, L)])
evals = np.array([], dtype=np.complex_)

mu_vals = []
imag_evals = []

for mu in frange(-cmath.pi/L, cmath.pi/L, 2*cmath.pi/(L*D)):
    L_matrix = np.zeros((2*N+1, 2*N+1), dtype=np.complex_)
    for n in range(-N, N+1):
        for m in range(-N, N+1):
            factor = np.array([(1j*(mu + 2*cmath.pi*m/L))**(3-p) for p in range(0,4)], dtype=np.complex_)
            idx = 2*N+(n-m)
            L_matrix[n+N, m+N] = np.dot(f_hats[:,idx], factor)
    eigs = np.linalg.eigvals(L_matrix)
    for e in eigs:
        mu_vals = np.append(mu_vals, [mu])
        imag_evals = np.append(imag_evals, [np.imag(e)])
    evals = np.append(evals, eigs)

plt.figure(1)
plt.scatter(mu_vals, imag_evals, marker='.')
plt.ylim([-50, 50])

plt.figure(2)
plt.scatter(evals.real, evals.imag, color=(0.05,0.75,0.5), marker='.')

fourier_U_coeffs = fs.fourier_coeffs(U, N, L)
fourier_U = lambda x: sum([fourier_U_coeffs[n+N] * np.exp(2j*p*cmath.pi*x/L) for p in range(-N,N+1,1)])

plt.figure(3)
plt.scatter([p for p in range(-N,N+1,1)], fourier_U_coeffs, marker='.')

plt.figure(4)
plt.plot([x for x in frange(-L,L,0.001)], [U(x) for x in frange(-L,L,0.001)])

plt.show()

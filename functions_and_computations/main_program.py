# This file is where "all the stuff happens." In short, the main computations currently being examined
# will be done here. After a solution's spectrum has been examined and no further computations need
# be performed, the relevant code (definition of U(y), special identities used to make the computations
# tractable, etc.) should (if necessary/appropriate) be saved in an appropriately-named file. Most of the
# time, solution_forms_for_C_equals_zero.py or weierstrass_solution_forms.py will be suitable for thus
# purpose.

import importlib
import numpy as np
import scipy.special
import cmath
import matplotlib.pyplot as plt
import scipy.misc

importlib.invalidate_caches()
gf = importlib.import_module('getfourier')
fs = importlib.import_module('fourier_series')
hill = importlib.import_module('FFHM_V2')
weier = importlib.import_module('weierstrass_elliptic_functions')

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

# Parameters
N = 35
D = 40
V = 10.
C = 0.
g2 = 1.1
g3 = -0.1
e1, e2, e3 = np.real(weier.weierstrass_Es(g2, g3))  # CASTING to REAL while we're still figuring things out!
jacobi_k = weier.weier_to_jacobi_k(e1, e2, e3)
E = (jacobi_k**2. - jacobi_k**4.) * V**2. / (2. * (2.*jacobi_k**2. - 1.)**2.)
omega1 = weier.omega1(g2, g3, e1)
omega3 = weier.omega3(g2, g3, e3)
L = 2.*omega1
y0 = L/2.

# Jacobi elliptic functions & complete elliptic integral of the first kind
# These take k as their second argument, NOT m!
sn = lambda y, k: scipy.special.ellipj(y, k**2.)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2.)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2.)[2]
K = lambda k: scipy.special.ellipk(k**2.)

fact = lambda y: weier.P(e1, e2, e3, 0.5*(y+y0)) - V/3.
denom = lambda y: (fact(y) - 2.*np.sqrt(-2.*E)) * (fact(y) + 2.*np.sqrt(-2.*E))
PPrime = lambda y: weier.PPrime(e1, e2, e3, 0.5*(y+y0))
PPrimePrime = lambda y: 6.*(weier.P(e1, e2, e3, 0.5*(y+y0)))**2 - 0.5*g2
U = lambda y: (np.sqrt(2.*E)*PPrime(y) + C*2.*fact(y)) / denom(y)
U_prime = lambda y: (denom(y) * (0.5*np.sqrt(2*E)*PPrimePrime(y) + C*PPrime(y)) -
                     PPrime(y)*fact(y)*(np.sqrt(2.*E)*PPrime(y) + 2.*C*fact(y))) / (denom(y))**2.

# Operator coefficients
f3 = lambda y: -1.
f2 = lambda y: 0
f1 = lambda y: V - 6.*U(y)**2.
f0 = lambda y: -12.*U(y)*U_prime(y)

f_hats = np.array([fs.fourier_coeffs(f3, N, L), fs.fourier_coeffs(f2, N, L), fs.fourier_coeffs(f1, N, L),
                   fs.fourier_coeffs(f0, N, L)])
evals = hill.FFHM(L,D,f_hats)

plt.figure(1)
plt.scatter(evals.real, evals.imag, color=(0.05,0.75,0.5), marker='.')

# fourier_U_coeffs = fs.fourier_coeffs(U, N, L)
# fourier_U = lambda x: sum([fourier_U_coeffs[N-p] * np.exp(2j*p*cmath.pi*x/L) for p in range(-N,N+1,1)])

# for n in range(-N, N+1):
#     print(str(n) + 'th coeff = ' + str(fourier_U_coeffs[N+n]))
#
# plt.figure(2)
# plt.plot([n for n in range(-N, N+1)], [np.real(fourier_U_coeffs[n]) for n in range(0,2*N+1)])
#
# plt.figure(3)
# plt.plot([n for n in range(-N, N+1)], [np.imag(fourier_U_coeffs[n]) for n in range(0,2*N+1)])

plt.figure(4)
plt.plot([x for x in frange(-L,L,0.01)], [U(x) for x in frange(-L,L,0.01)])

plt.figure(5)
plt.plot([x for x in frange(-L,L,0.01)], [U_prime(x) for x in frange(-L,L,0.01)])

# plt.figure(5)
# plt.plot([x for x in frange(-L,L,0.01)], [fourier_U(x) for x in frange(-L,L,0.01)])

# plt.figure(6)
# plt.scatter(mu_vals, imag_eigs, color=(0.8,0.05,0.4), marker='.')

plt.show()

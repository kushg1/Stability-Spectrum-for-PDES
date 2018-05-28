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

importlib.invalidate_caches()
gf = importlib.import_module('getfourier')
fs = importlib.import_module('fourier_series')
hill = importlib.import_module('FFHM_V2')
weier = importlib.import_module('weierstrass_backup')

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

# Parameters
N = 50
D = 49
k = 0.6
k_prime = np.sqrt(1.-k**2.)
V = 10.
C = 10.

# Jacobi elliptic functions
# These now take k as their second argument, NOT m!
sn = lambda y, k: scipy.special.ellipj(y, k**2)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2]
K = lambda k: scipy.special.ellipk(k**2)

mult_factor = 1j*np.sqrt(V/(1.-2.*k**2.))
U = lambda y: k * mult_factor / cn(np.float_(np.imag(mult_factor*y)), k_prime)
U_prime = lambda y: k * mult_factor * np.float_(np.imag(mult_factor*y)) * sn(np.float_(np.imag(mult_factor*y)), k_prime) * \
                    dn(np.float_(np.imag(mult_factor*y)), k_prime) / cn(np.float_(np.imag(mult_factor*y)), k_prime)**2.
L = 4.*K(k_prime)/np.float_(np.imag(mult_factor))

# Operator coefficients
f3 = lambda y: -1
f2 = lambda y: 0
f1 = lambda y: V - 6*U(y)**2
f0 = lambda y: -12*U(y)*U_prime(y)

f_hats = np.array([fs.fourier_coeffs(f3, N, L), fs.fourier_coeffs(f2, N, L), fs.fourier_coeffs(f1, N, L),
                   fs.fourier_coeffs(f0, N, L)])
evals, mu_vals, imag_eigs = hill.FFHM(L,D,f_hats,True)

plt.figure(1)
plt.scatter(evals.real, evals.imag, color=(0.05,0.75,0.5), marker='.')
plt.xlim([-200, 200])
plt.ylim([-200, 200])

fourier_U_coeffs = fs.fourier_coeffs(U, N, L)
fourier_U = lambda x: sum([fourier_U_coeffs[N-p] * np.exp(2j*p*cmath.pi*x/L) for p in range(-N,N+1,1)])

plt.figure(2)
plt.plot([x for x in frange(-L,L,0.01)], [np.imag(fourier_U(x)) for x in frange(-L,L,0.01)])

plt.figure(3)
plt.plot([x for x in frange(-L,L,0.01)], [np.imag(U(x)) for x in frange(-L,L,0.01)])


plt.figure(4)
plt.scatter(mu_vals, imag_eigs, color=(0.8,0.05,0.4), marker='.')

plt.show()
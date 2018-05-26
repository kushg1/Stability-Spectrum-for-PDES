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

# START DEBUGGING HERE

N = 100
D = 49
k = 0.6
V = 10.
C = 10.

sn = lambda y, k: scipy.special.ellipj(y, k**2)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2]
K = lambda k: scipy.special.ellipk(k**2)

# fact = lambda y: weier.P(e1, e2, e3, 0.5*(y+y0)) - V/3
# denom = lambda y: (fact(y) - 2*np.sqrt(-2*E)) * (fact(y) + 2*np.sqrt(-2*E))
# PPrime = lambda y: weier.PPrime(e1, e2, e3, 0.5*(y+y0))
# PPrimePrime = lambda y: 6*(weier.P(e1, e2, e3, 0.5*(y+y0)))**2 - 0.5*g2
# U = lambda y: (np.sqrt(2*E) * PPrime(y) + C * 2 * fact(y)) / denom(y)
# U_prime = lambda y: (np.sqrt(2*E) * (0.5*PPrimePrime(y)*denom(y)
#                      - (PPrime(y)**2)*fact(y)) + C*PPrime(y)) / (denom(y))**2


mult_factor = np.sqrt(V/(1.-2.*k**2.))
U = lambda y: k * mult_factor / cn(mult_factor*y, np.sqrt(1.-k**2.))
U_prime = lambda y: k * mult_factor**2. * sn(mult_factor*y, np.sqrt(1.-k**2.)) * \
                    dn(mult_factor*y, np.sqrt(1.-k**2.)) / cn(mult_factor*y, np.sqrt(1.-k**2.))**2.
L = 4.*K(np.sqrt(1.-k**2.))/mult_factor

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
plt.plot([x for x in frange(-L,L,0.01)], [fourier_U(x) for x in frange(-L,L,0.01)])

plt.figure(3)
plt.plot([x for x in frange(-L,L,0.01)], [U(x) for x in frange(-L,L,0.01)])

plt.figure(4)
plt.scatter(mu_vals, imag_eigs, color=(0.8,0.05,0.4), marker='.')

plt.show()
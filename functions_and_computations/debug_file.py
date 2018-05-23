import importlib
import numpy as np
import scipy.special
import cmath
import matplotlib.pyplot as plt

importlib.invalidate_caches()
gf = importlib.import_module('getfourier')
fs = importlib.import_module('fourier_series')
hill = importlib.import_module('FFHM_V2')

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

# START DEBUGGING HERE

# the spectrum should resemble Fig. 10
N = 40
D = 49
k = 1.8
V = 10.

sn = lambda y, k: scipy.special.ellipj(y, k**2)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2]

mult_factor = np.sqrt(V/(2.*k**2.-1.))
U = lambda y: k * mult_factor * dn(k*mult_factor*y, 1./k)
# double-check U_prime when plotting the spectrum!
U_prime = lambda y: -1. * mult_factor**2. * sn(k*mult_factor*y, 1./k) * cn(k*mult_factor*y, 1./k)
L = 2.*scipy.special.ellipk(1./k**2.) / (k * mult_factor)

f3 = lambda y: -1
f2 = lambda y: 0
f1 = lambda y: V - 6*U(y)**2            # CHECK THE SIGN OF V!!!!!
f0 = lambda y: -12*U(y)*U_prime(y)

f_hats = np.array([fs.fourier_coeffs(f3, N, L), fs.fourier_coeffs(f2, N, L), fs.fourier_coeffs(f1, N, L),
                   fs.fourier_coeffs(f0, N, L)])
evals = hill.FFHM(L,D,f_hats)

plt.figure(1)
plt.scatter(evals.real, evals.imag, color=(0.05,0.75,0.5), marker='.')
plt.xlim([-200, 200])
plt.ylim([-200, 200])

plt.show()
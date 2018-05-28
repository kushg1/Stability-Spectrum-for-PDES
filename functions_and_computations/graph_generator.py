import importlib
import numpy as np
import scipy.special
import cmath
import matplotlib.pyplot as plt
import os

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

sn = lambda y, k: scipy.special.ellipj(y, k**2)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2]
K = lambda k: scipy.special.ellipk(k**2)

N = 45
D = 49

dir = 'images'

if not os.path.exists(dir):
    os.mkdir(dir)

for V in frange(0.1, 0.5, 0.1):
    for k in frange(1/np.sqrt(2) + 0.05, 0.77, 0.01):
        f3 = lambda y: -1
        f2 = lambda y: 0
        f1 = lambda y: V - 6 * U(y) ** 2
        f0 = lambda y: -12 * U(y) * U_prime(y)
        mult_factor = np.sqrt(V / (2. * k ** 2. - 1.))
        k_prime = np.sqrt(1. - k**2.)
        U = lambda y: k * mult_factor * cn(mult_factor * y, k_prime)
        U_prime = lambda y: -k * mult_factor ** 2. * sn(mult_factor * y, k_prime) * dn(mult_factor * y, k_prime)
        L = 4. * K(k) / mult_factor
        f_hats = np.array([fs.fourier_coeffs(f3, N, L), fs.fourier_coeffs(f2, N, L),
                           fs.fourier_coeffs(f1, N, L), fs.fourier_coeffs(f0, N, L)])
        evals = hill.FFHM(L, D, f_hats)
        plt.figure()
        plt.scatter(evals.real, evals.imag, color=(0.05, 0.75, 0.5), marker='.')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.savefig(dir + '/4mkdvC0V' + str(V) + 'k' + str(k) + '.png')
        plt.xlim([-200, 200])
        plt.ylim([-200, 200])
        plt.savefig(dir + '/2mkdvC0V' + str(V) + 'k' + str(k) + '.png')
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.savefig(dir + '/1mkdvC0V' + str(V) + 'k' + str(k) + '.png')
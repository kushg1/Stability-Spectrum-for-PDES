# Computes the Jacobi elliptic functions when the inputs are complex.
# Cf. DLMF 22.6 & 22.8.

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def complex_jacobi(u, k, tol=10.**(-10.)):
    if abs(np.imag(u)) < tol:
        return sp.ellipj(np.real(u), k**2)
    real_ellip = sp.ellipj(np.real(u), k**2)
    sn_r, cn_r, dn_r = real_ellip[0], real_ellip[1], real_ellip[2]
    imag_ellip = sp.ellipj(np.imag(u), 1.-k**2)
    sn_c, cn_c, dn_c = imag_ellip[0], imag_ellip[1], imag_ellip[2]
    sn_c, cn_c, dn_c = 1j * sn_c / cn_c, 1. / cn_c, dn_c / cn_c
    denom = 1. - (k * sn_r * sn_c)**2
    sn = (sn_r * cn_c * dn_c + sn_c * cn_r * dn_r) / denom
    cn = (cn_r * cn_c - sn_r * dn_r * sn_c * dn_c) / denom
    dn = (dn_r * cn_r - k**2 * sn_r * cn_r * sn_c * dn_c) / denom
    return sn, cn, dn

k = 0.5
K = sp.ellipk(k**2.)
Kp = sp.ellipk(1.-k**2.)

plt.figure(1)
plt.plot([x for x in frange(-2.,2.,0.01)], [complex_jacobi(x*(4*K + 2j*Kp),k)[0] for x in frange(-2.,2.,0.01)])

plt.figure(2)
plt.plot([x for x in frange(-2.,2.,0.01)], [complex_jacobi(x*(6*K + 2j*Kp),k)[1] for x in frange(-2.,2.,0.01)])

plt.figure(3)
plt.plot([x for x in frange(-2.,2.,0.01)], [complex_jacobi(x*(2*K + 4j*Kp),k)[2] for x in frange(-2.,2.,0.01)])

plt.show()
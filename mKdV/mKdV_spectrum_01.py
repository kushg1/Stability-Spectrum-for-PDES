# mKdV_spectrum_01.py

# MODULES
import numpy as np
import scipy.integrate as integrate
import scipy.special
import scipy.misc
import cmath
import matplotlib.pyplot as plt
import weierstrass_ellip_library as weier

# PARAMETERS
N = 81                       # number of Fourier modes
D = 49                       # number of Floquet modes
g2 = 1.0
g3 = 0.1
e1, e2, e3 = weier.weierstrass_Es(g2, g3)
V = 10.
jacobi_k = weier.weier_to_jacobi_k(e1, e2, e3)
E = (jacobi_k**2 - jacobi_k**4) * V**2 / (2 * (2*jacobi_k**2 - 1)**2)
omega1 = weier.omega1(g2, g3, e1)
omega3 = weier.omega3(g2, g3, e3)
y0 = 0
C = 1.0

sn = lambda y, k: scipy.special.ellipj(y, k**2)[0];
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1];
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2];

# mult_factor = np.sqrt(V/(2*k**2-1));
#
# U = lambda y: k * mult_factor * cn(mult_factor*y,k)
# U_prime = lambda y: - k*mult_factor**2 * sn(mult_factor*y,k) *\
#                         dn(mult_factor*y,k)
# L = 4*scipy.special.ellipk(k**2)/mult_factor

fact = lambda y: weier.P(e1, e2, e3, 0.5*(y+y0)) - V/3
denom = lambda y: (fact(y) - 2*np.sqrt(-2*E)) * (fact(y) + 2*np.sqrt(-2*E))
PPrime = lambda y: weier.PPrime(e1, e2, e3, 0.5*(y+y0))
PPrimePrime = lambda y: 6*(weier.P(e1, e2, e3, 0.5*(y+y0)))**2 - 0.5*g2
U = lambda y: (np.sqrt(2*E) * PPrime(y) + C * 2 * fact(y)) / denom(y)
U_prime = lambda y: (np.sqrt(2*E) * (0.5*PPrimePrime(y)*denom(y) - (PPrime(y)**2)*fact(y)) + C*PPrime(y)) / (denom(y))**2
L = 2*omega1

# f(j) is the j-th term in the expression Sum[ f(j)*\partial_y^j], i.e. it is
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


# CHECK THE BOUNDS!!! Does it need to be from -1 to 1, rather than 0 to 2?
def complex_quad(fun, bound):
    real = lambda x: np.real(fun(x))
    imag = lambda x: np.imag(fun(x))
    return bound * (integrate.quad(real, 0.001, 2.)[0] + 1j * integrate.quad(imag, 0.001, 2.)[0])

def fourier_coeffs(fun, modes, period):
    cosines = np.zeros(modes+1, dtype=np.float_)
    sines = np.zeros(modes+1, dtype=np.float)
    output = np.zeros(2*modes+1, dtype=np.complex_)
    output[modes] = complex_quad(fun, 0.5 * period)
    output[modes] /= period
    for k in range(1, modes+1):
        cosines[k] = complex_quad(lambda x: fun(x) * np.cos((2*k*cmath.pi/period)*x), 0.5 * period)
        sines[k] = complex_quad(lambda x: fun(x) * np.sin((2*k*cmath.pi/period)*x), 0.5 * period)
        output[modes-k] = (np.complex_(cosines[k]) + 1j * np.complex_(sines[k])) / period
        output[modes+k] = (np.complex_(cosines[k]) - 1j * np.complex_(sines[k])) / period
    return output

# PROGRAM
f_hats = np.array([fourier_coeffs(f3, 2*N, L), fourier_coeffs(f2, 2*N, L), fourier_coeffs(f1, 2*N, L),
                   fourier_coeffs(f0, 2*N, L)])
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
plt.xlim([-150, 150])
plt.ylim([-200, 200])
plt.show()

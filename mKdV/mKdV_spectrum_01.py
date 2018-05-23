# mKdV_spectrum_01.py

# MODULES
import numpy as np
import scipy.integrate as integrate
import scipy.special
import scipy.misc
import cmath
import matplotlib.pyplot as plt

# PARAMETERS
N = 40                      # number of Fourier modes
D = 49                      # number of Floquet modes
V = 10.
k = 0.8

sn = lambda y, k: scipy.special.ellipj(y, k**2)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2]
K = lambda k: scipy.special.ellipk(k**2)

# Use this code to plot Figure 14
mult_factor = np.sqrt(V/(2*k**2-1))
U = lambda y: k * mult_factor * cn(mult_factor*y, k)
U_prime = lambda y: -k * mult_factor**2 * sn(mult_factor*y, k) * dn(mult_factor*y, k)
L = 4*K(k)/mult_factor

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

# integrates a complex function over a line in the complex plane (via a parametrization -1 <= t <= 1)
# from -bound to +bound (e.g., from -omega1 to +omega1 for a Weierstrass elliptic function)
def complex_quad(fun, bound):
    real = lambda t: np.real(bound*fun(t*bound))
    imag = lambda t: np.imag(bound*fun(t*bound))
    return integrate.quad(real, -1., 1.)[0] + 1j * integrate.quad(imag, -1., 1.)[0]

def fourier_coeffs(fun, modes, period):
    cosines = np.zeros(modes+1, dtype=np.complex_)
    sines = np.zeros(modes+1, dtype=np.complex_)
    output = np.zeros(2*modes+1, dtype=np.complex_)
    output[modes] = complex_quad(fun, 0.5 * period)
    output[modes] /= period
    for k in range(1, modes+1):
        cosines[k] = complex_quad(lambda x: fun(x) * np.cos((2*k*cmath.pi/period)*x), 0.5 * period)
        sines[k] = complex_quad(lambda x: fun(x) * np.sin((2*k*cmath.pi/period)*x), 0.5 * period)
        output[modes-k] = (cosines[k] + 1j * sines[k]) / period
        output[modes+k] = (cosines[k] - 1j * sines[k]) / period
    return output

# PROGRAM
f_hats = np.array([fourier_coeffs(f3, 2*N, L), fourier_coeffs(f2, 2*N, L), fourier_coeffs(f1, 2*N, L),
                   fourier_coeffs(f0, 2*N, L)])
evals = np.array([], dtype=np.complex_)

mu_vals = []
imag_eigs = []

for mu in frange(-cmath.pi/L, cmath.pi/L, 2*cmath.pi/(L*D)):
    L_matrix = np.zeros((2*N + 1, 2*N + 1), dtype=np.complex_)
    for n in range(-N, N+1):
        for m in range(-N, N+1):
            factor = np.array([(1j*(mu + 2*cmath.pi*m/L))**(3-p) for p in range(0,4)], dtype=np.complex_)
            idx = 2*N+(n-m)
            L_matrix[n+N, m+N] = np.dot(f_hats[:,idx], factor)
    eigs = np.linalg.eigvals(L_matrix)
    for e in eigs:
        mu_vals = np.append(mu_vals, [mu])
        imag_eigs = np.append(imag_eigs, [np.imag(e)])
    evals = np.append(evals, eigs)


plt.figure(1)
plt.scatter(mu_vals, imag_eigs, marker='.')
plt.xlim([-10, 10])
plt.ylim([-3,3])


plt.figure(2)
plt.scatter(evals.real, evals.imag, color=(0.05,0.75,0.5), marker='.')
plt.xlim([-80, 80])
plt.ylim([-200, 200])


plt.figure(3)
plt.plot([y for y in frange(-L, L, 0.01)], [U(y) for y in frange(-L, L, 0.01)], 'k-')


fourier_U_coeffs = fourier_coeffs(U, N, L)
fourier_U = lambda x: sum([fourier_U_coeffs[i+N] * np.exp(2*cmath.pi*1j*i*x/L) for i in range(-N,N+1,1)])

plt.figure(4)
plt.scatter([x for x in frange(-L,L,0.01)], [fourier_U(x) for x in frange(-L,L,0.01)], marker='.')

plt.figure(5)
plt.scatter([n for n in range(-N,N+1,1)], fourier_U_coeffs, marker='.')

for n in range(-N,N+1,1):
    print('n = ', n, ':', fourier_U_coeffs[n+N])

plt.show()

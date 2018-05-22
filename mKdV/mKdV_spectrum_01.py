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
N = 100                      # number of Fourier modes
D = 64                       # number of Floquet modes
g2 = 1.1
g3 = -0.1
e1, e2, e3 = weier.weierstrass_Es(g2, g3)
V = 10.
jacobi_k = weier.weier_to_jacobi_k(e1, e2, e3)
E = (jacobi_k**2 - jacobi_k**4) * V**2 / (2 * (2*jacobi_k**2 - 1)**2)
omega1 = weier.omega1(g2, g3, e1)
omega3 = weier.omega3(g2, g3, e3)
y0 = 0.
C = 0.

sn = lambda y, k: scipy.special.ellipj(y, k**2)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2]

# Use this code to plot Figure 14
# k = 0.85
# mult_factor = np.sqrt(V/(2*k**2-1))
# U = lambda y: k * mult_factor * cn(mult_factor*y, k)
# U_prime = lambda y: -k * mult_factor**2 * sn(mult_factor*y, k) * dn(mult_factor*y, k)
# L = 4*scipy.special.ellipk(k**2)/mult_factor


# Use this code to plot Figure 10
k = 1.8
mult_factor = np.sqrt(V/(2*k**2-1))
U = lambda y: k * mult_factor * dn(k*mult_factor*y, 1/k**2)
U_prime = lambda y: -k**4 * mult_factor**2 * sn(k*mult_factor*y, 1/k**2) * cn(k*mult_factor*y, 1/k**2)
L = 2*scipy.special.ellipk(1/k**2) / (k * mult_factor)

print(L)

# Case of C=0 but U takes an imaginary argument
# k = 0.2
# mult_factor = np.sqrt(V/(2*k**2-1))
# U = lambda y: k * mult_factor / cn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2))
# U_prime = lambda y: -1j * k * mult_factor**2 * sn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2)) * \
#                     dn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2)) / cn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2))**2
# L = 4*scipy.special.ellipk(k**2)/mult_factor

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

# integrates a complex function over a line in the complex plane (via a parametrization -1 <= t <= 1)
# from -bound to +bound (e.g., from -omega1 to +omega1 for a Weierstrass elliptic function)
def complex_quad(fun, bound):
    real = lambda t: np.real(fun(t*bound))
    imag = lambda t: np.imag(fun(t*bound))
    return bound * (integrate.quad(real, -1., 1.)[0] + 1j * integrate.quad(imag, -1., 1.)[0])

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
    print(cosines)
    print(sines)
    print(output)
    return output

# PROGRAM
f_hats = np.array([fourier_coeffs(f3, 2*N, L), fourier_coeffs(f2, 2*N, L), fourier_coeffs(f1, 2*N, L),
                   fourier_coeffs(f0, 2*N, L)])
evals = np.array([], dtype=np.complex_)

fourier_U_coeffs = fourier_coeffs(U, 2*N, L)
for k in range(-N,N+1,1):
    print('n = ' + str(k) + ': ' + str(fourier_U_coeffs[k]))

fourier_U = lambda x: sum([fourier_U_coeffs[i] * np.exp(2*cmath.pi*1j*i*x/L) for i in range(-N,N+1,1)])

plt.figure(1)
plt.scatter([a for a in frange(-3,3,0.01)], [fourier_U(k) for k in frange(-3,3,0.01)], marker='.')
plt.show()




xs = []
ys = []
#
# pts = []
#
# for e in f_hats[2]:
#     a = abs(e)
#     b = np.log(a)
#     pts = np.append(pts, b)
#
#
# the_range = range(-N,len(pts)-N)
# plt.figure(0)
# plt.plot(the_range, pts, marker='.')
# plt.ylim([-50, 50])
# plt.show()



for mu in frange(-cmath.pi/L, cmath.pi/L, 2*cmath.pi/(L*D)):
    L_matrix = np.zeros((2*N + 1, 2*N + 1), dtype=np.complex_)
    for n in range(-N, N+1):
        for m in range(-N, N+1):
            factor = np.array([(1j*(mu + 2*cmath.pi*m/L))**(3-p) for p in range(0,4)], dtype=np.complex_)
            idx = 2*N+(n-m)
            L_matrix[n+N, m+N] = np.dot(f_hats[:,idx], factor)
    eigs = np.linalg.eigvals(L_matrix)
    for e in eigs:
        xs = np.append(xs, mu)
        ys = np.append(ys, np.imag(e))
        # points = np.append(points, (mu, e))

    evals = np.append(evals, eigs)


plt.figure(1)
plt.scatter(xs, ys, marker='.')
plt.ylim([-50, 50])
plt.show()

plt.figure(2)
plt.scatter(evals.real, evals.imag, color=(0.05,0.75,0.5), marker='.')
plt.xlim([-80, 80])
plt.ylim([-200, 200])
plt.show()

plt.figure(3)   # consider adding horizontal lines and stretching the vertical axis to accentuate the "actual period"
plt.plot([y for y in frange(0., 50., 0.1)], [U(y) for y in frange(0., 50., 0.1)], 'k-')
plt.xlim([0, 50])
plt.ylim([-15, 15])
plt.show()

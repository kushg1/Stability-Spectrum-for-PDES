# Case of C != 0:
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

# Case of C=0, 1/sqrt(2) < k < 1
# k = 0.8
# mult_factor = np.sqrt(V/(2*k**2-1))
# U = lambda y: k * mult_factor * cn(mult_factor*y, k)
# U_prime = lambda y: -k * mult_factor**2 * sn(mult_factor*y, k) * dn(mult_factor*y, k)
# L = 4*scipy.special.ellipk(k**2) / mult_factor


# Case of C=0 but U takes an imaginary argument:
# k = 0.2
# mult_factor = np.sqrt(V/(2*k**2-1))
# U = lambda y: k * mult_factor / cn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2))
# U_prime = lambda y: -1j * k * mult_factor**2 * sn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2)) * \
#                     dn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2)) / cn(np.real(-1j*mult_factor*y), np.sqrt(1-k**2))**2
# L = 4*scipy.special.ellipk(k**2)/mult_factor


# Case of C = 0, k > 1:
# k = 1.8
# mult_factor = np.sqrt(V/(2*k**2-1))
# U = lambda y: k * mult_factor * dn(k*mult_factor*y, 1/k**2)
# U_prime = lambda y: -1 * mult_factor**2 * sn(k*mult_factor*y, 1/k**2) * cn(k*mult_factor*y, 1/k**2)
# L = 2*scipy.special.ellipk(1/k**2) / (k * mult_factor)
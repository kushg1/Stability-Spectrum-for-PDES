import importlib
import numpy as np

weier = importlib.import_module('weierstrass_backup')

# Some dummy variables
V = np.infty
C = np.infty

# Parameters and such
g2 = 1.1
g3 = -0.1
e1, e2, e3 = weier.weierstrass_Es(g2, g3)
jacobi_k = weier.weier_to_jacobi_k(e1, e2, e3)
E = (jacobi_k**2 - jacobi_k**4) * V**2 / (2 * (2*jacobi_k**2 - 1)**2)
omega1 = weier.omega1(g2, g3, e1)
omega3 = weier.omega3(g2, g3, e3)
L = 2*omega1
y0 = L/2.

# Functions
fact = lambda y: weier.P(e1, e2, e3, 0.5*(y+y0)) - V/3
denom = lambda y: (fact(y) - 2*np.sqrt(-2*E)) * (fact(y) + 2*np.sqrt(-2*E))
PPrime = lambda y: weier.PPrime(e1, e2, e3, 0.5*(y+y0))
PPrimePrime = lambda y: 6*(weier.P(e1, e2, e3, 0.5*(y+y0)))**2 - 0.5*g2  # UNVERIFIED
U = lambda y: (np.sqrt(2*E) * PPrime(y) + C * 2 * fact(y)) / denom(y)
U_prime = lambda y: (np.sqrt(2*E) * (0.5*PPrimePrime(y)*denom(y)
                     - (PPrime(y)**2)*fact(y)) + C*PPrime(y)) / (denom(y))**2
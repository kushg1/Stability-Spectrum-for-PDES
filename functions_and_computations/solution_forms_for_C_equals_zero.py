# This file contains the various forms of the focusing mKdV solution when C = 0, i.e., when
# the solution can we written entirely in terms of Jacobi elliptic functions.
# The numbering scheme is being used to write a program that will decide these various cases
# automatically.

import numpy as np
import scipy

sn = lambda y, k: scipy.special.ellipj(y, k**2)[0]
cn = lambda y, k: scipy.special.ellipj(y, k**2)[1]
dn = lambda y, k: scipy.special.ellipj(y, k**2)[2]
K = lambda k: scipy.special.ellipk(k**2)

# Some dummy variables so that the IDE doesn't keep freaking out:
V = np.infty
k = np.infty
k_prime = np.sqrt(1.-k**2.)

# CASE 1. 0 < k < 1/sqrt(2), V < 0
mult_factor = np.sqrt(V/(2.*k**2.-1.))
U = lambda y: k * mult_factor / cn(mult_factor*y, k_prime)
U_prime = lambda y: k * mult_factor**2. * sn(mult_factor*y, k_prime) * \
                    dn(mult_factor*y, k_prime) / cn(mult_factor*y, k_prime)**2.
L = 4.*K(k) / mult_factor

# CASE 2. 1/sqrt(2) < k < 1, V > 0
mult_factor = np.sqrt(V/(2.*k**2.-1.))
U = lambda y: k * mult_factor * cn(mult_factor*y, k_prime)
U_prime = lambda y: -k * mult_factor**2. * sn(mult_factor*y, k_prime) * dn(mult_factor*y, k_prime)
L = 4.*K(k_prime) / mult_factor

# CASE 3. 1 < k, V > 0
mult_factor = np.sqrt(V/(2.*k**2.-1.))
U = lambda y: k * mult_factor * cn(mult_factor*y, k)
U_prime = lambda y: -k * mult_factor**2. * sn(mult_factor*y, k) * dn(mult_factor*y, k)
L = 4.*K(k) / mult_factor

# CASE 4. ????? (I don't think this is an actual case)
mult_factor = np.sqrt(V/(2.*k**2.-1.))
U = lambda y: k * mult_factor * cn(mult_factor*y, k_prime)
U_prime = lambda y: -k * mult_factor**2. * sn(mult_factor*y, k_prime) * dn(mult_factor*y, k_prime)
L = 4.*K(k_prime) / mult_factor



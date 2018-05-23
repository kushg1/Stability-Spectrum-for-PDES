import numpy as np
import scipy.integrate as integrate

# integrates a complex function over a line in the complex plane (via a parametrization -1 <= t <= 1)
# from -bound to +bound (e.g., from -omega1 to +omega1 for a Weierstrass elliptic function)
def complex_quad(fun, bound):
    real = lambda t: np.real(bound * fun(t*bound))
    imag = lambda t: np.imag(bound * fun(t*bound))
    return integrate.quad(real, -1., 1.)[0] + 1j * integrate.quad(imag, -1., 1.)[0]
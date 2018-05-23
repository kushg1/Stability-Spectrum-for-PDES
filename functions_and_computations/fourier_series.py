import numpy as np
import cmath
import importlib

cmp = importlib.import_module('complex_integrator')

def fourier_coeffs(fun, modes, period):
    output = np.zeros(2*modes+1, dtype=np.complex_)
    output[modes] = cmp.complex_quad(fun, 0.5*period) / period
    for k in range(1, modes+1):
        output[modes-k] = cmp.complex_quad(lambda x: fun(x) * np.exp((-2j*k*cmath.pi/period)*x), 0.5*period) / period
        output[modes+k] = cmp.complex_quad(lambda x: fun(x) * np.exp((2j*k*cmath.pi/period)*x), 0.5*period) / period
    return output

# OLD STUFF:
# cosines = np.zeros(modes+1, dtype=np.float_)
# sines = np.zeros(modes+1, dtype=np.float)
# THIS WAS IN THE FOR LOOP:
# cosines[k] = complex_quad(lambda x: fun(x) * np.cos((2*k*cmath.pi/period)*x), 0.5 * period)
# sines[k] = complex_quad(lambda x: fun(x) * np.sin((2*k*cmath.pi/period)*x), 0.5 * period)
# output[modes-k] = (np.complex_(cosines[k]) + 1j * np.complex_(sines[k])) / period
# output[modes+k] = (np.complex_(cosines[k]) - 1j * np.complex_(sines[k])) / period
import numpy as np
import importlib

cmp = importlib.import_module('complex_integrator')

# gives the kth Fourier coefficient of the function
def getfourier(fun, k, period):
    return cmp.complex_quad(lambda x: (fun(x) * np.exp(-2j*k*np.pi*x/period)), 0.5*period) / period
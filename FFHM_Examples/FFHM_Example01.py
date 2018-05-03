import numpy, cmath

# a range function that takes floats as step size
# currently not being used--why did I make this?
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

# number of Fourier modes
N = 16
# number of Floquet modes
D = 100

c = numpy.zeros((2*N+1, 2*N+1), dtype=numpy.complex_)

for d in range(0,D):
    mu = 2.0*d/D
    for k in range(0,2*N+1):
        m = k-N
        c[k,k] = -(1j*(mu + 2*m))**2
    print('eigs for mu = ' + str(mu) + ': ' + str(numpy.linalg.eig(c)))

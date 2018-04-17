import numpy, cmath
import matplotlib.pyplot as plt

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
D = 100 # we have to choose this even

# Define parameter eps for d_x term.
eps = 0.1

c = numpy.zeros((2*N+1, 2*N+1), dtype=numpy.complex_)

evals = numpy.empty([], dtype=numpy.complex_)

# choose a mu interval symmetric about 0.
for d in range(-D/2,D/2):
    mu = 2.0*d/D
    for k in range(0,2*N+1):
        m = k-N
        fac = 1j*(mu+2*m) # derivative term
        c[k,k] = -(fac)**2 + eps*fac # -d_x^2 + eps*d_x
    evals = numpy.append(evals, numpy.linalg.eigvals(c))


    #print('eigs for mu = ' + str(mu) + ': ' + str(numpy.linalg.eig(c)))

plt.figure()
plt.scatter(evals.real, evals.imag)

plt.show()

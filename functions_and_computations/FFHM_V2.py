import numpy as np
import cmath

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def FFHM(L,D,f_hats,extras=False):
    evals = np.array([], dtype=np.complex_)
    N = len(f_hats[0,:])
    mid = int((N-1)/2)
    mu_vals = []
    imag_eigs = []
    for mu in frange(-cmath.pi/L, cmath.pi/L, 2*cmath.pi/(L*D)):
        f3_matrix = np.zeros((N, N), dtype=np.complex_) + np.diag([f_hats[0,mid] for q in range(0,N)])
        f2_matrix = np.zeros((N, N), dtype=np.complex_) + np.diag([f_hats[1,mid] for q in range(0,N)])
        f1_matrix = np.zeros((N, N), dtype=np.complex_) + np.diag([f_hats[2,mid] for q in range(0,N)])
        f0_matrix = np.zeros((N, N), dtype=np.complex_) + np.diag([f_hats[3,mid] for q in range(0,N)])
        for n in range(1,mid+1):
            f3_matrix += np.diag([f_hats[0,mid+n] for q in range(0,N-n)], n) \
                         + np.diag([f_hats[0,mid-n] for q in range(0,N-n)], -n)
            f2_matrix += np.diag([f_hats[1,mid+n] for q in range(0,N-n)], n) \
                         + np.diag([f_hats[1,mid-n] for q in range(0,N-n)], -n)
            f1_matrix += np.diag([f_hats[2,mid+n] for q in range(0,N-n)], n) \
                         + np.diag([f_hats[2,mid-n] for q in range(0,N-n)], -n)
            f0_matrix += np.diag([f_hats[3,mid+n] for q in range(0,N-n)], n) \
                         + np.diag([f_hats[3,mid-n] for q in range(0,N-n)], -n)
        for m in range(0,N):
            fact = (1j*mu + 2j*cmath.pi*(m-mid)/L)
            f3_matrix[:,m] = [f3_matrix[k,m] * fact**3 for k in range(0,N)]
            f2_matrix[:,m] = [f2_matrix[k,m] * fact**2 for k in range(0,N)]
            f1_matrix[:,m] = [f1_matrix[k,m] * fact for k in range(0,N)]
        eigs = np.linalg.eigvals(f3_matrix + f2_matrix + f1_matrix + f0_matrix)
        evals = np.append(evals, eigs)
        if extras:
            mu_vals = np.append(mu_vals, [mu for eig in eigs])
            imag_eigs = np.append(imag_eigs, np.imag(eigs))
    if extras:
        return [evals, mu_vals, imag_eigs]
    return evals
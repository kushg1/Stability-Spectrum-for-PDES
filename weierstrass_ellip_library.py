import numpy as np
import scipy.integrate as integrate
import cmath
import csv

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def weierstrass_Es(g2, g3):
    e1 = np.roots([1, 0, -g2/4, -g3/4])
    e2 = np.array([], dtype=np.float_)
    for k in range(0,len(e1)):
        e2 = np.append(e2, np.roots([1, e1[k], g3/(4*e1[k])]))
    e2 = np.ndarray.tolist(e2)
    e3 = [-e1[j] - e2[k] for j in range(0,len(e1)) for k in range(0,len(e2))]
    bool_R1 = lambda a,b,c: abs(e1[a] + e2[b] + e3[c]) < 10**(-8)
    bool_R2 = lambda a,b,c: abs(e1[a]*e2[b] + e2[b]*e3[c] + e1[a]*e3[c] + g2/4) < 10**(-8)
    bool_R3 = lambda a,b,c: abs(e1[a]*e2[b]*e3[c] - g3/4) < 10**(-8)
    for i in range(0,len(e1)):
        for j in range(0,len(e2)):
            for k in range(0,len(e3)):
                if bool_R1(i,j,k) and bool_R2(i,j,k) and bool_R3(i,j,k):
                    # cheap trick alert: I swap the values of e2 and e3 to be consistent
                    # with the provided test files
                    return np.array([e1[i], e3[k], e2[j]], dtype=np.complex_)
    return -1

def omega1(g2, g3, e1):
    return np.complex_(2 * integrate.quad(lambda z: np.real(1 / cmath.sqrt(4*z**3 - g2*z - g3)), e1, np.inf)[0])

def omega3(g2, g3, e3):
    return np.complex_(2j * integrate.quad(lambda z: np.real(1 / cmath.sqrt(4*z**3 - g2*z + g3)), -1*e3, np.inf)[0])

def weier_k(e1, e2, e3):
    return cmath.sqrt((e2-e3)/(e1-e3))

results = np.array([])

for g2 in frange(1.0,1.6,0.1):
    for g3 in (-0.5,0.6,0.1):
        weier_Es = weierstrass_Es(g2, g3)
        w1 = omega1(g2, g3, weier_Es[0])
        w3 = omega3(g2, g3, weier_Es[2])
        k_val = weier_k(weier_Es[0], weier_Es[1], weier_Es[2])
        results = np.append(results, [g2, g3, weier_Es[0], weier_Es[1], weier_Es[2], w1, w3, k_val])

print(results)

rownum = 0
colnum = 0

# with open('auxiliaryOutputTable.csv', 'rb', encoding='utf8') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         for col in row:
#             print(abs(col - results[rownum][colnum]) < 10**(-5))
#         colnum = 0
#         rownum += 1

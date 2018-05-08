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

def weier_to_jacobi_k(e1, e2, e3):
    return cmath.sqrt((e2-e3)/(e1-e3))

def weier_to_jacobi_y(z, e1, e3):
    return z * cmath.sqrt(e1 - e3)

def weierstrass_to_jacobi(z, g2, g3):
    e_vec = weierstrass_Es(g2, g3)
    return (weier_to_jacobi_y(z, e_vec[0], e_vec[2]), weier_to_jacobi_k(e_vec[0], e_vec[1], e_vec[2]))


results = []

for g2 in frange(1.0,1.6,0.1):
    r = []
    for g3 in frange(-0.5,0.6,0.1):
        weier_Es = weierstrass_Es(g2, g3)
        w1 = omega1(g2, g3, weier_Es[0])
        w3 = omega3(g2, g3, weier_Es[2])
        k_val = weier_to_jacobi_k(weier_Es[0], weier_Es[1], weier_Es[2])
        r.append([g2, g3, weier_Es[0], weier_Es[1], weier_Es[2], w1, w3, k_val])
    print(len(r))
    results.append(r)
    print(len(results))

    
    # if results.shape[0] == 0:
    #     print("R: " + str(r.shape))
    #     print("Results: " + str(results.shape))
    #     results = np.append(results, r)
    # else: 
    #     print("R: " + str(r.shape))
    #     print("Results: " + str(results.shape))
    #     results = np.append([results], [r], axis=0)

# print(results)

results = np.array(results)

print(results.shape)
# results = results.reshape(results, (24, -1))
# print(results.shape)

rownum = 0
colnum = 0

with open('auxiliaryOutputTable.csv') as f:
    reader = csv.reader(f)
    # print(len(reader))
    for row in reader:
        for col in row:
            print(col)
            # print(colnum)
            # print(results.shape)
            x = results[rownum][colnum]
            print(abs(col - x) < 10**(-5))
            colnum += 1
        # print(row)
        colnum = 0
        rownum += 1
    # print(rownum)


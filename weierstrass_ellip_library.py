import numpy as np
import scipy.integrate as integrate
import cmath
import csv
# from mpmath import sqrt, mpc, sin, ellipfun, mpf
import scipy

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def weierstrass_Es(g2, g3):
    roots = np.roots([4, 0, -g2, -g3])
    return np.fliplr([np.sort_complex(roots)])[0]

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

# TODO: WORKS IF e1, e2, e3 ARE REAL
def P(e1, e2, e3, z):
    Delta = 16 * ((e2 - e3) * (e3 - e1) * (e1 - e2)) ** 2
    m = np.real((e2 - e3) / (e1 - e3))
    print(np.sqrt(m))
    if Delta > 0:
        # TODO: Shouldn't necessarily be real
        zs = np.real(np.lib.scimath.sqrt(e1 - e3) * z)
        # print(type(zs))
        # print(m)
        ellip = scipy.special.ellipj(zs, m)
        sn = ellip[0]
        retval = e3 + (e1 - e3) / sn**2
    elif Delta < 0:
        H2 = 2 * (e2 ** 2) + e1 * e3
        zp = 2 * z * np.lib.scimath.sqrt(H2)
        ellip = scipy.special.ellipj(zp, m)
        cn = ellip[1]
        retval = e2 + H2 * (1 + cn) / (1 - cn)
    
    return retval

# TODO: WORKS IF e1, e2, e3 ARE REAL
def PPrime(e1, e2, e3, z):
    Delta = 16 * ((e2 - e3) * (e3 - e1) * (e1 - e2)) ** 2
    m = np.real((e2 - e3) / (e1 - e3))
    if Delta > 0:
        zs = np.lib.scimath.sqrt(e1 - e3) * z
        print(type(zs))
        print(type(m))
        ellip = scipy.special.ellipj(zs, m)
        sn, cn, dn = ellip[0], ellip[1], ellip[2]
        retval = -2 * (np.lib.scimath.sqrt((e1 - e3) ** 3)) * cn * dn / (sn ** 3)
    elif Delta < 0:
        H2 = 2 * (e2 ** 2) + e1 * e3
        zp = 2 * z * np.lib.scimath.sqrt(H2)
        ellip = scipy.special.ellipj(zp, m)
        sn, cn, dn = ellip[0], ellip[1], ellip[2]
        retval = -4 * (np.lib.scimath.sqrt(H2 ** 3)) * sn * dn / ((1 - cn) ** 2)

    return retval


def main():
    z = 1
    g2 = 1
    g3 = 0.1
    weier = weierstrass_Es(g2, g3)
    print(weier)
    p = P(weier[0], weier[1], weier[2], z)

    print(p)


    # results = []
    # for g2 in frange(1.0,1.6,0.1):
    #     r = []
    #     for g3 in frange(-0.5,0.6,0.1):
    #         weier_Es = weierstrass_Es(g2, g3)
    #         w1 = omega1(g2, g3, weier_Es[0])
    #         w3 = omega3(g2, g3, weier_Es[2])
    #         k_val = weier_k(weier_Es[0], weier_Es[1], weier_Es[2])
    #         r.append([g2, g3, weier_Es[0], weier_Es[1], weier_Es[2], w1, w3, k_val])
    #     print(len(r))
    #     results.append(r)
    #     print(len(results))

    #     # if results.shape[0] == 0:
    #     #     print("R: " + str(r.shape))
    #     #     print("Results: " + str(results.shape))
    #     #     results = np.append(results, r)
    #     # else: 
    #     #     print("R: " + str(r.shape))
    #     #     print("Results: " + str(results.shape))
    #     #     results = np.append([results], [r], axis=0)

    # # print(results)

    # results = np.array(results)

    # print(results.shape)
    # # results = results.reshape(results, (24, -1))
    # # print(results.shape)

    # rownum = 0
    # colnum = 0

    # with open('auxiliaryOutputTable.csv') as f:
    #     reader = csv.reader(f)
    #     # print(len(reader))
    #     for row in reader:
    #         for col in row:
    #             print(col)
    #             # print(colnum)
    #             # print(results.shape)
    #             x = results[rownum][colnum]
    #             print(abs(col - x) < 10**(-5))
    #             colnum += 1
    #         # print(row)
    #         colnum = 0
    #         rownum += 1
    #     # print(rownum)

if __name__ == "__main__":
    main()
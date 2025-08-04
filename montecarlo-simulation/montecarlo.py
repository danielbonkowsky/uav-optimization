from sympy import *
import random

wB = Matrix([0, 0, 0])

# Let K = M * N for ease of scheduling


H = symbols('H')
v = symbols('v')
T = symbols('T')
N = symbols('N')
TC = T / N

def wA(n, r, c):
    return Matrix([c + r*cos( (v*n*TC)/r),
                   H,
                   r*sin( (v*n*TC)/r )])

PUtx = symbols('PUtx')
GT = symbols('GT')
GR = symbols('GR')

lam = symbols('lam')

def PA(n, r, c, xk, zk):
    wU = Matrix([xk, 0, zk])

    return PUtx * GT * GR * (lam / (4 * pi * (wA(n, r, c) - wU).norm()))**2

PAtx = symbols('PAtx')

def PB(n, r, c):
    return PAtx * GT * GR * (lam / (4 * pi * wA(n, r, c).norm()))**2

N0 = symbols('N0')

def SNRUA(n, r, c, xk, zk, M):
    return PA(n, r, c, xk, zk) / (N0 / M)

def SNRAB(n, r, c):
    return PB(n, r, c) / N0

def SEUA(n, r, c, xk, zk, M):
    return log(1 + SNRUA(n, r, c, xk, zk, M), 2)

def SEAB(n, r, c):
    return log(1 + SNRAB(n, r, c), 2)

D = symbols('D')
RD = symbols('RD')

def generateUsers(K):
    users = []

    for i in range(K):
        xk = D + RD * random.uniform(-1, 1)
        zk = RD * random.uniform(-1, 1)
        users.append(Matrix([xk, 0, zk]))

    return users

def meanSE(alpha, r, c):
    pass

print(generateUsers(10))

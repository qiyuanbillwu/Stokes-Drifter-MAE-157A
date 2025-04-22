import numpy as np
import matplotlib.pyplot as plt

def compute_A(t0, t1):
    A = np.zeros((8, 8))
    for col in range(8):
        A[0, col] = t0**col
        A[4, col] = t1**col
        if col >= 1:
            A[1, col] = col * t0**(col - 1)
            A[5, col] = col * t1**(col - 1)
        if col >= 2:
            A[2, col] = col * (col - 1) * t0**(col - 2)
            A[6, col] = col * (col - 1) * t1**(col - 2)
        if col >= 3:
            A[3, col] = col * (col - 1) * (col - 2) * t0**(col - 3)
            A[7, col] = col * (col - 1) * (col - 2) * t1**(col - 3)
    return A



# global constants
g = 9.81
m = 0.5
dt = 0.01

# boundary points and conditions
x0, y0, z0 = 1, -1, 1
x1, y1, z1 = -1.5, 0, 2.5
x2, y2, z2 = 1, 1, 1
vy = 4.0
T = 5

t0, t1, t2 = 0, 1.5, 3

r0 = np.array([x0, y0, z0])
v0 = np.array([0, 0, 0])
a0 = np.array([0, 0, 0])
j0 = np.array([0, 0, 0])
r1 = np.array([x1, y1, z1])
v1 = np.array([0, vy, 0])
a1 = np.array([T/m, 0, -g])
j1 = np.array([0, 0, 0])
r2 = np.array([x2, y2, z2])
v2, a2, j2 = v0, a0, j0

A1 = compute_A(t0, t1)
A2 = compute_A(t1, t2)

#print(A1)

b1 = np.vstack((r0, v0, a0, j0, r1, v1, a1, j1))
b2 = np.vstack((r1, v1, a1, j1, r2, v2, a2, j2))

#print(b1)

a1 = np.linalg.solve(A1, b1)
a2 = np.linalg.solve(A2, b2)

#print(a1)

def get_state(t):
    if t < t0 or t > t2:
        print("time must be within range")
    if t < t1:
        a_coeff = a1
    else:
        a_coeff = a2

    #print(a_coeff)

    r = np.array([t**0, t**1, t**2, t**3, t**4, t**5, t**6, t**7]) @ a_coeff
    v = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6]) @ a_coeff
    a = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5]) @ a_coeff

    state = np.concatenate((r, v, a))
    print(state)

    return state

get_state(1.49)
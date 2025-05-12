import numpy as np
import matplotlib.pyplot as plt
from util import get_a_dot_hat, cross_matrix, allocation_matrix
from constants import J, l, d, m, g

a_matrix  = allocation_matrix(l, d)

# compute the matrix to solve polynomial coeffcient 
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

dt = 0.01

# boundary points and conditions
x0, y0, z0 = 0, -2, 2
x1, y1, z1 = 0, 0, 2
x2, y2, z2 = 0, 2, 2
vy = 4.0
T = 5
theta = 45 * np.pi / 180 # angle of the gate

t0, t1, t2 = 0, 1, 2

r0 = np.array([x0, y0, z0])
v0 = np.array([0, 0, 0])
a0 = np.array([0, 0, 0])
j0 = np.array([0, 0, 0])
r1 = np.array([x1, y1, z1])
v1 = np.array([0, vy, 0])
a1 = np.array([T*np.cos(theta)/m, 0, T*np.sin(theta)/m-g])
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

# gets the current state at time t, using the trajectory kinematics
# gets states bewteen 2 points and an intermediate point in bewteen 
# outputs position, velocity, accelration, jerk, snap, quarternion, angular velocity, angular acceleration
def get_state(t):
    if t < t0 or t > t2:
        if t < t0:
            r = r0
        else:
            r = r2
        v = v0
        a = v0
        j = v0
        s = v0
        q_d = np.array([1, 0, 0, 0])
        w = v0
        wdot = v0

        a_d = a + np.array([0 ,0, g])
        tau = J @ wdot + np.cross(w, J@w)
        # print("tau: ", tau)

        # thrust
        T = m * np.linalg.norm(a_d)

        # Combine total thrust and torques
        tau_full = np.array([T, *tau])

        # Solve for motor forces
        f = np.linalg.solve(a_matrix, tau_full)
        
        state = {
        "r": r,         # position
        "v": v,         # velocity
        "q": q_d,       # quarternion
        "w": w,         # angular velocity
        "wdot": wdot,   # angular acceleration
        "a": a,         # acceleration
        "j": j,         # jerk
        "s": s,         # snap
        "f": f          # forces
        }
        # print(state)

        return state
    if t < t1:
        a_coeff = a1
    else:
        a_coeff = a2

    #print(a_coeff)

    T_d_hat = np.array([0, 0, 1])
    I = np.identity(3)

    # calculate the kinematics at current time
    r = np.array([t**0, t**1, t**2, t**3, t**4, t**5, t**6, t**7]) @ a_coeff
    v = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6]) @ a_coeff
    a = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5]) @ a_coeff
    j = np.array([0, 0, 0, 6, 24*t, 60*t**2, 120*t**3, 210*t**4]) @ a_coeff 
    s = np.array([0, 0, 0, 0, 24, 120*t, 360*t**2, 840*t**3]) @ a_coeff 

    a_d = a + np.array([0 ,0, g])
    #print("a_d: ", a_d)
    #print("norm(a_d): ", np.linalg.norm(a_d))
    a_d_hat = a_d / np.linalg.norm(a_d)
    theta = np.arccos(np.dot(T_d_hat, a_d_hat))
    # print("theta: ", theta)

    if t == t0 or t == t2:  # accounts for when the denominator is 0 
        n_hat = np.array([0, 0, 1])
        w = np.array([0, 0, 0])
        wdot = np.array([0, 0, 0])
        q_d = np.concatenate(([np.cos(theta/2)], n_hat*np.sin(theta)))
    else: 
        n = np.cross(T_d_hat, a_d_hat)
        # print('n: ', n)
        n_hat = n / np.linalg.norm(n)

        # print("n_hat: ", n_hat)
        n_cross = cross_matrix(n_hat)
        R_d = I + np.sin(theta) * n_cross + (1-np.cos(theta)) * n_cross @ n_cross 
        q_d = np.concatenate(([np.cos(theta/2)], n_hat*np.sin(theta)))
        # print("quarternion: ", q_d)

        a_hat_dot = get_a_dot_hat(a_d, j)
        w = np.transpose(R_d) @ a_hat_dot
        # print("a_hat_dot: ", a_hat_dot)
        # print(get_a_dot_hat(a_d, j))

        wx = -w[1]
        w[1] = -w[0]
        w[0] = wx
        w[2] = 0

        a_hat_doubledot = s / np.linalg.norm(a_d) - (2 * j * (np.transpose(a_d) @ j) + a_d * (np.transpose(j) @ j + np.transpose(a_d) @ s)) / np.linalg.norm(a_d)**3 
        + 3 * a_d * (np.transpose(a_d) @ j)**2 / np.linalg.norm(a_d)**5
        wdot = np.transpose(R_d) @ a_hat_doubledot - cross_matrix(w) @ np.transpose(R_d) @ a_hat_dot

    tau = J @ wdot + np.cross(w, J@w)
    # print("tau: ", tau)

    # thrust
    T = m * np.linalg.norm(a_d)

    # Combine total thrust and torques
    tau_full = np.array([T, *tau])

        # Solve for motor forces
    f = np.linalg.solve(a_matrix, tau_full)

    # print("w: ", w)
    # print("wdot: ", wdot)

    # print("r: ", r)
    # print("v: ", v)
    # print("a: ", a)
    # print(a_d)
    # print("a_d_hat: ", a_d_hat)
    # print("jerk: ", j)
    # print("snap: ", s)

    state = {
        "r": r,         # position
        "v": v,         # velocity
        "q": q_d,       # quarternion
        "w": w,         # angular velocity
        "wdot": wdot,   # angular acceleration
        "a": a,         # acceleration
        "j": j,         # jerk
        "s": s,          # snap
        "f": f          # force
    }
    # print(state)

    return state

# get_state(-1)

# get states between 2 points 
def get_state_simple(t):
    # boundary points and conditions

    # moving along the x axis
    #x0, y0, z0 = -1, 0, 1
    #x2, y2, z2 = 1, 0, 1 

    # moving along the y axis
    # x0, y0, z0 = 0, -1, 1
    # x2, y2, z2 = 0, 1, 1 

    # moving along the z axis
    # x0, y0, z0 = 0, 0, 0.5
    # x2, y2, z2 = 0, 0, 3 

    # moving along x, y, and z in a straight line
    x0, y0, z0 = -1, -1, 1
    x2, y2, z2 = 1, 1, 3 

    t0, t2 = 0, 5

    r0 = np.array([x0, y0, z0])
    v0 = np.array([0, 0, 0])
    a0 = np.array([0, 0, 0])
    j0 = np.array([0, 0, 0])
    r2 = np.array([x2, y2, z2])
    v2, a2, j2 = v0, a0, j0

    if t < t0 or t > t2:
        if t < t0:
            r = r0
        else:
            r = r2
        v = v0
        a = v0
        j = v0
        s = v0
        q_d = np.array([1, 0, 0, 0])
        w = v0
        wdot = v0
        
        a_d = a + np.array([0 ,0, g])
        tau = J @ wdot + np.cross(w, J@w)
        # print("tau: ", tau)

        # thrust
        T = m * np.linalg.norm(a_d)

        # Combine total thrust and torques
        tau_full = np.array([T, *tau])

        # Solve for motor forces
        f = np.linalg.solve(a_matrix, tau_full)
        
        state = {
        "r": r,         # position
        "v": v,         # velocity
        "q": q_d,       # quarternion
        "w": w,         # angular velocity
        "wdot": wdot,   # angular acceleration
        "a": a,         # acceleration
        "j": j,         # jerk
        "s": s,         # snap
        "f": f          # forces
        }
        #print(state)

        return state

    A = compute_A(t0, t2)
    b = np.vstack((r0, v0, a0, j0, r2, v2, a2, j2))
    a_coeff = np.linalg.solve(A, b)

    T_d_hat = np.array([0, 0, 1])
    I = np.identity(3)

    # calculate the kinematics at current time
    r = np.array([t**0, t**1, t**2, t**3, t**4, t**5, t**6, t**7]) @ a_coeff
    v = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6]) @ a_coeff
    a = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5]) @ a_coeff
    j = np.array([0, 0, 0, 6, 24*t, 60*t**2, 120*t**3, 210*t**4]) @ a_coeff 
    s = np.array([0, 0, 0, 0, 24, 120*t, 360*t**2, 840*t**3]) @ a_coeff 

    a_d = a + np.array([0 ,0, g])
    # print("a_d: ", a_d)
    #print("norm(a_d): ", np.linalg.norm(a_d))
    a_d_hat = a_d / np.linalg.norm(a_d)
    # print("a_d_hat: ", a_d_hat)
    theta = np.arccos(np.dot(T_d_hat, a_d_hat))
    # print("theta: ", theta)

    if np.array_equal(a_d_hat, np.array([0,0,1])):  # accounts for when the denominator is 0 
        n_hat = np.array([0, 0, 1])
        w = np.array([0, 0, 0])
        wdot = np.array([0, 0, 0])
        q_d = np.concatenate(([np.cos(theta/2)], n_hat*np.sin(theta)))
    else: 
        n = np.cross(T_d_hat, a_d_hat)
        # print('n: ', n)
        n_hat = n / np.linalg.norm(n)

        # print("n_hat: ", n_hat)
        n_cross = cross_matrix(n_hat)
        R_d = I + np.sin(theta) * n_cross + (1-np.cos(theta)) * n_cross @ n_cross 
        q_d = np.concatenate(([np.cos(theta/2)], n_hat*np.sin(theta)))
        # print("quarternion: ", q_d)

        a_hat_dot = get_a_dot_hat(a_d, j)
        w = np.transpose(R_d) @ a_hat_dot
        # print("a_hat_dot: ", a_hat_dot)
        # print(get_a_dot_hat(a, j))

        wx = -w[1]
        w[1] = -w[0]
        w[0] = wx
        w[2] = 0

        a_hat_doubledot = s / np.linalg.norm(a_d) - (2 * j * (np.transpose(a_d) @ j) + a_d * (np.transpose(j) @ j + np.transpose(a_d) @ s)) / np.linalg.norm(a_d)**3 
        + 3 * a_d * (np.transpose(a_d) @ j)**2 / np.linalg.norm(a_d)**5
        wdot = np.transpose(R_d) @ a_hat_doubledot - cross_matrix(w) @ np.transpose(R_d) @ a_hat_dot

    a_d = a + np.array([0 ,0, g])
    tau = J @ wdot + np.cross(w, J@w)
    # print("tau: ", tau)

    # thrust
    T = m * np.linalg.norm(a_d)
    # print("T: ", T)

    # Combine total thrust and torques
    tau_full = np.array([T, *tau])
    # print("tau_full (traj): ", tau_full)

    # Solve for motor forces
    f = np.linalg.solve(a_matrix, tau_full)
    # print("f (traj): ", f)
    
    state = {
    "r": r,         # position
    "v": v,         # velocity
    "q": q_d,       # quarternion
    "w": w,         # angular velocity
    "wdot": wdot,   # angular acceleration
    "a": a,         # acceleration
    "j": j,         # jerk
    "s": s,         # snap
    "f": f          # forces
    }

    return state


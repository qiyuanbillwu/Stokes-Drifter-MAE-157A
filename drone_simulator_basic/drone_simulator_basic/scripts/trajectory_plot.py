import numpy as np
import matplotlib.pyplot as plt
from trajectory import get_state, get_state_simple, t0, t1, t2

# get_state(0)
# get_state_simple(0.01)

pos = []
vel = []
acc = []
w = []
wdot = []
forces = []

dt = 0.01

ts = np.arange(t0, t2, dt)

for t in ts:
    state = get_state(t)
    # print(state['f'])
    pos.append(state['r'])
    vel.append(state['v'])
    acc.append(state['a'])
    w.append(state['w'])
    wdot.append(state['wdot'])
    forces.append(state['f'])

pos = np.array(pos)
vel = np.array(vel)
acc = np.array(acc)
w = np.array(w)
wdot = np.array(wdot)
forces = np.array(forces)
# print(ts)
# print(forces[:,1])

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[:,0], pos[:,1], pos[:,2], label='Drone trajectory')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(0,4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.figure(2)
plt.plot(ts, pos[:, 0], label = "x")
plt.plot(ts, pos[:, 1], label = "y", linestyle='--')
plt.plot(ts, pos[:, 2], label = "z", linestyle='--')
plt.legend()
plt.xlabel('t')
plt.ylabel('position')

plt.figure(3)
plt.plot(ts, vel[:, 0], label = "v_x")
plt.plot(ts, vel[:, 1], label = "v_y", linestyle='--')
plt.plot(ts, vel[:, 2], label = "v_z", linestyle='--')
plt.legend()
plt.xlabel('t')
plt.ylabel('velocity')

# plt.figure(4)
# plt.plot(ts, acc[:, 0], label = "a_x")
# plt.plot(ts, acc[:, 1], label = "a_y", linestyle='--')
# plt.plot(ts, acc[:, 2], label = "a_z", linestyle='--')
# plt.legend()

# plt.figure(5)
# plt.plot(ts, w[:, 0], label = "w_x")
# plt.plot(ts, w[:, 1], label = "w_y", linestyle='--')
# plt.plot(ts, w[:, 2], label = "w_z", linestyle='--')
# plt.legend()

# plt.figure(6)
# plt.plot(ts, wdot[:, 0], label = "wdot_x")
# plt.plot(ts, wdot[:, 1], label = "wdot_y", linestyle='--')
# plt.plot(ts, wdot[:, 2], label = "wdot_z", linestyle='--')
# plt.legend()

plt.figure(7)
plt.plot(ts, forces[:, 0], label = 'f1')
plt.plot(ts, forces[:, 1], label = 'f2')
plt.plot(ts, forces[:, 2], label = 'f3')
plt.plot(ts, forces[:, 3], label = 'f4')
plt.legend()
plt.xlabel('t')
plt.ylabel('motor force')
plt.show()

# get_state(0.01)
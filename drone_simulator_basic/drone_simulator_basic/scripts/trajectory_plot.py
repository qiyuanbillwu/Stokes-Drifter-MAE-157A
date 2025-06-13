import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trajectory import get_state, get_state_simple, t0, t1, t2, x1, y1, z1, theta

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

# Rectangle parameters (x-axis aligned)
rect_center = np.array([x1, y1, z1])
width, height = 0.4, 0.7
rotation_angle = theta * 180 / np.pi  # degrees

# Create rectangle vertices (YZ plane)
def create_rotated_rectangle(center, width, height, angle):
    corners = np.array([
        [0, -width/2, -height/2],
        [0, width/2, -height/2],
        [0, width/2, height/2],
        [0, -width/2, height/2]
    ])
    # Rotation matrix (x-axis)
    theta = np.radians(angle)
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    # Apply rotation and translation
    return np.dot(corners, rot.T) + center

# Create and add rectangle
rect_verts = create_rotated_rectangle(rect_center, width, height, rotation_angle)
rect = Poly3DCollection([rect_verts], alpha=0.3, facecolor='green', edgecolor='k')

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15, azim=160)
ax.plot(pos[:,0], pos[:,1], pos[:,2], label='Drone trajectory')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(0,4)
ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_zlabel('z / m')
ax.add_collection3d(rect)

# plt.figure(2)
# plt.plot(ts, pos[:, 0], label = "x")
# plt.plot(ts, pos[:, 1], label = "y", linestyle='--')
# plt.plot(ts, pos[:, 2], label = "z", linestyle='--')
# plt.legend()
# plt.xlabel('t / s')
# plt.ylabel('position / m')

# plt.figure(3)
# plt.plot(ts, vel[:, 0], label = "v_x")
# plt.plot(ts, vel[:, 1], label = "v_y", linestyle='--')
# plt.plot(ts, vel[:, 2], label = "v_z", linestyle='--')
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('velocity')

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
plt.xlabel('t / s')
plt.ylabel('motor force / N')
plt.show()

# get_state(0.01)
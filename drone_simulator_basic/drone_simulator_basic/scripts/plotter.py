### Import python packages ###
import numpy as np
import matplotlib.pyplot as plt
from trajectory import get_state, get_state_simple

# Update with actual file name in the data director
file_name = "data_2025-05-01_10-49-26.csv"

# Load in data as giant matrix
data = np.loadtxt("../data/"+file_name, delimiter=',')


t = data[:, 0]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]
vx = data[:, 4]
vy = data[:, 5]
vz = data[:, 6]

# initialize empty arrays
xd, yd, zd = [], [], []
vxdes, vydes, vzdes = [], [], []

for ti in t:
    traj = get_state(ti)
    xd.append(traj['r'][0])
    yd.append(traj['r'][1])
    zd.append(traj['r'][2])
    vxdes.append(traj['v'][0])
    vydes.append(traj['v'][1])
    vzdes.append(traj['v'][2])

# Convert to numpy arrays (optional, but convenient for plotting)
xd = np.array(xd)
yd = np.array(yd)
zd = np.array(zd)
vxdes = np.array(vxdes)
vydes = np.array(vydes)
vzdes = np.array(vzdes)

# --- Position vs Time ---
plt.figure(1)
plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')
plt.plot(t, xd, label='xd')
plt.plot(t, yd, label='yd')
plt.plot(t, zd, label='zd')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Position vs Time')
plt.legend()
plt.grid()

# --- 3D Drone Trajectory ---

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Drone trajectory')
ax.plot(xd, yd, zd, label = 'Desired trajectory')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Drone Trajectory')
ax.legend()
ax.grid()

# --- Velocity vs Time ---
plt.figure(3)
plt.plot(t, vx, label='vx')
plt.plot(t, vy, label='vy')
plt.plot(t, vz, label='vz')
plt.plot(t, vxdes, label='vxdes')
plt.plot(t, vydes, label='vydes')
plt.plot(t, vzdes, label='vzdes')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity vs Time')
plt.legend()
plt.grid()

print("Final Actual Position:   x = {:.3f}, y = {:.3f}, z = {:.3f}".format(x[-1], y[-1], z[-1]))
print("Final Desired Position:  xd = {:.3f}, yd = {:.3f}, zd = {:.3f}".format(xd[-1], yd[-1], zd[-1]))

plt.show()
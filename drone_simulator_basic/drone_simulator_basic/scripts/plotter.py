### Import python packages ###
import numpy as np
import matplotlib.pyplot as plt
from trajectory import get_state

# Update with actual file name in the data director
file_name = "data_2025-04-28_14-29-03.csv"

# Load in data as giant matrix
data = np.loadtxt("../data/"+file_name, delimiter=',')
print(data)


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
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity vs Time')
plt.legend()
plt.grid()

plt.show()
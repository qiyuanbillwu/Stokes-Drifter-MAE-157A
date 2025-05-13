### Import python packages ###
import numpy as np
import matplotlib.pyplot as plt
from trajectory import get_state, get_state_simple

# Update with actual file name in the data director
file_name = "data/data_2025-05-13_09-45-53.csv"

# Load in data as giant matrix
data = np.loadtxt("../"+file_name, delimiter=',')

## UPDATE, used for equillibrium force calc
m = 0.7437
g = 9.81

t = data[:, 0]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]
vx = data[:, 4]
vy = data[:, 5]
vz = data[:, 6]

qw = data[:, 7]
qx = data[:, 8]
qy = data[:, 9]
qz = data[:, 10]


f1 = data[:, 14]
f2 = data[:, 15]
f3 = data[:, 16]
f4 = data[:, 17]

# initialize empty arrays
xd, yd, zd = [], [], []
vxdes, vydes, vzdes = [], [], []
qwdes,qxdes,qydes,qzdes = [], [], [], []
wxdes, wydes, wzdes = [], [], []

for ti in t:
    # traj = get_state_simple(ti)
    traj = get_state(ti)
    xd.append(traj['r'][0])
    yd.append(traj['r'][1])
    zd.append(traj['r'][2])
    vxdes.append(traj['v'][0])
    vydes.append(traj['v'][1])
    vzdes.append(traj['v'][2])
    qwdes.append(traj['q'][0])
    qxdes.append(traj['q'][1])
    qydes.append(traj['q'][2])
    qzdes.append(traj['q'][3])
    wxdes.append(traj['w'][0])
    wydes.append(traj['w'][1])
    wzdes.append(traj['w'][2])

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
plt.plot(t, xd, label='xd', linestyle='--')
plt.plot(t, yd, label='yd', linestyle='--')
plt.plot(t, zd, label='zd', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Position vs Time')
plt.legend()
plt.grid()
plt.ylim(-2,4)


plt.figure(4)
plt.plot(t, qw, label='qw')
plt.plot(t, qx, label='qx')
plt.plot(t, qy, label='qy')
plt.plot(t, qz, label='qz')
plt.plot(t, qwdes, label='qwd', linestyle='--')
plt.plot(t, qxdes, label='qxd', linestyle='--')
plt.plot(t, qydes, label='qyd', linestyle='--')
plt.plot(t, qzdes, label='qzd', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Quaternion [m]')
plt.title('Quaternion vs Time')
plt.legend()
plt.grid()
plt.ylim(-0.25,0.25)


# -- Motor Forces vs. Time --
plt.figure(101)
plt.plot(t, f1, label="f1 [N]");
plt.plot(t, f2, label="f2 [N]");
plt.plot(t, f3, label="f3 [N]");
plt.plot(t, f4, label="f4 [N]");
plt.title("Motor Forces")
plt.xlabel("Force [N]");
plt.ylabel("Time [s]");

equForce = np.ones(len(t)) * (m*g / 4);
plt.plot(t, equForce, label="Hovering Force [N]");

plt.legend();

# Legend

# --- 3D Drone Trajectory ---
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Drone trajectory')
ax.plot(xd, yd, zd, label = 'Desired trajectory', linestyle='--')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Drone Trajectory')
ax.legend()
ax.grid()
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(0,4)

# --- Velocity vs Time ---
plt.figure(3)
plt.plot(t, vx, label='vx')
plt.plot(t, vy, label='vy')
plt.plot(t, vz, label='vz')
plt.plot(t, vxdes, label='vxdes', linestyle='--')
plt.plot(t, vydes, label='vydes', linestyle='--')
plt.plot(t, vzdes, label='vzdes', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity vs Time')
plt.legend()
plt.grid()


print("Final Actual Position:   x = {:.3f}, y = {:.3f}, z = {:.3f}".format(x[-1], y[-1], z[-1]))
print("Final Desired Position:  xd = {:.3f}, yd = {:.3f}, zd = {:.3f}".format(xd[-1], yd[-1], zd[-1]))

# -- Performance of Outer + Inner Controller --
plt.figure(4)
plt.title("Desired vs Actual Orientation")
#plt.plot(t, qw, color="purple")
plt.plot(t, qx, color="red", label='qx')
plt.plot(t, qy, color="blue", label='qy')
plt.plot(t, qz, color="green", label='qz')

#plt.plot(t, qw_d, color="purple", linestyle="--")
plt.plot(t, qxdes, color="red", linestyle="--", label='qx_des')
plt.plot(t, qydes, color="blue", linestyle="--", label='qy_des')
plt.plot(t, qzdes, color="green", linestyle="--", label='qz_des')

# -- Performance of Outer + Inner Controller --

wx_d_calc = np.zeros(len(t))
wy_d_calc = np.zeros(len(t))
wz_d_calc = np.zeros(len(t))

wx_d_calc[1] = 0;
wy_d_calc[1] = 0;
wz_d_calc[1] = 0;

plt.plot(t, wx_d_calc, color="red", linestyle=":", label = 'wx_d_calc')
plt.plot(t, wy_d_calc, color="blue", linestyle=":", label = 'wx_d_calc')
plt.plot(t, wz_d_calc, color="green", linestyle=":", label = 'wx_d_calc')

plt.show()
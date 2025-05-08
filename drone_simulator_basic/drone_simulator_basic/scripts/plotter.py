### Import python packages ###
import numpy as np
import matplotlib.pyplot as plt
from trajectory import get_state, get_state_simple
from util import angular_velocity_body_wxyz

# Update with actual file name in the data director
file_name = "data/data_2025-05-08_10-48-53.csv"

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

f1 = data[:, 14]
f2 = data[:, 15]
f3 = data[:, 16]
f4 = data[:, 17]

qw_d = data[:, 18]
qx_d = data[:, 19]
qy_d = data[:, 20]
qz_d = data[:, 21]

wx_d = data[:, 22]
wy_d = data[:, 23]
wz_d = data[:, 24]

# initialize empty arrays
xd, yd, zd = [], [], []
vxdes, vydes, vzdes = [], [], []
qwdes,qxdes,qydes,qzdes = [], [], [], []
wxdes, wydes, wzdes = [], [], []

for ti in t:
    traj = get_state_simple(ti)
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
plt.plot(t, x, label='x', color="red")
plt.plot(t, y, label='y', color="blue")
plt.plot(t, z, label='z', color="green")
plt.plot(t, xd, label='xd', linestyle='--', color="red")
plt.plot(t, yd, label='yd', linestyle='--', color="blue")
plt.plot(t, zd, label='zd', linestyle='--', color="green")
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
plt.figure(102)
plt.title("Desired vs Actual Orientation")
#plt.plot(t, qw, color="purple")
plt.plot(t, qx, color="red")
plt.plot(t, qy, color="blue")
plt.plot(t, qz, color="green")

#plt.plot(t, qw_d, color="purple", linestyle="--")
plt.plot(t, qx_d, color="red", linestyle="--")
plt.plot(t, qy_d, color="blue", linestyle="--")
plt.plot(t, qz_d, color="green", linestyle="--")

# -- Performance of Outer + Inner Controller --
plt.figure(103)
plt.title("Desired vs Actual Angular Velocity")
plt.plot(t, wx, color="red")
plt.plot(t, wy, color="blue")
plt.plot(t, wz, color="green")

wx_d_calc = np.zeros(len(t))
wy_d_calc = np.zeros(len(t))
wz_d_calc = np.zeros(len(t))

dt = (t[3] - t[2]);

print(dt)
for i in range(0,len(t)-1):
    qd2 = np.array([qw_d[i], qx_d[i], qy_d[i], qz_d[i]]);
    qd1 = np.array([qw_d[i+1], qx_d[i+1], qy_d[i+1], qz_d[i+1]]);
    wx_d_calc[i+1], wy_d_calc[i+1], wz_d_calc[i+1] = angular_velocity_body_wxyz(qd1, qd2, dt)

wx_d_calc[1] = 0;
wy_d_calc[1] = 0;
wz_d_calc[1] = 0;

plt.plot(t, wx_d_calc, color="red", linestyle=":")
plt.plot(t, wy_d_calc, color="blue", linestyle=":")
plt.plot(t, wz_d_calc, color="green", linestyle=":")

plt.plot(t, wx_d, color="red", linestyle="--")
plt.plot(t, wy_d, color="blue", linestyle="--")
plt.plot(t, wz_d, color="green", linestyle="--")

#plt.figure(104)
#plt.plot(t, wx_d_calc/wx_d);
#plt.plot(t, wy_d_calc/wy_d);
#plt.plot(t, wz_d_calc/wz_d);

plt.show()
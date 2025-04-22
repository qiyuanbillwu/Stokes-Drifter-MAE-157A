# Filename: main.py
# Author: ...
# Created: ...
# Description: Drone simulator

### Import python packages ###
import numpy as np
import datetime

### Import custom modules and classes ###
import dynamics
from rlController import outer_loop_controller, inner_loop_controller

def printStates(t,f,state,n):
    print("t = ", t, " sec")
    print("Forces: (f1,f2,f3,f4) --> (", f[0], ", ", f[1], ", ", f[2], ", ", f[3], ")");
    print("Position [m]:    (px,py,pz)    --> (", state[0], ", ", state[1], ", ", state[2], ")");
    print("Velocity [m/s]:  (vz,vy,vz)    --> (", state[3], ", ", state[4], ", ", state[5], ")");
    print("Orientation:     (qw,qx,qy,qz) --> (", state[6], ", ", state[7], ", ", state[8], ", ", state[9], ")");
    n_rot = n;
    print("Normal Vector:")
    print("Ang Vel [rad/s]: (wx,wy,wz)    --> (", state[10], ", ", state[11], ", ", state[12], ")");

##########################################
############ Drone Simulation ############
##########################################

# Save data flag
save_data = False

# Initial conditions
t = 0.

state = np.zeros(13)
f = np.zeros(4)

# x, y, z
# Zero Position
state[0] = 0.
state[1] = 0.
state[2] = 0.

# vx, vy, vz
# Zero Velocity
state[3] = 0.
state[4] = 0.
state[5] = 0.

# qw, qx, qy, qz
# Initially 'upright' position
state[6] = 1.
state[7] = 0.
state[8] = 0.
state[9] = 0.

n = [1,0,0];

# wx, wy, wz
# Zero Initial Angular Velocity
state[10] = 1.
state[11] = 0.
state[12] = 0.

# Summary of States Array
# state = [posX, posY, posZ, velX, velY, velZ, qw, qx, qy, qz, wx, wy, wz]
# index >>  0     1     2     3     4     5    6   7   8   9   10  11  12

# Final time
tf = 10.

# Simulation rate
rate = 500
dt = 1./rate

# Gravity
g = 9.81

# Other parameters?
#placeholder for now
m = 0.6     # mass of drone [kg]
l = 0.115  # meters [m]
Cd = 0.01   # drag coefficient of propellers [PLACEHOLDER]
Cl = 0.1    # lift coefficent of propellers  [PLACEHOLDER]
Ixx = 0.005; # [kg/m2]
Iyy = 0.005; # [kg/m2]
Izz = 0.002; # [kg/m2]

printStates(t,f,state);
print("")

# Initialize dynamics
dyn = dynamics.dynamics(np.array([g,m,l,Cd,Cl,Ixx,Iyy,Izz]), dt)

# Initialize data array that contains useful info (probably should add more)
data = np.append(t,state)
data = np.append(data,f)

# Propagate dynamics with control inputs
t = t+dt;
state = dyn.propagate(state, f, dt)

printStates(t,f,state);
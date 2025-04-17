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
state[0] = x0
state[1] = y0
state[2] = z0

# vx, vy, vz
state[3] = 0.
state[4] = 0.
state[5] = 0.

# qw, qx, qy, qz
state[6] = 1.
state[7] = 0.
state[8] = 0.
state[9] = 0.

# wx, wy, wz
state[10] = 0.
state[11] = 0.
state[12] = 0.

# Final time
tf = 10.

# Simulation rate
rate = 500
dt = 1./rate

# Gravity
g = 9.8

# Other parameters?
#placeholder for now
m = 10 #mass of drone
l = 0.2  # meters
c = 0.01 # drag coefficient
J = np.diag([0.005, 0.005, 0.009])


# Initialize dynamics
dyn = dynamics.dynamics(np.array([g,m]), dt)

# Initialize data array that contains useful info (probably should add more)
data = np.append(t,state)
data = np.append(data,f)

# Desired position (hover point)
xd, yd, zd = 0.0, 0.0, 1.0  # meters


# Simulation loop
running = True
while running:
    # Get new desired state from trajectory planner
    # xd, yd, zd, ... = get_desired_state(t)

    # Run outer-loop controller to get thrust and references for inner loop 
    # Outer-loop controller
    T, phi_des, theta_des, psi_des = outer_loop_controller(state, xd, yd, zd, mass=m, g=g)

    # Run inner-loop controller to get motor forces 
    # Inner-loop controller
    
    f = inner_loop_controller(state, T, phi_des, theta_des, psi_des, l=l, c=c, J=J, quat_to_rot_func=dyn.quat_to_rot)

    # Propagate dynamics with control inputs
    state = dyn.propagate(state, f, dt)
 
    # If z to low then indicate crash and end simulation
    if state[2] < 0.1:
        print("CRASH!!!")
        break

    # Update data array (this can probably be done in a much cleaner way...)
    tmp = np.append(t,state)
    tmp = np.append(tmp,f)
    data = np.vstack((data,tmp))

    # Update time
    t += dt 

    # If time exceeds final time then stop simulator
    if t >= tf:
        running = False

# If save_data flag is true then save data
if save_data:
    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{date_time_string}.csv"
    np.savetxt("../data/"+file_name, data, delimiter=",")





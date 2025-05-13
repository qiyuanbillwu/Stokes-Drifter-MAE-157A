# Filename: main.py
# Author: ...
# Created: ...
# Description: Drone simulator
#test

### Import python packages ###
import numpy as np
import datetime
import os

print("Current Working Directory:", os.getcwd())

### Import custom modules and classes ###
import dynamics
from rlController import outer_loop_controller, inner_loop_controller
from trajectory import get_state, get_state_simple

##########################################
############ Drone Simulation ############
##########################################


# Save data flag
save_data = True

# Initial conditions
t = 0.0

state = np.zeros(13)
f = np.zeros(4)

# x, y, z
# Zero Position

#assuming start from 1,1,1

initial_state = get_state(0.0)

state[0:3] = initial_state['r']

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

# wx, wy, wz
# Zero Initial Angular Velocity
state[10] = 0.
state[11] = 0.
state[12] = 0.

# Summary of States Array
# state = [posX, posY, posZ, velX, velY, velZ, qw, qx, qy, qz, wx, wy, wz]
# index >>  0     1     2     3     4     5    6   7   8   9   10  11  12

# Final time
tf = 4

# Simulation rate
rate = 500
dt = 1./rate

# Gravity
g = 9.81

# Other parameters?
#placeholder for now
m = 0.7437  # mass of drone [kg]
l = 0.115   # meters [m]
Cd = 0.01   # drag coefficient of propellers [PLACEHOLDER]
Cl = 0.1    # lift coefficent of propellers  [PLACEHOLDER]
J = np.diag([0.00225577, 0.00360365, 0.00181890]) # [kg/m2]

# Initialize dynamics
dyn = dynamics.dynamics([g,m,l,Cd,Cl,J], dt)

# Initialize data array that contains useful info (probably should add more)
data = np.append(t,state)
data = np.append(data,f)

lastVelError = 0
prev_filtered_derivative = 0
# Simulation loop
running = True
while running:
    # Get new desired state from trajectory planner
    # xd, yd, zd, ... = get_desired_state(t)


    # trajectory = get_state_simple(t) # for simple trajectories between 2 points
    trajectory = get_state(t)

    # Run outer-loop controller to get thrust and references for inner loop 
    # Outer-loop controller
    T, q_des, omega_des, lastVelError, prev_filtered_derivative = outer_loop_controller(state, trajectory, m, g, dt, lastVelError, prev_filtered_derivative)

    # Run inner-loop controller to get motor forces 
    # Inner-loop controller
    
    f = inner_loop_controller(state, q_des, omega_des, T, l, dyn.d)

    # Propagate dynamics with control inputs
    #print(state.shape)
    #print(f.shape)
    state = dyn.propagate(state, f, dt)
 
    # If z to low then indicate crash and end simulation
    if state[2] < 0.1:
        print("CRASH!!!")
        break

    # Update data array (this can probably be done in a much cleaner way...)
    tmp = np.append(t,state)
    tmp = np.append(tmp,f)
    #tmp = np.append(tmp,q_des)
    #tmp = np.append(tmp,omega_des)
    data = np.vstack((data,tmp))

    # Update time
    t += dt 

    # If time exceeds final time then stop simulator
    #print(t)
    if t >= tf:
        running = False

# Summary of Output Lines
# state = [ t,  posX,  posY, posZ, velX, velY, velZ, qw, qx, qy, qz, wx, wy, wz], f = [f1, f2, f3, f4]
# index >>  0     1     2     3      4     5     6   7   8   9   10  11  12, 13        14  15  16  17

# If save_data flag is true then save data
if save_data:
    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{date_time_string}.csv"

    file_path = os.path.abspath("../data/" + file_name)
    print("Saving to:", file_path)

    np.savetxt("../data/"+file_name, data, delimiter=",")
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
from util import addNoiseToPercievedState;
from util import quat_to_rot, quat_multiply, quat_conjugate, qdot_from_omega, get_a_dot_hat, allocation_matrix

##########################################
############ Drone Simulation ############
##########################################

from drone_env import DroneEnv  # update path as needed
from stable_baselines3 import TD3

# Initialize your environment (make sure it matches the training env)
env = DroneEnv()

# Load the trained model from file
model = TD3.load("logs/best_model/best_model", env=env)


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
tf = 3

# Simulation rate
rate = 500
dt = 1./rate

# Gravity
g = 9.81

# Import Values from constants.py
from constants import g, m, l, Cd, Cl, d, J

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
    # Desired trajectory at current time
    desired = get_state(t)

    # Calculate errors to build observation (same as in DroneEnv.step)
    goal_pos = desired['r']
    goal_vel = desired['v']

    pos_error = goal_pos - state[0:3]
    vel_error = goal_vel - state[3:6]

    # Run outer-loop controller for references (needed to construct obs)
    T, q_des, omega_des, lastVelError, prev_filtered_derivative = outer_loop_controller(state, desired, m, g, dt, lastVelError, prev_filtered_derivative)

    q_curr = state[6:10]
    omega = state[10:13]

    q_e = quat_multiply(quat_conjugate(q_des), q_curr)
    R_e = quat_to_rot(q_e)
    omega_error = omega - R_e @ omega_des

    obs = np.concatenate([[t], state, pos_error, vel_error, q_e[1:], omega_error])

    # Normalize obs to float32 as agent expects
    obs = obs.astype(np.float32)

    # Query RL agent for normalized motor thrusts [0, 1]
    action, _ = model.predict(obs, deterministic=True)

    # Map normalized thrust to actual force range
    f_min, f_max = 0.054*9.81, 0.393*9.81
    f_agent = f_min + action * (f_max - f_min)

    # Propagate drone dynamics using RL agent's forces
    state = env.dyn.propagate(state, f_agent, dt)

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
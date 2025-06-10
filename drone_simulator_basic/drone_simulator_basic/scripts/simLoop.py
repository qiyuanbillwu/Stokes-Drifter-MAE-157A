### Import python packages ###
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

print("Current Working Directory:", os.getcwd())

### Import custom modules and classes ###
import dynamics
from rlController import outer_loop_controller, inner_loop_controller
from trajectory import get_state, get_state_simple
import matplotlib.animation as animation
from util import addNoiseToPercievedState;

def sim(dt, dt_sensorUpdate, tf, forces, state, dyn, trajFunc, save_data, allow_crash, noiseTuple):
    t = 0.0;
    f = forces;

    # Set Initial Position to Match Trajectory
    trajectory = trajFunc(t)
    x0, y0, z0 = trajectory['r'];
    state[0:3] = [x0,y0,z0]
    print(state)

    # Initialize data array that contains useful info (probably should add more)
    data = np.append(t,state)
    data = np.append(data,f)
    data = np.append(data,[1,0,0,0])
    data = np.append(data,[0,0,0])

    lastVelError = 0
    prev_filtered_derivative = 0

    # positional error, other error
    posErr = noiseTuple[0]; # 0.05 = cm std dev
    otherErr_percentage = noiseTuple[1]; # 15% = 0.15
    lastSensorUpdate = t;
    percievedState = addNoiseToPercievedState(state, posErr, otherErr_percentage)
    dataP = np.append(t,percievedState)

    # Simulation loop
    running = True
    while running:
        # Get new desired state from trajectory planner
        # xd, yd, zd, ... = get_desired_state(t)

        # trajectory = get_state_simple(t) # for simple trajectories between 2 points
        trajectory = trajFunc(t)

        if ((lastSensorUpdate + dt_sensorUpdate) <= t):
            # Update Percieved State
            percievedState = addNoiseToPercievedState(state, posErr, otherErr_percentage);
            lastSensorUpdate = lastSensorUpdate + dt_sensorUpdate;

        # Run outer-loop controller to get thrust and references for inner loop 
        # Outer-loop controller
        T, q_des, omega_des, lastVelError, prev_filtered_derivative = outer_loop_controller(percievedState, trajectory, dyn.m, dyn.g, dt, lastVelError, prev_filtered_derivative)

        # Run inner-loop controller to get motor forces 
        # Inner-loop controller
        f = inner_loop_controller(percievedState, q_des, omega_des, T, dyn.l, dyn.d)

        # Propagate dynamics with control inputs
        #print(state.shape)
        #print(f.shape)
        state = dyn.propagate(state, f, dt)
    
        # If z to low then indicate crash and end simulation
        if (state[2] <= 0) and (allow_crash):
            print("CRASH!!!")
            break

        # Update data array (this can probably be done in a much cleaner way...)
        tmp = np.append(t,state)
        tmp = np.append(tmp,f)
        tmp = np.append(tmp,q_des)
        tmp = np.append(tmp,omega_des)
        data = np.vstack((data,tmp))

        tmp = np.append(t,percievedState)
        dataP = np.vstack((dataP, tmp))

        # Update time
        t += dt 

        # If time exceeds final time then stop simulator
        #print(t)
        if t >= tf:
            running = False

        # Summary of Output Lines
        # state = [ t,  posX,  posY, posZ, velX, velY, velZ, qw, qx, qy, qz, wx, wy, wz], f = [f1, f2, f3, f4], q_des = [qw_des]
        # index >>  0     1     2     3      4     5     6   7   8   9   10  11  12, 13        14  15  16  17              18

        # If save_data flag is true then save data
        if save_data:
            now = datetime.datetime.now()
            date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"data_{date_time_string}.csv"

            file_path = os.path.abspath("../data/" + file_name)
            print("Saving to:", file_path)

            np.savetxt("../data/"+file_name, data, delimiter=",")

    return data, dataP;

def cleanSlate():
    t0 = 0;
    f0 = [0,0,0,0];
    s0 = np.zeros(13);
    s0[0] = 1;
    s0[1] = 1;
    s0[2] = 1;
    s0[6] = 1;

    return t0, f0, s0;
# Filename: testCases_innerLoop.py
# Author: Jonathan
# Created: 5/6/2025

### Import python packages ###
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

print("Current Working Directory:", os.getcwd())

### Import custom modules and classes ###
import util
import dynamics
from rlController import outer_loop_controller, inner_loop_controller
from trajectory import get_state, get_state_simple

###################################################
############ Drone Simulation Function ############
###################################################

def cleanSlate():
    t0 = 0;
    f0 = [0,0,0,0];
    s0 = np.zeros(13);
    s0[0] = 1;
    s0[1] = 1;
    s0[2] = 1;
    s0[6] = 1;

    return t0, f0, s0;

def sim(dt, tf, forces, state, dyn, innerFunc,save_data):
    t = 0.0;
    f = forces;

    # Initialize data array that contains useful info (probably should add more)
    data = np.append(t,state)
    data = np.append(data,f)
    data = np.append(data,[1,0,0,0])
    data = np.append(data,[0,0,0])

    lastVelError = 0
    # Simulation loop
    running = True
    while running:
        # Get new desired state from trajectory planner
        # xd, yd, zd, ... = get_desired_state(t)
        #trajectory = get_state_simple(t)

        # Run outer-loop controller to get thrust and references for inner loop 
        # Outer-loop controller
        #T, q_des, omega_des, lastVelError = outer_loop_controller(state, trajectory, m, g, dt, lastVelError)
        T, q_des, omega_des = innerFunc(t);

        # Run inner-loop controller to get motor forces 
        # Inner-loop controller
        f = inner_loop_controller(state, q_des, omega_des, T, l, dyn.d)

        # Propagate dynamics with control inputs
        state = dyn.propagate(state, f, dt)
    
        # If z to low then indicate crash and end simulation
        #if state[2] < 0.1:
        #    print("CRASH!!!")
        #    break

        # Update data array (this can probably be done in a much cleaner way...)
        tmp = np.append(t,state)
        tmp = np.append(tmp,f)
        tmp = np.append(tmp,q_des)
        tmp = np.append(tmp,omega_des)

        #print(tmp)
        data = np.vstack((data,tmp))

        # Update time
        t += dt 

        # If time exceeds final time then stop simulator
        print(t)
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

    return data;


def plotQuat(dataOut):
    tArr = dataOut[:,0];
    qwArr = dataOut[:,7];
    qxArr = dataOut[:,8];
    qyArr = dataOut[:,9];
    qzArr = dataOut[:,10];
    plt.plot(tArr, qwArr, label="qx", color = "purple")
    plt.plot(tArr, qxArr, label="qx", color = "red")
    plt.plot(tArr, qyArr, label="qy", color = "blue")
    plt.plot(tArr, qzArr, label="qz", color = "green")
    plt.plot(tArr, dataOut[:,18], label="qw_des", linestyle="--", color = "purple")
    plt.plot(tArr, dataOut[:,19], label="qx_des", linestyle="--", color = "red")
    plt.plot(tArr, dataOut[:,20], label="qy_des", linestyle="--", color = "blue")
    plt.plot(tArr, dataOut[:,21], label="qz_des", linestyle="--", color = "green")
    plt.legend()
    plt.show()

def plotW(dataOut):
    tArr = dataOut[:,0];
    wxArr = dataOut[:,11];
    wyArr = dataOut[:,12];
    wzArr = dataOut[:,13];
    plt.plot(tArr, wxArr, label="wx")
    plt.plot(tArr, wyArr, label="wy")
    plt.plot(tArr, wzArr, label="wz")
    plt.plot(tArr, dataOut[:,22], label="omegaX_des", linestyle="--")
    plt.plot(tArr, dataOut[:,23], label="omegaY_des", linestyle="--")
    plt.plot(tArr, dataOut[:,24], label="omegaZ_des", linestyle="--")
    plt.legend()
    plt.show()

t0, f0, s0 = cleanSlate()

rate = 500;
dt = 1.0/rate;

# DYNAMICS CLASS
g = 9.81
m = 0.7437  # mass of drone [kg]
l = 0.115   # meters [m]
Cd = 0.01   # drag coefficient of propellers [PLACEHOLDER]
Cl = 0.1    # lift coefficent of propellers  [PLACEHOLDER]
J = np.diag([0.00225577, 0.00360365, 0.00181890]) # [kg/m2]

# Initialize dynamics
dyn = dynamics.dynamics([g,m,l,Cd,Cl,J], dt)

# Test Cases
def innerFunc_case1(t):
    # Step Response
    if t < 1:
        T = 10
        q_des = [1,0,0,0];
        omega_des = [0,0,0];
    if t > 1:
        T = 10;
        q_des = [0.924,0.383,0,0];
        omega_des = [0,0,0];

    return T, q_des, omega_des;

#tf = 10.0;
#dataOut = sim(dt, tf, f0, s0, dyn, innerFunc_case1, False);
#plotQuat(dataOut)
#plotW(dataOut)

# Passed w/ T = 0
# Passed w/ T = 10
# Seems to not work when omega_des is non-zero, but this could be me over constraining the system, lets try a real trajectory

# Test Cases
def innerFunc_case2(t):
    # Linear Ramp, starting at 0,0,0 and going to some rollF, yawF, pitchY
    rF = np.deg2rad(-45);
    yF = np.deg2rad(45);
    pF = np.deg2rad(45);
    tf = 3.0;
    
    wx = rF / tf;
    #wx = 0;
    wy = yF / tf;
    #wy = 0;
    wz = pF / tf;
    #wz = 0;

    omega_des = [wx, wy, wz];
    T = 0;

    if (t > tf): 
        omega_des = [0, 0, 0];
        t = tf;

    roll = wx * t;
    pitch = wy * t;
    yaw = wz * t;

    q_des = util.euler_to_quaternion(roll, pitch, yaw)

    return T, q_des, omega_des;

tf = 10.0;
dataOut = sim(dt, tf, f0, s0, dyn, innerFunc_case2, False);
plotQuat(dataOut)
plotW(dataOut)
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
import dynamics
from rlController import outer_loop_controller, inner_loop_controller
from trajectory import get_state, get_state_simple
import matplotlib.animation as animation
from util import addNoiseToPercievedState;

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

def sim(dt, tf, forces, state, dyn, trajFunc, save_data, allow_crash, noiseTuple):
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
        T, q_des, omega_des, lastVelError, prev_filtered_derivative = outer_loop_controller(percievedState, trajectory, m, g, dt, lastVelError, prev_filtered_derivative)

        # Run inner-loop controller to get motor forces 
        # Inner-loop controller
        f = inner_loop_controller(percievedState, q_des, omega_des, T, l, dyn.d)

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

# Plot Orientation
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
    plt.title("Orientation Tracking")
    plt.legend()
    plt.show()

# Plot Angular Velocities
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

def plotPos(dataOut, trajFunc, titleStr = "Position vs Time"):
    # --- Position vs Time ---
    t = dataOut[:,0];
    x = dataOut[:,1];
    y = dataOut[:,2];
    z = dataOut[:,3];

    xd, yd, zd = [], [], []
    vxdes, vydes, vzdes = [], [], []
    qwdes,qxdes,qydes,qzdes = [], [], [], []
    wxdes, wydes, wzdes = [], [], []

    for ti in t:
        # traj = get_state_simple(ti)
        traj = trajFunc(ti)
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

    plt.figure(1)
    plt.plot(t, x, label='x',color='red')
    plt.plot(t, y, label='y',color="blue")
    plt.plot(t, z, label='z',color="green")
    plt.plot(t, xd, label='xd', linestyle='--',color='red')
    plt.plot(t, yd, label='yd', linestyle='--',color="blue")
    plt.plot(t, zd, label='zd', linestyle='--',color="green")
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title(titleStr)
    plt.legend()
    plt.grid()
    plt.ylim(-2,4)
    plt.show()

def plotPerc(data, dataP):
    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]
    xP = dataP[:,1]
    yP = dataP[:,2]
    zP = dataP[:,3]
    plt.plot(t,x,linestyle='-',color='red', label="x")
    plt.plot(t,xP,linestyle='--',color='red')
    plt.plot(t,y,linestyle='-',color='blue',label="y")
    plt.plot(t,yP,linestyle='--',color='blue')
    plt.plot(t,z,linestyle='-',color='green',label="z")
    plt.plot(t,zP,linestyle='--',color='green')
    plt.title("Percieved Position vs. Actual Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Pos [m]")
    plt.legend()
    plt.show()

def animateDronePath(dataOut, trajFunc):
    # Shows a 3D quiver plot with the normal vector of the drone
    t = dataOut[:,0];
    x = dataOut[:,1];
    y = dataOut[:,2];
    z = dataOut[:,3];

    qw = dataOut[:,7];
    qx = dataOut[:,8];
    qy = dataOut[:,9];
    qz = dataOut[:,10];

    xd, yd, zd = [], [], []
    for ti in t:
        traj = trajFunc(ti)
        xd.append(traj['r'][0])
        yd.append(traj['r'][1])
        zd.append(traj['r'][2])

    zDrone = [0,0,1]

    def update(frame):
        # for each frame, update the data stored on each artist.
        x = t[:frame]
        y = z[:frame]
        # update the scatter plot:
        data = np.stack([x, y]).T
        scat.set_offsets(data)
        # update the line plot:
        line2.set_xdata(t[:frame])
        line2.set_ydata(z2[:frame])
        return (scat, line2)
    
    fig, ax = plt.subplots()
    
    trajDES = ax.plot(xd, yd, zd, c="b", s=5, label='Desired Trajectory')
    dronePOSE = ax.plot(x[0], y[0], z[0], label='Drone POSE')[0]
    ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
    ax.legend()

    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
    plt.show()

## WHERE THE STUFF HAPPENS ========================================================================================
### SET UP INITIAL STATES & DRONE
t0, f0, s0 = cleanSlate()

rate = 1000;
dt = 1.0/rate;

rate_sensorUpdate = 100;
dt_sensorUpdate = 1.0/rate_sensorUpdate;

if (rate_sensorUpdate > rate):
    dt = dt_sensorUpdate

noiseTuple = (0.1,0.5)

# Import Values from constants.py
from constants import g, m, l, Cd, Cl, d, J

# Initialize dynamics
droneDyn = dynamics.dynamics([g,m,l,Cd,Cl,J], dt)

titleStr = f"Position vs Time | Noise: {noiseTuple[0]*100}cm POS, {noiseTuple[1]*100}% IMU | {rate_sensorUpdate} Hz Sensor Refresh"
trajFunc = get_state
dataOut, dataP = sim(dt, 3.0, f0, s0, droneDyn, trajFunc, False, True, noiseTuple)
plotPos(dataOut, trajFunc, titleStr)
plotQuat(dataOut)

plotPerc(dataOut,dataP)
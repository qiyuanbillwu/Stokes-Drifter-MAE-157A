# Author: Jonathan
# Created: 6/10/2025

### Import python packages ###
import dynamics
from trajectory import get_state
from simLoop import sim, cleanSlate
from plotter import plotPos, plotQuat, plotPerceivedPosition

# Define Simulation Parameters
rate = 1000; 
dt = 1.0/rate;

rate_sensorUpdate = 100; 
dt_sensorUpdate = 1.0/rate_sensorUpdate;

tf = 3.0;

if (rate_sensorUpdate > rate):
    dt = dt_sensorUpdate

noiseTuple = (0.1,0.1)
trajFunc = get_state

# Initial state, all zeros
t0, f0, s0 = cleanSlate()

# Initialize Dynamics
from constants import g, m, l, Cd, Cl, d, J
droneDyn = dynamics.dynamics([g,m,l,Cd,Cl,J], dt)

# Run Simulation
dataOut, dataP = sim(dt, dt_sensorUpdate, tf, f0, s0, droneDyn, trajFunc, False, True, noiseTuple)

# Analyze Data
titleStr = f"Position vs Time | Noise: {noiseTuple[0]*100}cm POS, {noiseTuple[1]*100}% IMU | {rate_sensorUpdate} Hz Sensor Refresh"
plotPos(dataOut, trajFunc, titleStr)
plotQuat(dataOut)
plotPerceivedPosition(dataOut,dataP)
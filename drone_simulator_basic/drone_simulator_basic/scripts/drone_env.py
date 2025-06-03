import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dynamics import dynamics
from constants import g, m, l, Cd, Cl, d, J
from trajectory import get_state
from util import addNoiseToPercievedState
from rlController import outer_loop_controller
from rlController import inner_loop_controller

lower_bound = 0.054*9.81
upper_bound = 0.393*9.81

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 1./500
        self.dyn = dynamics([g, m, l, Cd, Cl, J], self.dt)
        self.max_time = 3.0
        self.t = 0.0
        state = np.zeros(13)
        state[0:3] = get_state(0.0)['r']
        state[6] = 1.0  # initial quaternion
        self.state = state

        # Action: 4 motor thrusts normalized
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Observation:
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.t = 0.0
        state = np.zeros(13)
        state[0:3] = get_state(0.0)['r']
        state[6] = 1.0  # initial quaternion

        self.state = state
        self.traj = get_state  # You can later randomize target
        self.lastVelError = 0
        self.prev_filtered_derivative = 0
        goal_pos = self.traj(self.t)['r']
        goal_vel = self.traj(self.t)['v']

        pos_error = goal_pos - self.state[0:3]
        vel_error = goal_vel - self.state[3:6]
        obs = np.concatenate([[self.t], self.state, pos_error, vel_error])

        return obs, {}
    
    def step(self, action):
        # Map action [0, 1] → actual motor force range
        f_min, f_max = lower_bound, upper_bound
        f_agent = f_min + action * (f_max - f_min)

        # Get desired state
        desired = self.traj(self.t)

        # Outer-loop controller gives desired attitude & angular rate
        T, q_des, omega_des, self.lastVelError, self.prev_filtered_derivative = outer_loop_controller(
            self.state, desired, m, g, self.dt, self.lastVelError, self.prev_filtered_derivative
        )

        # Inner-loop controller gives the ground truth motor forces
        f_true = inner_loop_controller(self.state, q_des, omega_des, T, l, d)

        # Simulate dynamics using agent's thrusts
        self.state = self.dyn.propagate(self.state, f_agent, self.dt)
        self.t += self.dt

        #print(f"f_agent: {f_agent}, f_true: {f_true}")

        goal_pos = desired['r']
        goal_vel = desired['v']

        pos_error = goal_pos - self.state[0:3]
        vel_error = goal_vel - self.state[3:6]
        obs = np.concatenate([[self.t], self.state, pos_error, vel_error])

        # Reward is negative MSE between agent and true thrusts
        abs_error = np.mean(np.abs(f_agent - f_true))
        
        reward = 1 - (1.0 * np.linalg.norm(pos_error) + 0.1 * np.linalg.norm(vel_error) + 0.5 * abs_error)

        terminated = self.state[2] < 0.1
        truncated = self.t >= self.max_time

        if truncated and not terminated:
            reward += 1000.0 

        if terminated:
            reward -= 1000.0

        return obs, reward, terminated, truncated, {}

    def render(self):
        #print(self.state)
        pass  # Optional: visualizations

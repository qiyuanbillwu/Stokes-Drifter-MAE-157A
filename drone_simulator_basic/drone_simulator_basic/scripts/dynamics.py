# Filename: dynamics.py
# Author: ...
# Created: ...
# Description: Dynamics for drone simulator

import numpy as np

class dynamics: 
	def __init__(self, params, dt):
		# Initialize Params (Need to add more!)
		self.g = params[0]
		self.m = params[1]
		# self.J = ... Inertia Tensor
		# self.l = ... Moment Arm
		# self.c = ... Propeller Drag Coefficient

	# This is meant to give the rates of each state
	def rates(self, state, f):
		# Get rotation matrix from current quaterion
		R = self.quat_to_rot([state[6], state[7], state[8], state[9]])

		# Get thrust from motor forces f
		T = f[0] + f[1] + f[2] + f[3]
		
		# Velocities
		dx = state[3]
		dy = state[4]
		dz = state[5]

		# Accelerations
		dvx = R[0,2] * T  / self.m
		dvy = R[1,2] * T  / self.m
		dxz = R[2,2] * T  / self.m - self.g

		# Orientation
		# dqw = ...
		# dqx = ...
		# dqy = ...
		# dqz = ...

		# Angular Velocities
		Jinv = np.linalg.inv(J)
		# dwx = Jinv * ...
		# dwy = Jinv * ...
		# dwz = Jinv * ...

		res = np.array([dx, dy, dz, dvx, dvy, dvz, dqw, dqx, dqy, dqz, dwx, dwy, dwz])

		return res

	# Numerical integration scheme (can do better than Euler!)
	def propagate(self, state, f, dt):
		state += dt * self.rates(state, f)
		return state

	# Helper function that converts a quaternion to rotation matrix
	def quat_to_rot(q):
		# R = ...
		return R


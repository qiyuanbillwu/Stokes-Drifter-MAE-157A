# Filename: dynamics.py
# Author: ...
# Created: ...
# Description: Dynamics for drone simulator

import numpy as np
from util import quat_to_rot, rot_to_quat, cross_matrix, allocation_matrix, quat_multiply

class dynamics: 
	def __init__(self, params, dt):
		# Initialize Params (Need to add more!)
		self.g = params[0];
		self.m = params[1];
		self.l = params[2];
		self.Cd = params[3]; # Propeller Drag Coefficient
		self.Cl = params[4]; # Propeller Lift Coefficient
		self.J = params[5]; 
	
		self.d = self.Cd / self.Cl; # Ratio of drag coefficient to lift / thrust coefficient 

	# This is meant to give the rates of each state
	def rates(self, state, f):
		# Get rotation matrix from current quaterion
		q = [state[6], state[7], state[8], state[9]];
		R = quat_to_rot(q);
		w = np.array([state[10], state[11], state[12]]);

		# Get thrust from motor forces f
		A = allocation_matrix(self.l,self.d);
		[T,tauX,tauY,tauZ] = np.matmul(A,f);
		# print(T,tauX,tauY,tauZ);
		
		# Velocities
		dx = state[3]
		dy = state[4]
		dz = state[5]

		# Accelerations
		dvx = R[0,2] * T  / self.m
		dvy = R[1,2] * T  / self.m
		dvz = R[2,2] * T  / self.m - self.g
		# print("Rotation Matrix = ", R)
		#print("Thrust = ", T)
		#print("Mass = ", self.m)

		# Orientation
		# Rdot = np.matmul(R, cross_matrix(w));
		# dq = rot_to_quat(Rdot);
		# print("dq: ", dq)
		dq = 0.5 * quat_multiply(q, [0, w[0], w[1], w[2]])
		# print(0.5 * quat_multiply(q, [0, w[0], w[1], w[2]]))
		
		dqw = dq[0];
		dqx = dq[1];
		dqy = dq[2];
		dqz = dq[3];

		# Angular Velocities
		Jinv = np.linalg.inv(self.J)
		w = np.array(w)
		dw = np.matmul(Jinv, (np.cross(-w,np.matmul(self.J,w)) + [tauX, tauY, tauZ]));
		dwx = dw[0];
		dwy = dw[1];
		dwz = dw[2];

		res = np.array([dx, dy, dz, dvx, dvy, dvz, dqw, dqx, dqy, dqz, dwx, dwy, dwz]);
		return res

	# Numerical integration scheme (can do better than Euler!)
	def propagate(self, state, f, dt):
		state += dt * self.rates(state, f)
		return state
	
	def propagateRK4(self, state, f, dt):
		"""
		RK4 integration step for a system defined by self.rates(state, f)
		
		Parameters:
			state : np.array
				The current state of the system
			f : external input or forcing function
			dt : float
				Time step

		Returns:
			np.array : The updated state after one RK4 step
		"""
		k1 = self.rates(state, f)
		k2 = self.rates(state + 0.5 * dt * k1, f)
		k3 = self.rates(state + 0.5 * dt * k2, f)
		k4 = self.rates(state + dt * k3, f)

		state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
		return state


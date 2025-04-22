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
		w = [state[10], state[11], state[12]];

		# Get thrust from motor forces f
		T = f[0] + f[1] + f[2] + f[3];

		AllocationMatrix = np.array([
			
		])
		torque = np.matmul(AllocationMatrix, f);
		
		# Velocities
		dx = state[3]
		dy = state[4]
		dz = state[5]

		# Accelerations
		dvx = R[0,2] * T  / self.m
		dvy = R[1,2] * T  / self.m
		dvz = R[2,2] * T  / self.m - self.g

		# Orientation
		Rdot = np.matmul(R, self.cross_matrix(w));
		dq = self.rot_to_quat(Rdot);
		dqw = dq[0];
		dqx = dq[1];
		dqy = dq[2];
		dqz = dq[3];

		# Angular Velocities
		Jinv = np.linalg.inv(J)
		dw = np.matmul(Jinv, (np.cross(-w,np.matmul(J,w)) + torque));
		# dwx = Jinv * ...
		# dwy = Jinv * ...
		# dwz = Jinv * ...

		res = np.array([dx, dy, dz, dvx, dvy, dvz, dqw, dqx, dqy, dqz, dwx, dwy, dwz])

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

	# Helper function that converts a quaternion to rotation matrix
	# https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
	def quat_to_rot(q):
		"""
		Converts a quaternion (qw, qx, qy, qz) into a 3x3 rotation matrix.
		"""
		qw, qx, qy, qz = q

		R = np.zeros((3, 3))

		R[0, 0] = 1 - 2*(qy**2 + qz**2)
		R[0, 1] = 2*(qx*qy - qz*qw)
		R[0, 2] = 2*(qx*qz + qy*qw)

		R[1, 0] = 2*(qx*qy + qz*qw)
		R[1, 1] = 1 - 2*(qx**2 + qz**2)
		R[1, 2] = 2*(qy*qz - qx*qw)

		R[2, 0] = 2*(qx*qz - qy*qw)
		R[2, 1] = 2*(qy*qz + qx*qw)
		R[2, 2] = 1 - 2*(qx**2 + qy**2)

		return R

	def rot_to_quat(R):
		"""
		Convert a 3x3 rotation matrix to a unit quaternion (qw, qx, qy, qz).

		Parameters:
			R : np.ndarray
				3x3 rotation matrix

		Returns:
			np.ndarray : Quaternion as [qw, qx, qy, qz]
		"""
		trace = np.trace(R)

		if trace > 0:
			S = 2.0 * np.sqrt(trace + 1.0)
			qw = 0.25 * S
			qx = (R[2, 1] - R[1, 2]) / S
			qy = (R[0, 2] - R[2, 0]) / S
			qz = (R[1, 0] - R[0, 1]) / S
		else:
			# Find the largest diagonal element and choose case accordingly
			if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
				S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
				qw = (R[2, 1] - R[1, 2]) / S
				qx = 0.25 * S
				qy = (R[0, 1] + R[1, 0]) / S
				qz = (R[0, 2] + R[2, 0]) / S
			elif R[1, 1] > R[2, 2]:
				S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
				qw = (R[0, 2] - R[2, 0]) / S
				qx = (R[0, 1] + R[1, 0]) / S
				qy = 0.25 * S
				qz = (R[1, 2] + R[2, 1]) / S
			else:
				S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
				qw = (R[1, 0] - R[0, 1]) / S
				qx = (R[0, 2] + R[2, 0]) / S
				qy = (R[1, 2] + R[2, 1]) / S
				qz = 0.25 * S

		quat = np.array([qw, qx, qy, qz])
		return quat / np.linalg.norm(quat)  # Ensure unit quaternion
	
	def cross_matrix(v):
		"""
		Returns the skew-symmetric cross-product matrix of a 3D vector v.
		
		Parameters:
			v : np.ndarray or list-like of shape (3,)
				A 3D vector [vx, vy, vz]
		
		Returns:
			np.ndarray : 3x3 skew-symmetric matrix such that cross(a, v) == cross_matrix(v) @ a
		"""
		vx, vy, vz = v
		return np.array([
			[ 0,   -vz,  vy],
			[ vz,   0,  -vx],
			[-vy,  vx,   0 ]
		])
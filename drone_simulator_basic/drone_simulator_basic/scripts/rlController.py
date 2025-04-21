
import numpy as np

#get thrust and desired orientation
def outer_loop_controller(state, trajectory, mass, g):
    # Extract current state
    pos = state[0:3]
    vel = state[3:6]

    # Extract desired position and velocity from trajectory at current time step
    xd, yd, zd = trajectory['position']
    xdot, ydot, zdot = trajectory['velocity']

    # Position and velocity errors
    e_pos = np.array([xd, yd, zd]) - pos
    e_vel = np.array([xdot, ydot, zdot]) - vel

    # PID gains for position control
    Kp_pos = np.array([1.5, 1.5, 10.0])  # Adjust these gains as necessary
    Kd_pos = np.array([1.0, 1.0, 5.0])  # Adjust these gains as necessary

    # Desired acceleration
    a_des = Kp_pos * e_pos + Kd_pos * e_vel + np.array([0, 0, g])   

    # Compute the desired thrust (along the z-axis)
    T = mass * a_des[2]

    # Desired roll and pitch (based on desired horizontal accelerations)
    phi_des = -a_des[1] / g
    theta_des = a_des[0] / g
    psi_des = 0.0  # You can adjust the yaw if necessary (for an angled gate, if needed)

    return T, phi_des, theta_des, psi_des

def inner_loop_controller(state, T, phi_des, theta_des, psi_des, l, c, J, quat_to_rot_func):
    # Extract current orientation and angular velocity
    qw, qx, qy, qz = state[6:10]
    omega = state[10:13]

    # Convert quaternion to rotation matrix
    R = quat_to_rot_func([qw, qx, qy, qz])

    # Extract current Euler angles (phi, theta, psi)
    phi = np.arctan2(R[2, 1], R[2, 2])
    theta = -np.arcsin(R[2, 0])
    psi = np.arctan2(R[1, 0], R[0, 0])

    # Orientation and angular velocity errors
    e_ang = np.array([phi_des - phi, theta_des - theta, psi_des - psi])
    e_omega = -omega

    # PD gains for angular control
    Kp_ang = np.array([8.0, 8.0, 3.0])
    Kd_ang = np.array([1.5, 1.5, 0.8])

    # Compute the desired torques
    tau = Kp_ang * e_ang + Kd_ang * e_omega

    # Mixing matrix to convert thrust and torques to motor forces
    mix = np.array([
        [1, 1, 1, 1],  # Total thrust
        [l, -l, -l, l],  # Roll
        [-l, -l, l, l],  # Pitch
        [c, -c, c, -c]  # Yaw
    ])

    # Full control vector (thrust + torques)
    tau_full = np.array([T, *tau])

    # Solve for motor forces (f1, f2, f3, f4)
    f = np.linalg.solve(mix, tau_full)

    return f

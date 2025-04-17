
import numpy as np

#get thrust and desired orientation
def outer_loop_controller(state, xd, yd, zd, mass, g):

    # Simple PID gains for position controller
    Kp_pos = np.array([1.5, 1.5, 10.0])
    Kd_pos = np.array([1.0, 1.0, 5.0])

    # Position and velocity
    pos = state[0:3]
    vel = state[3:6]

    # Position and velocity errors
    e_pos = np.array([xd, yd, zd]) - pos
    e_vel = -vel

    # Desired acceleration + gravity compensation
    a_des = Kp_pos * e_pos + Kd_pos * e_vel + np.array([0, 0, g])
            
    # Desired total thrust (along body z-axis)
    T = mass * a_des[2]

    # Desired roll and pitch from desired horizontal accelerations
    phi_des = -a_des[1] / g
    theta_des = a_des[0] / g
    psi_des = 0.0  # fixed yaw for now

    return T, phi_des, theta_des, psi_des

def inner_loop_controller(state, T, phi_des, theta_des, psi_des, l, c, J, quat_to_rot_func):
    # Current orientation and angular velocity
    qw, qx, qy, qz = state[6:10]
    omega = state[10:13]

    # Rotation matrix from quaternion
    R = quat_to_rot_func([qw, qx, qy, qz])

    # Extract Euler angles
    phi = np.arctan2(R[2,1], R[2,2])
    theta = -np.arcsin(R[2,0])
    psi = np.arctan2(R[1,0], R[0,0])

    # Orientation error and angular velocity error
    e_ang = np.array([phi_des - phi, theta_des - theta, psi_des - psi])
    e_omega = -omega

    # PD gains
    Kp_ang = np.array([8.0, 8.0, 3.0])
    Kd_ang = np.array([1.5, 1.5, 0.8])

    # Desired torques
    tau = Kp_ang * e_ang + Kd_ang * e_omega

    # Mixing matrix to convert [T, τx, τy, τz] to motor forces
    mix = np.array([
        [1,  1,  1,  1],                    # total thrust
        [ l, -l, -l,  l],                   # roll
        [-l, -l,  l,  l],                   # pitch
        [ c, -c,  c, -c]                    # yaw
    ])

    tau_full = np.array([T, *tau])
    f = np.linalg.solve(mix, tau_full)
    f = np.clip(f, 0, np.inf)  # Ensure all motor thrusts are positive

    return f
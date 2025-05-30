
import numpy as np
from util import quat_to_rot, quat_multiply, quat_conjugate, qdot_from_omega, get_a_dot_hat, allocation_matrix
#must always account for double covering with quaternions to prevent unwinding

#qy is not correct for some reason

#get thrust and desired orientation
def outer_loop_controller(state, trajectory, mass, g, dt, lastVelError, prev_filtered_derivative):
    # Extract current state
    pos = state[0:3]
    vel = state[3:6]

    # Extract desired position and velocity from trajectory at current time step
    xd, yd, zd = trajectory['r']
    vxdes, vydes, vzdes = trajectory['v']
    axdes, aydes, azdes = trajectory['a']

    # Position and velocity errors
    #e_pos = np.array([xd, yd, zd]) - pos
    #e_vel = np.array([vxdes, vydes, vzdes]) - vel
    e_pos = pos - np.array([xd, yd, zd]) 
    e_vel = vel - np.array([vxdes, vydes, vzdes]) 

    # PID gains for position control
    Kp = np.array([1, 0.4, 10])  # Adjust these gains as necessary
    Kd = np.array([10, 10, 10])  # Adjust these gains as necessary

    accel_des = np.array([axdes, aydes, azdes])

    # Desired acceleration
    a = accel_des - Kp * e_pos - Kd * e_vel + np.array([0, 0, g])   

    # Compute the desired thrust (along the z-axis)
    T = mass * np.linalg.norm(a)

    a_hat = a / (np.linalg.norm(a))
    e3_hat = np.array([0.0, 0.0, 1.0])

    cross_part = np.cross(e3_hat, a_hat) 
    cross_part = (1 / (np.sqrt(2*(1 + e3_hat.T @ a_hat)))) * cross_part

    first_part = (1 / (np.sqrt(2*(1 + e3_hat.T @ a_hat)))) * (1 + e3_hat.T @ a_hat)

    q_des = np.array([first_part, cross_part[0], cross_part[1], cross_part[2]])

    # print("q_des: ", q_des)

    R_d = quat_to_rot(q_des)

    # =============
    # Acceleration Tracking
    raw_derivative = (e_vel - lastVelError) / dt # Error in acceleration over a single timestep

    # Apply low-pass filter to derivative only
    alpha = 0.2
    filtered_derivative = alpha * prev_filtered_derivative + (1 - alpha) * raw_derivative

    prev_filtered_derivative = filtered_derivative

    a_dot = trajectory['j'] - Kp * e_vel - Kd * filtered_derivative

    lastVelError = e_vel

    adot_hat = get_a_dot_hat(a, a_dot)

    # omega is in the form of (wy, -wx, 0)
    # want omega_des in the form (wx, wy, 0)
    omega = R_d.T @ adot_hat
    omega_des = omega
    omega_des[0] = -omega[1]
    omega_des[1] = omega[0]

    # omega_des = np.cross(a_hat, adot_hat)
    omega_des[2] = 0  # if yaw is not tracked

    return T, q_des, omega_des, lastVelError, prev_filtered_derivative

#prob correct
def inner_loop_controller(state, q_des, omega_des, T, l, d):
    # Extract current quaternion and angular velocity
    q_curr = state[6:10]       # [qw, qx, qy, qz]
    omega = state[10:13]       # [wx, wy, wz]

    # ============================
    # Orientation error quaternion: q_e = q_des^* ⊗ q_curr
    q_e = quat_multiply(quat_conjugate(q_des), q_curr)

    # PD gains
    Kp_vec = np.array([10.0, 10.0, 10.0])
    Kd_vec = np.array([0.2, 0.2, 0.2])

    Kp = np.diag(Kp_vec)
    Kd = np.diag(Kd_vec) 

    Lambda = np.array([0.2, 0.2, 0.2])


    # Sign correction to avoid unwinding
    s = np.sign(q_e[0]) if q_e[0] != 0 else 1

    # Rotation matrix from desired quaternion (for omega_d transformation)
    R_e = quat_to_rot(q_e)
    
    # Angular velocity error
    omega_e = omega - R_e @ omega_des
    q_dot_e = qdot_from_omega(q_des, omega_e)

    # Control torque
    tau = -s * Kp @ q_e[1:] - Kd @ omega_e - Lambda * s * q_dot_e[1:]

    # Mixer matrix to solve for motor forces
    mix = allocation_matrix(l,d)

    # Combine total thrust and torques
    tau_full = np.array([T, *tau])

    # Solve for motor forces
    f = np.linalg.solve(mix, tau_full)

    #print(f)

    lower_bound = 0.054*9.81
    upper_bound = 0.393*9.81

    f_clipped = np.clip(f, lower_bound, upper_bound)

    return f_clipped

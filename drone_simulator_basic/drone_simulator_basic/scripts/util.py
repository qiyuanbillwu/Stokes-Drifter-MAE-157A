import numpy as np

# function to calculate the rate of change of a unit vector a_hat
def get_a_dot_hat(a, adot):
    if np.linalg.norm(a) == 0:
        print("Error: a is 0!")
        print("Error: a is 0!")
        print("Error: a is 0!")
        print("Error: a is 0!")
        print("Error: a is 0!")
        print("Error: a is 0!")
        print("Error: a is 0!")
        print("Error: a is 0!")
        print("Error: a is 0!")
        print("Error: a is 0!")

    return adot / np.linalg.norm(a) - a * (a.T @ adot) / np.linalg.norm(a)**3

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

def allocation_matrix(l,d):
    #  Front
    #    ^
    #    |
    # 1      2
    #    |
    # 4      3

    # 1 CCW
    # 2 CW
    # 3 CCW
    # 4 CW

    return np.array([
    [1, 1, 1, 1],        # Total thrust
    [l, l, -l, -l],      # Roll
    [l, -l, -l, l],      # Pitch
    [-d, d, -d, d]       # Yaw
    ])
    
    #try this 
    # return np.array([
    # [1, 1, 1, 1],        # Total thrust
    # [l, l, -l, -l],      # Roll
    # [-l, l, l, -l],      # Pitch
    # [d, -d, d, -d]       # Yaw
    # ])
    

# ==============================================
# alternative allocation matrix
# ==============================================

# def allocation_matrix(l,d):
#     #  Front
#     #    ^
#     #    |
#     # 2      1
#     #    |
#     # 3      4

#     # 1 CW
#     # 2 CCW
#     # 3 CW
#     # 4 CCW

#     return np.array([
#     [1, 1, 1, 1],        # Total thrust
#     [0, -l, 0, l],      # Roll
#     [-l, 0, l, 0],      # Pitch
#     [d, -d, d, -d]       # Yaw
#     ])

def quat_multiply(q1, q2):
    # Hamilton product of two quaternions
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def qdot_from_omega(q, omega):
    """Compute q_dot from quaternion q and angular velocity omega"""
    omega_quat = np.array([0, *omega])
    return 0.5 * quat_multiply(q, omega_quat)

import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll: Rotation around X-axis (in radians)
        pitch: Rotation around Y-axis (in radians)
        yaw: Rotation around Z-axis (in radians)
    
    Returns:
        A tuple (w, x, y, z) representing the quaternion.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)

import numpy as np

def euler_rates_to_body_rates_XYZ(roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot):
    """
    Convert XYZ Euler angle derivatives to body angular velocity (omega_x, omega_y, omega_z).

    Args:
        roll: float, roll angle (X), radians
        pitch: float, pitch angle (Y), radians
        yaw: float, yaw angle (Z), radians
        roll_dot: float, roll rate (d/dt), radians/sec
        pitch_dot: float, pitch rate (d/dt), radians/sec
        yaw_dot: float, yaw rate (d/dt), radians/sec

    Returns:
        np.array of shape (3,), body angular velocity [omega_x, omega_y, omega_z]
    """
    sr = np.sin(roll)
    cr = np.cos(roll)
    sp = np.sin(pitch)
    cp = np.cos(pitch)
    sy = np.sin(yaw)
    cy = np.cos(yaw)

    # Euler XYZ rates to body rates matrix
    J = np.array([
        [1, 0, -sp],
        [0, cr, sr * cp],
        [0, -sr, cr * cp]
    ])

    euler_dot = np.array([roll_dot, pitch_dot, yaw_dot])

    omega = J @ euler_dot

    return omega

from scipy.spatial.transform import Rotation as R

def angular_velocity_body_wxyz(q1_wxyz, q2_wxyz, dt):
    """
    Estimate angular velocity in the body frame from two quaternions (w,x,y,z) over time dt.

    Args:
        q1_wxyz : quaternion at time t (w,x,y,z)
        q2_wxyz : quaternion at time t+dt (w,x,y,z)
        dt      : timestep (float)

    Returns:
        omega_body : angular velocity in body frame at time t, shape (3,)
    """
    # Convert (w,x,y,z) → (x,y,z,w) for use with scipy
    q1_xyzw = np.array([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
    q2_xyzw = np.array([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])

    # Rotation objects
    r1 = R.from_quat(q1_xyzw)
    r2 = R.from_quat(q2_xyzw)

    # Relative rotation: R_delta = R2 * R1⁻¹
    r_delta = r2 * r1.inv()

    # Axis-angle representation → rotation vector (ω·dt)
    rotvec = r_delta.as_rotvec()

    # Angular velocity in world frame
    omega_world = rotvec / dt

    # Convert to body frame at time t
    omega_body = r1.inv().apply(omega_world)

    return omega_body

import numpy as np

def normalize_quaternion_wxyz(q):
    """
    Normalize a quaternion in (w, x, y, z) format.

    Args:
        q : array-like of shape (4,) -- quaternion [w, x, y, z]

    Returns:
        np.array of shape (4,) -- normalized quaternion [w, x, y, z]
    """
    q = np.asarray(q, dtype=np.float64)
    norm_q = np.linalg.norm(q)

    if norm_q < 1e-8:
        raise ValueError("Cannot normalize a zero-magnitude quaternion")

    return q / norm_q

def noise_percent(idealValue, mu, sigma):
    #mu, sigma = 0, 0.1 # example mean and standard deviation
    s = np.random.normal(mu, sigma, 1)
    return idealValue * (1 + s);

def noise_offset(idealValue, mu, sigma):
    #mu, sigma = 0, 0.1 # example mean and standard deviation
    s = np.random.normal(mu, sigma, 1)
    return idealValue * s;

# Summary of States Array
# state = [posX, posY, posZ, velX, velY, velZ, qw, qx, qy, qz, wx, wy, wz]
# index >>  0     1     2     3     4     5    6   7   8   9   10  11  12

def addNoiseToPercievedState(state, stdNoisePos, percNoiseOther):
    pos = state[0:3]
    nPos = noise_percent(pos,0,stdNoisePos)

    other = state[3:13]
    nOther = noise_percent(other,0,percNoiseOther)

    # Normalize Unit Quaternion
    nQ = nOther[3:7];
    nOther[3:7] = normalize_quaternion_wxyz(nQ)

    nState = np.append(nPos, nOther);
    #print(nState)

    return nState;

def rotate_vector_by_quat(v, q):
    # v: 3D vector
    v_q = [0] + list(v)  # convert to pure quaternion
    q_conj = quat_conjugate(q)
    return quat_multiply(quat_multiply(q, v_q), q_conj)[1:4]
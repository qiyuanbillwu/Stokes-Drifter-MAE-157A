import numpy as np

# function to calculate the rate of change of a unit vector a_hat
def get_a_dot_hat(a, adot):
    if np.linalg.norm(a) == 0:
        print("Error: a is 0!")
    return adot / np.linalg.norm(a) - a * (np.transpose(a) @ adot) / np.linalg.norm(a)**3

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
    [-l, l, l, -l],      # Roll
    [l, l, -l, -l],      # Pitch
    [d, -d, d, -d]       # Yaw
    ])

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

def quaternion_error(q_des, q_curr):
    # Compute q_e = q_des^* ⊗ q_curr
    return quat_multiply(quat_conjugate(q_des), q_curr)

def qdot_from_omega(q, omega):
    """Compute q_dot from quaternion q and angular velocity omega"""
    omega_quat = np.array([0, *omega])
    return 0.5 * quat_multiply(q, omega_quat)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trajectory import get_state, x1, y1, z1, theta
from util import rotate_vector_by_quat

file_name = "/data/data_2025-06-04_12-59-30.csv"
data = np.loadtxt("../"+file_name, delimiter=',')

# Data from File
ts = data[:, 0]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]
vx = data[:, 4]
vy = data[:, 5]
vz = data[:, 6]

qw = data[:, 7]
qx = data[:, 8]
qy = data[:, 9]
qz = data[:, 10]

# Compute Acceleration Vector at Each Timestep
ahat = np.zeros((len(ts),3))
for i in range(0,len(ahat)):
    ahat[i,:] = rotate_vector_by_quat([0,0,1], [qw[i], qx[i], qy[i], qz[i]])

# Desired Trajectory
desiredPos = np.array([get_state(t)['r'] for t in ts])
adhat = np.array([get_state(t)['adhat'] for t in ts])

# Setup Figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=15, azim=160)

# Draw desired trajectory (static, dashed blue)
ax.plot(desiredPos[:, 0], desiredPos[:, 1], desiredPos[:, 2], 'g--', label='Desired Trajectory')

# Rectangle parameters (x-axis aligned)
rect_center = np.array([x1, y1, z1])
width, height = 0.4, 0.7
rotation_angle = theta * 180 / np.pi  # degrees

# Create rectangle vertices (YZ plane)
def create_rotated_rectangle(center, width, height, angle):
    corners = np.array([
        [0, -width/2, -height/2],
        [0, width/2, -height/2],
        [0, width/2, height/2],
        [0, -width/2, height/2]
    ])
    # Rotation matrix (x-axis)
    theta = np.radians(angle)
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    # Apply rotation and translation
    return np.dot(corners, rot.T) + center

# Create and add rectangle
rect_verts = create_rotated_rectangle(rect_center, width, height, rotation_angle)
rect = Poly3DCollection([rect_verts], alpha=0.3, facecolor='green', edgecolor='k')
ax.add_collection3d(rect)

# Animation elements
line, = ax.plot([], [], [], 'b', label='Actual Trajectory')
quiver = ax.quiver([], [], [], [], [], [], length=0.5, color='black', label='Thrust')
quiverDes = ax.quiver([], [], [], [], [], [], length=0.5, color='r', label='Desired Thrust')
point = ax.plot([], [], [], 'ko', markersize=8)[0]
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

# Axis limits and labels
ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(0, 4),
       xlabel='x', ylabel='y', zlabel='z')
ax.legend()

# Animation functions
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    quiver.set_segments([])
    quiverDes.set_segments([])
    point.set_data([], [])
    point.set_3d_properties([])
    time_text.set_text('')
    return line, quiver, point, time_text

def update(frame):
    if frame > 0:
        p = np.array([x[:frame], y[:frame], z[:frame]])
        line.set_data(p[0], p[1])
        line.set_3d_properties(p[2])
        
        p = np.array([x[frame-1], y[frame-1], z[frame-1]])
        point.set_data([p[0]], [p[1]])
        point.set_3d_properties([p[2]])
        
        quiver.set_segments([[p, p + 0.4*ahat[frame-1]]])
        time_text.set_text(f'Time: {ts[frame-1]:.2f}s')

        pDes = desiredPos[frame-1]
        quiverDes.set_segments([[pDes, pDes + 0.4*adhat[frame-1]]])
    return line, quiver, point, time_text

# Run animation
fps = 600
ani = FuncAnimation(fig, update, frames=len(ts), init_func=init,
                   interval=1000/fps, blit=False)

ani.save("../data/simulated_45deg_animation.mp4",
         writer='ffmpeg',
         fps=fps,
         #extra_args=['-vcodec', 'libx264']
)

plt.show()
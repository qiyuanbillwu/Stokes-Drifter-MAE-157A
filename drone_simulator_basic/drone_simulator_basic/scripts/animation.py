import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trajectory import get_state, x1, y1, z1, theta

# Initialize data
t0, t2, dt = 0, 3, 0.01
ts = np.arange(t0, t2, dt)
pos = np.array([get_state(t)['r'] for t in ts])
adhat = np.array([get_state(t)['adhat'] for t in ts])

# Setup figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=15, azim=160)

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
line, = ax.plot([], [], [], 'b', label='Trajectory')
quiver = ax.quiver([], [], [], [], [], [], length=0.5, color='r', label='Thrust')
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
    point.set_data([], [])
    point.set_3d_properties([])
    time_text.set_text('')
    return line, quiver, point, time_text

def update(frame):
    if frame > 0:
        line.set_data(pos[:frame, 0], pos[:frame, 1])
        line.set_3d_properties(pos[:frame, 2])
        
        p = pos[frame-1]
        point.set_data([p[0]], [p[1]])
        point.set_3d_properties([p[2]])
        
        quiver.set_segments([[p, p + 0.4*adhat[frame-1]]])
        time_text.set_text(f'Time: {ts[frame-1]:.2f}s')
    return line, quiver, point, time_text

# Run animation
fps = 20
ani = FuncAnimation(fig, update, frames=len(ts), init_func=init,
                   interval=1000/fps, blit=False)
plt.show()
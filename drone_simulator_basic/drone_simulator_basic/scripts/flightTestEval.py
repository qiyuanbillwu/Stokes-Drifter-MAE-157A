import pandas as pd
import matplotlib.pyplot as plt

#increase gain for positional error

csv_file_path = '../flightData/test1_inner_loop.csv'

csv_file_path2 = '../flightData/test1_outer_loop.csv'

# Read the CSV into a pandas DataFrame
dfInner = pd.read_csv(csv_file_path)

df = pd.read_csv(csv_file_path2)


start_index = df[df['throttle'] > 0].index.min()

# Filter the DataFrame to only include data from that point onward
df = df.loc[start_index:].reset_index(drop=True)

df['t'] = df['t'] - df['t'].iloc[0]

df2 = dfInner

df2 = df2[df2['t'] >= df['t'].iloc[0]].reset_index(drop=True)

# Normalize time to start at 0
df2['t'] = df2['t'] - df2['t'].iloc[0]

# ---- Quaternion Plot ----
fig4 = plt.figure(4,figsize=(10, 6))
plt.plot(df2['t'], df2['qw'], label='qw (actual)', linestyle='-')
plt.plot(df2['t'], df2['qwd'], label='qw (desired)', linestyle='--')
plt.plot(df2['t'], df2['qx'], label='qx (actual)', linestyle='-')
plt.plot(df2['t'], df2['qxd'], label='qx (desired)', linestyle='--')
plt.plot(df2['t'], df2['qy'], label='qy (actual)', linestyle='-')
plt.plot(df2['t'], df2['qyd'], label='qy (desired)', linestyle='--')
plt.plot(df2['t'], df2['qz'], label='qz (actual)', linestyle='-')
plt.plot(df2['t'], df2['qzd'], label='qz (desired)', linestyle='--')
plt.title('Actual vs Desired Quaternion Components')
plt.xlabel('Time [s]')
plt.ylabel('Quaternion Value')
plt.xlim(36,63)
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- Angular Velocity Plot ----
fig5 = plt.figure(5,figsize=(10, 6))
plt.plot(df2['t'], df2['wx'], label='wx (actual)', linestyle='-')
plt.plot(df2['t'], df2['wxd'], label='wx (desired)', linestyle='--')
plt.plot(df2['t'], df2['wy'], label='wy (actual)', linestyle='-')
plt.plot(df2['t'], df2['wyd'], label='wy (desired)', linestyle='--')
plt.plot(df2['t'], df2['wz'], label='wz (actual)', linestyle='-')
plt.plot(df2['t'], df2['wzd'], label='wz (desired)', linestyle='--')
plt.title('Actual vs Desired Angular Velocity')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend()
plt.xlim(36,63)
plt.grid(True)
plt.tight_layout()

# ---- Position Plot ----
fig1 = plt.figure(1, figsize=(10, 6))
plt.plot(df['t'], df['x'], label='x (actual)', linestyle='-')
plt.plot(df['t'], df['xd'], label='x (desired)', linestyle='--')
plt.plot(df['t'], df['y'], label='y (actual)', linestyle='-')
plt.plot(df['t'], df['yd'], label='y (desired)', linestyle='--')
plt.plot(df['t'], df['z'], label='z (actual)', linestyle='-')
plt.plot(df['t'], df['zd'], label='z (desired)', linestyle='--')
plt.title('Actual vs Desired Position')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- Velocity Plot ----
fig2 = plt.figure(2, figsize=(10, 6))
plt.plot(df['t'], df['vx'], label='vx (actual)', linestyle='-')
plt.plot(df['t'], df['vxd'], label='vx (desired)', linestyle='--')
plt.plot(df['t'], df['vy'], label='vy (actual)', linestyle='-')
plt.plot(df['t'], df['vyd'], label='vy (desired)', linestyle='--')
plt.plot(df['t'], df['vz'], label='vz (actual)', linestyle='-')
plt.plot(df['t'], df['vzd'], label='vz (desired)', linestyle='--')
plt.title('Actual vs Desired Velocity')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Throttle vs Time
fig3 = plt.figure(3, figsize=(10, 6))
plt.plot(df['t'], df['throttle'], label='Throttle', color='green')
plt.xlabel('Time [s]')
plt.ylabel('Throttle')
plt.title('Throttle vs Time')
plt.grid(True)
plt.legend()
plt.show()


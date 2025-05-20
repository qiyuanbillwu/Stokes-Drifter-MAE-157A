import numpy as np

# Gravity
g = 9.81

# Other parameters?
#placeholder for now
m = 0.745  # mass of drone [kg]
l = 0.115   # meters [m]
Cd = 0.01   # drag coefficient of propellers [PLACEHOLDER]
Cl = 0.1    # lift coefficent of propellers  [PLACEHOLDER]

# make sure this is agreement with the allocation matrix
J = np.diag([0.00225577, 0.00360365, 0.00181890]) # [kg/m2]
d = Cd / Cl
### Import python packages ###
import numpy as np
import matplotlib.pyplot as plt

# Update with actual file name in the data director
file_name = "temp.csv"

# Load in data as giant matrix
data = np.loadtxt("../data/"+file_name, delimiter=',')
print(data)

plt.figure(1)
# Add code here #
# plt.plot(...)	#
#				#
plt.grid()

plt.figure(2)
# Add code here #
# plt.plot(...)	#
#				#
plt.grid()

plt.show()
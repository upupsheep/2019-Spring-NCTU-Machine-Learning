# Import packages
import numpy as np
from matplotlib import pyplot as plt

# Generate sample data
x = np.sort(5 * np.random.rand(100, 1), axis=0)
# x = np.arange(1.0, 40.0, 0.5)
y = np.sin(x).ravel()
# y = x * x * x * x  # x^4

# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(20))

x = np.asarray(x)
y = np.asarray(y)

points = np.c_[x, y]
print(points)

np.savetxt('testfile_sin100.txt', points, delimiter=',')
# print(x)
# print(x.shape)

# plt.plot(x, y, 'ro')
# plt.show()

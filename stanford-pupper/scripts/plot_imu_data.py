import numpy as np
import matplotlib.pyplot as plt

x_gt = []
x_kalman = np.load('data/imu_test.npz')['x']
time = np.load('data/imu_test.npz')['t']
for t in time:
    if t < 1:
        x_gt.append(0)
    elif t < 4.5:
        x_gt.append(-0.3)
    elif t < 7.5:
        x_gt.append(0)
    elif t < 12.4:
        x_gt.append(0.3)
    else:
        x_gt.append(0.0)

plt.figure()
plt.plot(time, x_kalman, label='kalman')
plt.plot(time, x_gt, label='approx gt')
plt.legend()
plt.show()

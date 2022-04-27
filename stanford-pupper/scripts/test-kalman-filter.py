
import time
import numpy as np
from woofer.imu import Imu


imu = Imu()


def main():
    history_x = []
    history_y = []
    time_history = []
    start_time = time.time()
    finished = False
    while not finished:
        velocity = imu.read_imu_vel()
        history_x.append(velocity[0])
        history_y.append(velocity[1])
        time_elapsed = time.time() - start_time
        finished = time_elapsed > 20
        time_history.append(time_elapsed)

    np.savez("imu_test.npz", x=np.array(history_x), y=np.array(history_y), t=np.array(time_history))


if __name__ == "__main__":
    main()    



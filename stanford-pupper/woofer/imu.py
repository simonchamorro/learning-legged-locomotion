
import time
import adafruit_bno055
from adafruit_extended_bus import ExtendedI2C as I2C
import numpy as np
from woofer.kalman import Kalman


class Imu:
    def __init__(self):
        i2c = I2C(4)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c, address=0x29)

        self.last_val = 0xFFFF
        self.prev_time = time.time()

        self.kalman = Kalman()
    

    def read_imu_vel(self):
        accel = np.array(list(self.sensor.acceleration)) - np.array(list(self.sensor.gravity))
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
    
        vel_measurement = accel * dt
        velocity = self.kalman.compute_kalman(vel_measurement)
        return velocity
    

    def read_ang_vel(self):
        return np.array(list(self.sensor.gyro))

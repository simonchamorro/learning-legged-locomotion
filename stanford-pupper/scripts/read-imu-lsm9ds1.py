# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
import adafruit_lsm9ds1
from adafruit_extended_bus import ExtendedI2C as I2C

# Create sensor object, communicating over the board's I2C bus
i2c = I2C(4) 
sensor = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)

# Main loop will read the acceleration, magnetometer, gyroscope, Temperature
# values every second and print them out.
while True:
    # Read acceleration, magnetometer, gyroscope, temperature.
    accel_x, accel_y, accel_z = sensor.acceleration
    mag_x, mag_y, mag_z = sensor.magnetic
    gyro_x, gyro_y, gyro_z = sensor.gyro
    temp = sensor.temperature
    # Print values.
    print(
        "Acceleration (m/s^2): ({0:0.3f},{1:0.3f},{2:0.3f})".format(
            accel_x, accel_y, accel_z
        )
    )
    print(
        "Magnetometer (gauss): ({0:0.3f},{1:0.3f},{2:0.3f})".format(mag_x, mag_y, mag_z)
    )
    print(
        "Gyroscope (rad/sec): ({0:0.3f},{1:0.3f},{2:0.3f})".format(
            gyro_x, gyro_y, gyro_z
        )
    )
    print("Temperature: {0:0.3f}C".format(temp))
    # Delay for a second.
    time.sleep(1.0)
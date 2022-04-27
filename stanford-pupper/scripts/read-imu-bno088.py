from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ACCELEROMETER
from adafruit_extended_bus import ExtendedI2C as I2C

i2c = I2C(4)
bno = BNO08X_I2C(i2c, address=0x4B)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)

while True:
    accel_x, accel_y, accel_z = bno.acceleration
    print("X: %0.6f  Y: %0.6f Z: %0.6f  m/s^2" % (accel_x, accel_y, accel_z))
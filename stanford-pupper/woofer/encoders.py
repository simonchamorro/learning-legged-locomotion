
import time
import numpy as np
import pigpio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from adafruit_extended_bus import ExtendedI2C as I2C


class Encoders:
    def __init__(self):
        # Key = leg_idx, motor_idx
        # Value = motor bin number for mux
        self.motor2bin = {(0, 0):  [0, 0, 0],
                    (0, 1):  [1, 0, 0],
                    (0, 2):  [0, 1, 0],
                    (1, 0):  [1, 1, 0],
                    (1, 1):  [0, 0, 1],
                    (1, 2):  [1, 0, 1],
                    (2, 0):  [0, 0, 0],
                    (2, 1):  [1, 0, 0],
                    (2, 2):  [0, 1, 0],
                    (3, 0):  [1, 1, 0],
                    (3, 1):  [0, 0, 1],
                    (3, 2):  [1, 0, 1],}

        # GPIO pins, not the same as pin numbers
        # see: http://abyz.me.uk/rpi/pigpio/index.html#Type_3
        self.mux_pins_front = [16, 26, 19]
        # self.mux_pins_back = [6, 5, 0]

        # Neutral positions
        self.neutral_pos = [0, 45, -45]

        # Init i2c object
        i2c = I2C(4)

        # Create Pi object
        self.pi = pigpio.pi()

        # Create the ADC object using the I2C bus
        ads = ADS.ADS1015(i2c)

        # Create single-ended input on channel 0
        self.chan0 = AnalogIn(ads, ADS.P0)
        self.chan1 = AnalogIn(ads, ADS.P1)

        # Load encoders calibration
        self.encoder_calib = np.load("data/encoder_config.npz")["encoder_calib"]

        # Init last pos
        self.last_pos = np.zeros((4, 3))
        self.last_time = np.ones((4, 3))*time.time()
        self.current_motor = (0, 0)
        self.set_motor(self.pi, (0, 1))


    def set_motor(self, pi, motor=0, mux_pins=[]):
        if self.current_motor == motor:
            pass
        
        bin_num = self.motor2bin[motor]
        for idx, pin in enumerate(mux_pins):
            pi.write(pin, bin_num[idx])
        self.current_motor = motor
            

    def get_motor_name(self, i, j):
        motor_type = {0: "hip", 1: "thigh", 2: "calf"} 
        leg_pos = {0: "front-right", 1: "front-left", 2: "back-right", 3: "back-left"}
        final_name = motor_type[i] + " " + leg_pos[j]
        return final_name
    

    def read_encoder_position(self, leg_idx, motor_idx):

        # Set the mux to desired motor
        self.set_motor(self.pi, (leg_idx, motor_idx), self.mux_pins_front)
        # self.set_motor(self.pi, (leg_idx, motor_idx), self.mux_pins_back)

        # Choose right channel and read value
        if motor_idx < 2:
            chan = self.chan0
        else:
            chan = self.chan1
        enc_reading = chan.voltage

        # Convert to angle value
        neutral_voltage = self.encoder_calib[leg_idx, motor_idx, 0]
        neutral_angle = self.neutral_pos[motor_idx]
        volt_deg_ratio = self.encoder_calib[leg_idx, motor_idx, 1]
        enc_angle = (enc_reading - neutral_voltage) / volt_deg_ratio + neutral_angle
        return enc_angle
    

    def read_encoder_vel(self, leg_idx, motor_idx):
        pos = self.read_encoder_position(leg_idx, motor_idx)
        current_time = time.time()
        dt = current_time - self.last_time[leg_idx, motor_idx]
        ang_vel = (pos - self.last_pos[leg_idx, motor_idx]) * dt
        self.last_time[leg_idx, motor_idx] = current_time
        return ang_vel


    def read_positions(self):
        positions = np.zeros(4, 3)
        for leg_idx in range(4):
            for motor_idx in range(3):
                pos = self.read_encoder_position(leg_idx, motor_idx)
                positions[leg_idx, motor_idx] = pos
        return positions
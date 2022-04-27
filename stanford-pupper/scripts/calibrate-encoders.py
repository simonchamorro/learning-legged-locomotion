
import time
import numpy as np
import pigpio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from adafruit_extended_bus import ExtendedI2C as I2C
from pupper.HardwareInterface import HardwareInterface


# Key = leg_idx, motor_idx
# Value = motor bin number for mux
MOTOR2BIN = {(0, 0):  [0, 0, 0],
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
MUX_PINS_FRONT = [16, 26, 19]
# MUX_PINS_BACK = [6, 5, 0]

# Neutral positions
NEUTRAL_POS = [0, 45, -45]


def set_motor(pi, motor=0, mux_pins=[]):
    bin_num = MOTOR2BIN[motor]
    for idx, pin in enumerate(mux_pins):
        pi.write(pin, bin_num[idx])

def get_motor_name(i, j):
    motor_type = {0: "hip", 1: "thigh", 2: "calf"} 
    leg_pos = {0: "front-right", 1: "front-left", 2: "back-right", 3: "back-left"}
    final_name = motor_type[i] + " " + leg_pos[j]
    return final_name

def degrees_to_radians(input_array):
    return np.pi / 180.0 * input_array

def calibrate_encoders(hardware_interface):
    # Create the I2C bus
    i2c = I2C(4)
    delay = 0.6

    # Create Pi object
    pi = pigpio.pi()

    # Create the ADC object using the I2C bus
    ads = ADS.ADS1015(i2c)

    # Create single-ended input on channel 0
    chan0 = AnalogIn(ads, ADS.P0)
    chan1 = AnalogIn(ads, ADS.P1)
    
    # Encoder matrix
    # encoder_config[leg_idx, motor_idx] = [neutral_voltage, volts / deg ratio, min, max]
    encoder_config = np.zeros((4, 3, 4)) 

    print("\n===========================\n")
    for leg_idx in range(4):
        for motor_idx in range(3):
            if motor_idx < 2:
                chan = chan0
            else:
                chan = chan1
            motor_config_done = False
            while not motor_config_done:
                motor_name = get_motor_name(motor_idx, leg_idx)
                motor_pos = NEUTRAL_POS[motor_idx]
                print("\n\nCalibrating the **" + motor_name + " motor **")
                move_input = str(input("Enter 'a' to start calibrating, the motor will move to neutral position, then neutral + 30 and neutral - 30:\n"))
                
                if not move_input == "a":   
                    pass
                
                # Set the mux to desired motor
                set_motor(pi, (leg_idx, motor_idx), MUX_PINS_FRONT)
                # set_motor(pi, (leg_idx, motor_idx), MUX_PINS_BACK)
                
                # Set the motor to the neutral position
                time.sleep(delay)
                hardware_interface.set_actuator_position(degrees_to_radians(motor_pos), motor_idx, leg_idx) 
                
                # Read the encoder value
                time.sleep(delay)
                neutral_voltage = chan.voltage
                encoder_config[leg_idx, motor_idx, 0] = neutral_voltage
                
                # Set the motor to the max position and read value
                time.sleep(delay)
                hardware_interface.set_actuator_position(degrees_to_radians(motor_pos + 30), motor_idx, leg_idx)
                time.sleep(delay)
                max_val = chan.voltage
                
                # Set the motor to the min position
                time.sleep(delay)
                hardware_interface.set_actuator_position(degrees_to_radians(motor_pos - 30), motor_idx, leg_idx)
                time.sleep(delay)
                min_val = chan.voltage

                # Set the motor to the neutral position
                time.sleep(delay)
                hardware_interface.set_actuator_position(degrees_to_radians(motor_pos), motor_idx, leg_idx)
                
                ratio = (max_val - min_val) / 60.0
                encoder_config[leg_idx, motor_idx, 1] = ratio
                encoder_config[leg_idx, motor_idx, 2] = min_val
                encoder_config[leg_idx, motor_idx, 3] = max_val
                
                motor_config_done = True
        
    return encoder_config

    
def main():
    """Main program
    """
    hardware_interface = HardwareInterface()
    config = calibrate_encoders(hardware_interface)
    print("\n\n CALIBRATION COMPLETE!\n")
    print("Calibrated encoder values:")
    print(config)
    np.savez("data/encoder_config.npz", encoder_calib=config)
    a = np.load("data/encoder_config.npz")["encoder_calib"]
    print(a)

if __name__ == "__main__":
    main()    

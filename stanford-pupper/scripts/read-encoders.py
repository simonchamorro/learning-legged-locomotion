
import time
import numpy as np
from woofer.encoders import Encoders



def main():
    encoders = Encoders()

    while True:
        for leg_idx in range(4):
            for motor_idx in range(3):
                enc_angle = encoders.read_encoder_position(leg_idx, motor_idx)

                print("\n===========================\n")
                motor_name = encoders.get_motor_name(motor_idx, leg_idx)
                print(motor_name)
                print("Encoder angle: {}".format(enc_angle))
                
        time.sleep(1.0)



if __name__ == "__main__":
    main()    
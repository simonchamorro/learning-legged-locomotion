from ntpath import join
import numpy as np
import pybullet

class Encoders:
    def __init__(self):
        pass

    def read_pos_vel(self):
        # Read data for all joints
        pos = np.zeros(12)
        vel = np.zeros(12)
        for i in range(12):
            joint_state = pybullet.getJointState(1, 2*i)
            pos[i] = joint_state[0]
            vel[i] = joint_state[1]
        return pos, vel

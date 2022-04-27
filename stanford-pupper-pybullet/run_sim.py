import pybullet as p
import pybullet_data
import time
import numpy as np


from policy.utils import extract_last_model_save
from sim.Sim import Sim, PupperSim

import torch
import torch.nn as nn
import torch.functional as F

import os
import glob

def main(default_velocity=np.zeros(2), default_yaw_rate=0.0, policy_path=None, use_policy=False):
    # Create Simulation
    sim = PupperSim(xml_path="sim/pupper_pybullet_out.xml", policy_path=policy_path, use_policy=use_policy)
    # Run it
    sim.run(verbose=True)

if __name__ == "__main__":
    main(policy_path=extract_last_model_save("./policy/pupper/*"), use_policy=True)

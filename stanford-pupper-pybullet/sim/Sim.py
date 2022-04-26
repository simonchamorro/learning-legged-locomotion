import numpy as np
import pybullet
import pybullet_data
import time
import torch
import torch.nn as nn

from .HardwareInterface import HardwareInterface
from .IMU import IMU
from .Encoders import Encoders

from common.State import State
from common.Controller import Controller
from common.Command import Command
from pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics
from policy.utils import get_policy

class Robot:
    def __init__(self):
        return

class Pupper(Robot):

    def __init__(self,config):
        super().__init__()
        self.encoders = Encoders()
        self.imu = IMU()
        self.controller = Controller(config, four_legs_inverse_kinematics,)
        self.state = State()
        self.command = Command()
        self.last_action = None

        # ###################
        # ROBOT CONFIGURATION
        # ###################
        self.default_yaw_rate = 0.0
        self.action_size = 12
        self.observation_size = 48
        self.default_velocity = torch.zeros((2,))

        # Default methods
        self.init_robot_values()
        self.emulate_joystick_movement()

        # --------------------

    def init_robot_values(self):
        """
        Call mandatory robot stuff
        """
        self.last_action = torch.zeros((self.action_size,))
        self.imu._simulator_observation()

    def emulate_joystick_movement(self):
        """
        Emulate a joystick movement to give an initial velocity to the pupper
        """
        # Emulate a JoyStick Movement in simulation
        self.command.activate_event = 1
        self.controller.run(self.state, self.command)
        self.command.activate_event = 0
        self.command.trot_event = 1
        self.controller.run(self.state, self.command)
        # Reset command and give a velocity to follow
        self.command = Command()
        self.command.horizontal_velocity = self.default_velocity
        self.command.yaw_rate = self.default_yaw_rate

    def done_action(self, state, actions, use_policy=False, use_offset=False):
        """
        Send action/command to the controller depending on the policy usage
        """
        if use_policy:
            self.controller.send_action(self.state, actions, use_offset)
        else:
            self.controller.run(self.state, self.command)

class Sim:
    def __init__(
        self,
        xml_path,
        kp=0.25,
        kv=0.5,
        max_torque=10,
        g=-9.81,
    ):
        # Set up PyBullet Simulator
        pybullet.connect(pybullet.GUI)  # or p.DIRECT for non-graphical version
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        pybullet.setGravity(0, 0, g)
        self.model = pybullet.loadMJCF(xml_path)
        print("")
        print("Pupper body IDs:", self.model)
        numjoints = pybullet.getNumJoints(self.model[1])
        print("Number of joints in converted MJCF: ", numjoints)
        print("Joint Info: ")
        for i in range(numjoints):
            print(pybullet.getJointInfo(self.model[1], i))
        self.joint_indices = list(range(0, 24, 2))

    def step(self):
        pybullet.stepSimulation()


class PupperSim(Sim):
    """
    Custom simulation for the pupper. It allows us to create simply different simulation
    by overiding methods.
    """

    def __init__(self,xml_path, policy_path=None, *args, **kwargs):
        super().__init__(xml_path, *args, **kwargs)

        print("******* Building Default Stanford-Pupper simulation ******")
        self.policy = None
        self.use_policy = policy_path is not None
        self.config = Configuration()
        self.interface = HardwareInterface(self.model, self.joint_indices)
        self.robot = Pupper(self.config)
        if self.use_policy:
            self.policy = get_policy(policy_path)

        # Counter of elapsed time or last update
        self.last_control_update = 0
        self.sim_elapsed_time = 0

        # #######################################
        # Simulation configuration / init method
        # #######################################
        ## Simulation
        # Sim seconds per sim step
        self.timesteps = 240 * 60 * 10  # simulate for a max of 10 minutes
        self.sim_steps_per_sim_second = 60
        self.sim_dt = 1.0 / self.sim_steps_per_sim_second
        self.config.dt = self.sim_dt
        # Simulation appearance
        self.config.z_clearance = 0.05
        ## Action
        self.ACTION_SCALE = 0.25
        self.ACT_CLIP = 100.0
        ## Observation
        self.OBS_TARGET = [0.1, 0.0]
        self.OBS_LIN_VEL_SCALE = 2.0
        self.OBS_ANG_VEL_SCALE = 0.25
        self.OBS_DOF_POS_SCALE = 1.0
        self.OBS_DOF_VEL_SCALE = 0.05
        self.OBS_CLIP = 100.0
        self.OBS_DEFAULT_JOINT_POS = torch.Tensor([-0.15, 0.5, -1.0, 0.15, 0.5, -1.0,
                                          -0.15, 0.7, -1.0, 0.15, 0.7, -1.0])

        print("* Simulation running ... ")


    def pupper(self):
        return self.robot

    def build_actions(self, obs):
        """
        Takes observations and return scaled actions.
        """
        # get action from policy and scale them
        ac_na = self.policy(obs) * self.ACTION_SCALE
        # clip action
        ac_na = ac_na.clamp(-self.ACT_CLIP, self.ACT_CLIP)
        # update pupper last action
        self.pupper().last_action = ac_na
        # reshape actions
        ac_na_good_order= ac_na.view(4,3).T
        return ac_na_good_order

    def build_observations(self):
        """
        Build env observations from internal states of the agent.
        """
        # Get IMU measurement
        self.pupper().state.quat_orientation = self.pupper().imu.read_orientation()

        # Get joint positions and velocities
        joint_pos, joint_vel = self.pupper().encoders.read_pos_vel()
        lin_vel, ang_vel, projected_gravity = self.pupper().imu._simulator_observation()

        # TODO: Construct observation and make sure it is
        # consistent with Isaac Gym values (offsets, scales, etc)
        obs = [torch.Tensor(lin_vel) * self.OBS_LIN_VEL_SCALE,
               torch.Tensor(ang_vel) * self.OBS_ANG_VEL_SCALE,
               torch.Tensor(projected_gravity),
               torch.Tensor([self.pupper().default_velocity[0], 0, 0]) * self.OBS_LIN_VEL_SCALE,
               (torch.Tensor(joint_pos) - self.OBS_DEFAULT_JOINT_POS) * self.OBS_DOF_POS_SCALE,
               torch.Tensor(joint_vel * self.OBS_DOF_VEL_SCALE),
               self.pupper().last_action]

        obs = torch.cat(obs, dim=-1)

        return obs

    def interact_with_env(self):
        """
        Generate observation, actions and make the robot move
        """
        # Improve the step
        obs = self.build_observations()

        if self.use_policy:
            actions = self.build_actions(obs)
            self.pupper().done_action(self.pupper().state, actions, use_policy=self.use_policy, use_offset=True)

        # Update the pwm widths going to the servos
        self.interface.set_actuator_postions(self.pupper().state.joint_angles)
        return

    def run(self, verbose=False):
        """
        Run the whole pupper simulation

        verbose: printing logs (bool)
        """
        start = time.time()
        for steps in range(self.timesteps):
            self.sim_elapsed_time = self.sim_dt * steps

            if self.sim_elapsed_time - self.last_control_update > self.config.dt:
                self.last_control_update = self.sim_elapsed_time

                # take a step of updating the simulation
                self.interact_with_env()

            # update the env
            super().step()

            if verbose:
                # Performance testing
                elapsed = time.time() - start
                if ((steps + 1) % 1000) == 0:
                    print("Sim seconds elapsed: {}, Real seconds elapsed: {}".format(round(self.sim_elapsed_time, 3), round(elapsed, 3)))

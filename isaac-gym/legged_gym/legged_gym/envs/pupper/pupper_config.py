# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class PupperCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.18]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint":     0.15,    # [rad]
            "RL_hip_joint":     0.15,    # [rad]
            "FR_hip_joint":     -0.15,    # [rad]
            "RR_hip_joint":     -0.15,    # [rad]
            "FL_thigh_joint":     0.5,    # [rad]
            "RL_thigh_joint":     0.7,   # [rad]
            "FR_thigh_joint":     0.5,    # [rad]
            "RR_thigh_joint":     0.7,    # [rad]
            "FL_calf_joint":    -1.0,   # [rad]
            "RL_calf_joint":    -1.0,   # [rad]
            "FR_calf_joint":    -1.0,   # [rad]
            "RR_calf_joint":    -1.0,   # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 10.0}  # [N*m/rad]
        damping = {"joint": 0.05}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pupper/pupper.urdf"
        collapse_fixed_joints = False
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "gripper", "upper", "lower"]
        terminate_after_contacts_on = ["base", "upperarm", "lowerarm", "hip", "torso"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter


class PupperFlatCfg(PupperCfg):
    class env(PupperCfg.env):
        num_observations = 48

    class terrain( PupperCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False


class PupperHighLegMassCfg(PupperFlatCfg):
    class env(PupperFlatCfg.env):
        num_envs = 4096

    class init_state(PupperFlatCfg.init_state):
        pos = [0.0, 0.0, 0.20]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.15,  # [rad]
            "RL_hip_joint": 0.15,  # [rad]
            "FR_hip_joint": -0.15,  # [rad]
            "RR_hip_joint": -0.15,  # [rad]
            "FL_upper_joint": 0.0,  # [rad]
            "RL_upper_joint": 0.0,  # [rad]
            "FR_upper_joint": 0.0,  # [rad]
            "RR_upper_joint": 0.0,  # [rad]
            "FL_lower_joint": 0.0,  # [rad]
            "RL_lower_joint": 0.0,  # [rad]
            "FR_lower_joint": 0.0,  # [rad]
            "RR_lower_joint": 0.0,  # [rad]
        }

    class control(PupperFlatCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 10.0}  # [N*m/rad] # p_gains in code (penalty for wrong pos)
        damping = {"joint": 0.05}  # [N*m*s/rad] # d_gains in code (penalty for wrong vel)
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PupperFlatCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pupper/pupper-v2-higherlegmass.urdf"
        collapse_fixed_joints = False
        foot_name = "foot"
        penalize_contacts_on = ["upper", "lower"]
        # penalize_contacts_on = []
        terminate_after_contacts_on = ["torso", "hip"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(PupperFlatCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(PupperFlatCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class PupperGo1MassCfg(PupperFlatCfg):
    class env(PupperFlatCfg.env):
        num_envs = 4096

    class init_state(PupperFlatCfg.init_state):
        pos = [0.0, 0.0, 0.24]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint":     0.15,    # [rad]
            "RL_hip_joint":     0.15,    # [rad]
            "FR_hip_joint":     -0.15,    # [rad]
            "RR_hip_joint":     -0.15,    # [rad]
            "FL_thigh_joint":     0.9,    # [rad]
            "RL_thigh_joint":     1.1,   # [rad]
            "FR_thigh_joint":     0.9,    # [rad]
            "RR_thigh_joint":     1.1,    # [rad]
            "FL_calf_joint":    -1.7,   # [rad]
            "RL_calf_joint":    -1.7,   # [rad]
            "FR_calf_joint":    -1.7,   # [rad]
            "RR_calf_joint":    -1.7,   # [rad]
        }

    class control(PupperFlatCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 20.0}  # [N*m/rad]
        damping = {"joint": 1.049}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PupperFlatCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pupper/pupper-go1-mass.urdf"
        collapse_fixed_joints = False
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "gripper"]
        terminate_after_contacts_on = ["hip","base", "upperarm", "lowerarm"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(PupperFlatCfg.rewards):
        soft_dof_pos_limit = 1.0 # lower this value?
        base_height_target = 0.25

        class scales(PupperFlatCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class PupperA1MassCfg(PupperFlatCfg):
    class env(PupperFlatCfg.env):
        num_envs = 4096

    class init_state(PupperFlatCfg.init_state):
        pos = [0.0, 0.0, 0.24]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint":     0.15,    # [rad]
            "RL_hip_joint":     0.15,    # [rad]
            "FR_hip_joint":     -0.15,    # [rad]
            "RR_hip_joint":     -0.15,    # [rad]
            "FL_thigh_joint":     0.9,    # [rad]
            "RL_thigh_joint":     1.1,   # [rad]
            "FR_thigh_joint":     0.9,    # [rad]
            "RR_thigh_joint":     1.1,    # [rad]
            "FL_calf_joint":    -1.7,   # [rad]
            "RL_calf_joint":    -1.7,   # [rad]
            "FR_calf_joint":    -1.7,   # [rad]
            "RR_calf_joint":    -1.7,   # [rad]
        }

    class control(PupperFlatCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 20.0}  # [N*m/rad]
        damping = {"joint": 1.049}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PupperFlatCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pupper/pupper-a1-mass.urdf"
        collapse_fixed_joints = False
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "gripper"]
        terminate_after_contacts_on = ["hip","base", "upperarm", "lowerarm"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter


class PupperCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pupper"


class PupperFlatCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pupper_flat"
        max_iterations = 500


class PupperFlatHighLegMassCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 50
        max_iterations = 500
        experiment_name = "pupper_flat_high_leg_mass"


class PupperA1MassCfgPPO(PupperFlatCfgPPO):
    class runner(PupperFlatCfgPPO.runner):
        experiment_name = "pupper_flat_a1_mass"


class PupperGo1MassCfgPPO(PupperFlatCfgPPO):
    class runner(PupperFlatCfgPPO.runner):
        experiment_name = "pupper_flat_go1_mass"

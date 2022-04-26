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

from legged_gym.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class A1RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 20.0}  # [N*m/rad]
        damping = {"joint": 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "gripper"]
        terminate_after_contacts_on = ["base", "upperarm", "lowerarm"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class A1RoughTableCfg(A1RoughCfg):
    class terrain(A1RoughCfg.terrain):
        table_leg_fraction = 0.333 # gets overlayed into other terrains


class A1RoughArmCfg(A1RoughCfg):
    class env(A1RoughCfg.env):
        # num_envs = 9  # TODO: was 4096
        num_envs = 4096
        # num_observations = 244  # 235 + 3 arm
        num_observations = 250  # 235 + 3 arm
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 15  # 12 leg + 3 arm

    class init_state(A1RoughCfg.init_state):
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
            "shoulder_joint": 0,
            "upperarm_joint": 0,
            "lowerarm_joint": 0,
        }

    class asset(A1RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1-arm.urdf"

    # class rewards(A1RoughCfg.rewards):
    #     soft_dof_pos_limit = 0.9
    #     base_height_target = 0.25
    #
    #     class scales(A1RoughCfg.rewards.scales):
    #         torques = -0.0002
    #         dof_pos_limits = -10.0

    class normalization(A1RoughCfg.normalization):
        class obs_scales(A1RoughCfg.normalization.obs_scales):
            lin_vel_arm = 0.5

    class commands(A1RoughCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4 + 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        # (in heading mode ang_vel_yaw is recomputed from heading error)
        # PLUS lin_vel_arm_x, lin_vel_arm_y, lin_vel_arm_z
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        with_arm = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            lin_vel_arm_x = [-0.1, 0.1]  # min max [m/s]
            lin_vel_arm_y = [-0.1, 0.1]  # min max [m/s]
            lin_vel_arm_z = [-0.1, 0.1]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards(A1RoughCfg.rewards):
        tracking_sigma_arm = 0.25  # tracking reward = exp(-error^2/sigma)

        class scales(A1RoughCfg.rewards.scales):
            action_rate = -0.01
            ang_vel_xy = -0.01
            base_height = -0.5
            collision = -2
            dof_pos_limits = -20
            feet_air_time = 1
            lin_vel_z = -2  # top 2 most important, positive correlation
            termination = -2  # top 3 most important, neg corr
            torques = -0.0002
            tracking_ang_vel = 0.1
            tracking_lin_vel = 2  # top 1  most important, pos corr
            tracking_lin_vel_arm = 1.5  # top 4 most important, neg corr


class A1RoughArmNoMoveCfg(A1RoughCfg):
    class env(A1RoughCfg.env):
        # num_envs = 9  # TODO: was 4096
        num_envs = 4096
        num_observations = 244
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 15  # 12 leg + 3 arm

    class init_state(A1RoughArmCfg.init_state):
        pass

    class asset(A1RoughArmCfg.asset):
        pass


class A1RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_a1"


class A1RoughArmCfgPPO(A1RoughCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_a1_arm"


class A1RoughArmNoMoveCfgPPO(A1RoughCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_a1_arm_nm"


class A1RoughRGBCameraCfg(A1RoughCfg):
    class env(A1RoughCfg.env):
        num_envs = 32
        use_images = True
        img_dims = (84, 84, 4)
        img_enc_output_dim = 3136  # 9216 for 128


class A1RoughRGBCameraPPO(A1RoughCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_a1_rgb"

    class policy(LeggedRobotCfgPPO.policy):
        img_dims = (84, 84, 4)
        img_enc_output_dim = 3136  # 9216 for 128


class A1FlatRGBCameraCfg(A1RoughCfg):
    class env(A1RoughCfg.env):
        num_envs = 32
        use_images = True
        num_observations = 48
        img_dims = (84, 84, 4)
        img_enc_output_dim = 3136  # 9216 for 128

    class terrain(A1RoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class asset(A1RoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(A1RoughCfg.rewards):
        max_contact_force = 350.0

        class scales(A1RoughCfg.rewards.scales):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.0
            # feet_contact_forces = -0.01

    class commands(A1RoughCfg.commands):
        heading_command = False
        resampling_time = 4.0

        class ranges(A1RoughCfg.commands.ranges):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand(A1RoughCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        friction_range = [
            0.0,
            1.5,
        ]


class A1FlatRGBCameraPPO(A1RoughCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "flat_a1_rgb"
        load_run = -1
        max_iterations = 300

    class policy(LeggedRobotCfgPPO.policy):
        img_enc_output_dim = 3136  # 9216 for 128
        img_channels = 4


class A1FlatCfg(A1RoughCfg):
    class env(A1RoughCfg.env):
        num_observations = 48

    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

class A1FlatCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "a1_flat"
        max_iterations = 500
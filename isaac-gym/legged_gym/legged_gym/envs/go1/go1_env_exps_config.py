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

from pupperfetch.legged_gym.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from pupperfetch.legged_gym.envs.go1.go1_config import Go1FlatCfg, Go1FlatCfgPPO, Go1Cfg, Go1CfgPPO

# Action repeat
class Go1FlatCfg1(Go1FlatCfg):
    class control( Go1FlatCfg.control ):
        decimation = 1
class Go1FlatCfgPPO1(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_repeat1"

class Go1FlatCfg2(Go1FlatCfg):
    class control( Go1FlatCfg.control ):
        decimation = 2
class Go1FlatCfgPPO2(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_repeat2"

class Go1FlatCfg3(Go1FlatCfg):
    class control( Go1FlatCfg.control ):
        decimation = 3
class Go1FlatCfgPPO3(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_repeat3"

class Go1FlatCfg4(Go1FlatCfg):
    class control( Go1FlatCfg.control ):
        decimation = 4
class Go1FlatCfgPPO4(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_repeat4"

class Go1FlatCfg5(Go1FlatCfg):
    class control( Go1FlatCfg.control ):
        decimation = 5
class Go1FlatCfgPPO5(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_repeat5"

class Go1FlatCfg6(Go1FlatCfg):
    class control( Go1FlatCfg.control ):
        decimation = 6
class Go1FlatCfgPPO6(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_repeat6"


# Control type
class Go1FlatCfgV(Go1FlatCfg):
    class control( Go1FlatCfg.control ):
        control_type = "V"
class Go1FlatCfgPPOV(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_ctrl_vel"

class Go1FlatCfgT(Go1FlatCfg):
    class control( Go1FlatCfg.control ):
        control_type = "T"
class Go1FlatCfgPPOT(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_ctrl_torque"


# Curriculum
class Go1FlatCfgCmdCurr(Go1FlatCfg):
    class commands( Go1FlatCfg.commands ):
        curriculum = True
class Go1FlatCfgPPOCmdCurr(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_cmd_curriculum"

class Go1CfgNoCurr(Go1Cfg):
    class commands( Go1Cfg.commands ):
        curriculum = False
    class terrain (Go1Cfg.terrain):
        curriculum = False
class Go1CfgPPONoCurr(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_no_curriculum"

class Go1CfgCmdCurr(Go1Cfg):
    class commands( Go1Cfg.commands ):
        curriculum = True
    class terrain (Go1Cfg.terrain):
        curriculum = False
class Go1CfgPPOCmdCurr(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_cmd_curriculum"

class Go1CfgTerrainCurr(Go1Cfg):
    class commands( Go1Cfg.commands ):
        curriculum = False
    class terrain (Go1Cfg.terrain):
        curriculum = True
class Go1CfgPPOTerrainCurr(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_terrain_curriculum"

class Go1CfgBothCurr(Go1Cfg):
    class commands( Go1Cfg.commands ):
        curriculum = True
    class terrain (Go1Cfg.terrain):
        curriculum = True
class Go1CfgPPOBothCurr(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_cmd_terrain_curriculum"


# Reward Ablations Flat
class Go1FlatCfgRewNoActionRate(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            action_rate = 0.0
class Go1FlatCfgPPORewNoActionRate(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_action_rate"

class Go1FlatCfgRewNoAngVelXY(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            ang_vel_xy = 0.0
class Go1FlatCfgPPORewNoAngVelXY(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_ang_vel_xy"

class Go1FlatCfgRewNoCollision(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            collision = 0.0
class Go1FlatCfgPPORewNoCollision(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_collision"

class Go1FlatCfgRewNoDofAcc(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            dof_acc = 0.0
class Go1FlatCfgPPORewNoDofAcc(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_dof_acc"

class Go1FlatCfgRewNoAirTime(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            feet_air_time = 0.0
class Go1FlatCfgPPORewNoAirTime(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_air_time"

class Go1FlatCfgRewNoVelZ(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            lin_vel_z = 0.0
class Go1FlatCfgPPORewNoVelZ(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_vel_z"

class Go1FlatCfgRewNoTorque(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            torques = 0.0
class Go1FlatCfgPPORewNoTorque(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_torque"

class Go1FlatCfgRewNoAngVel(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            tracking_ang_vel = 0.0
class Go1FlatCfgPPORewNoAngVel(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_ang_vel"

class Go1FlatCfgRewNoLinVel(Go1FlatCfg):
    class rewards(Go1FlatCfg.rewards):
        class scales(Go1FlatCfg.rewards.scales):
            tracking_lin_vel = 0.0
class Go1FlatCfgPPORewNoLinVel(Go1FlatCfgPPO):
    class runner(Go1FlatCfgPPO.runner):
        experiment_name = "go1_flat_rew_no_lin_vel"


# Reward Ablations Rough
class Go1CfgRewNoActionRate(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            action_rate = 0.0
class Go1CfgPPORewNoActionRate(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_action_rate"

class Go1CfgRewNoAngVelXY(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            ang_vel_xy = 0.0
class Go1CfgPPORewNoAngVelXY(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_ang_vel_xy"

class Go1CfgRewNoCollision(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            collision = 0.0
class Go1CfgPPORewNoCollision(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_collision"

class Go1CfgRewNoDofAcc(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            dof_acc = 0.0
class Go1CfgPPORewNoDofAcc(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_dof_acc"

class Go1CfgRewNoAirTime(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            feet_air_time = 0.0
class Go1CfgPPORewNoAirTime(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_air_time"

class Go1CfgRewNoVelZ(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            lin_vel_z = 0.0
class Go1CfgPPORewNoVelZ(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_vel_z"

class Go1CfgRewNoTorque(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            torques = 0.0
class Go1CfgPPORewNoTorque(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_torque"

class Go1CfgRewNoAngVel(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            tracking_ang_vel = 0.0
class Go1CfgPPORewNoAngVel(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_ang_vel"

class Go1CfgRewNoLinVel(Go1Cfg):
    class rewards(Go1Cfg.rewards):
        class scales(Go1Cfg.rewards.scales):
            tracking_lin_vel = 0.0
class Go1CfgPPORewNoLinVel(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        experiment_name = "go1_rew_no_lin_vel"




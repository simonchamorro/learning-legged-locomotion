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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO, A1FlatCfg, A1FlatCfgPPO
from .go1.go1_config import Go1Cfg, Go1CfgPPO, Go1FlatCfg, Go1FlatCfgPPO
from .pupper.pupper_config import PupperCfg, PupperCfgPPO, PupperFlatCfg, PupperFlatCfgPPO, PupperHighLegMassCfg
from .pupper.pupper_config import PupperHighLegMassCfg, PupperFlatHighLegMassCfgPPO, PupperA1MassCfg, PupperA1MassCfgPPO, PupperGo1MassCfg, PupperGo1MassCfgPPO

from .go1.go1_env_exps_config import (
    Go1FlatCfg1,
    Go1FlatCfg2,
    Go1FlatCfg3,
    Go1FlatCfg4,
    Go1FlatCfg5,
    Go1FlatCfg6,
    Go1FlatCfgV,
    Go1FlatCfgT,
    Go1FlatCfgCmdCurr,
    Go1CfgNoCurr,
    Go1CfgCmdCurr,
    Go1CfgTerrainCurr,
    Go1CfgBothCurr,
    Go1FlatCfgPPO1,
    Go1FlatCfgPPO2,
    Go1FlatCfgPPO3,
    Go1FlatCfgPPO4,
    Go1FlatCfgPPO5,
    Go1FlatCfgPPO6,
    Go1FlatCfgPPOV,
    Go1FlatCfgPPOT,
    Go1FlatCfgPPOCmdCurr,
    Go1CfgPPONoCurr,
    Go1CfgPPOCmdCurr,
    Go1CfgPPOTerrainCurr,
    Go1CfgPPOBothCurr,
    Go1FlatCfgRewNoActionRate,
    Go1FlatCfgPPORewNoActionRate,
    Go1FlatCfgRewNoAngVelXY,
    Go1FlatCfgPPORewNoAngVelXY,
    Go1FlatCfgRewNoCollision,
    Go1FlatCfgPPORewNoCollision,
    Go1FlatCfgRewNoDofAcc,
    Go1FlatCfgPPORewNoDofAcc,
    Go1FlatCfgRewNoAirTime,
    Go1FlatCfgPPORewNoAirTime,
    Go1FlatCfgRewNoVelZ,
    Go1FlatCfgPPORewNoVelZ,
    Go1FlatCfgRewNoTorque,
    Go1FlatCfgPPORewNoTorque,
    Go1FlatCfgRewNoAngVel,
    Go1FlatCfgPPORewNoAngVel,
    Go1FlatCfgRewNoLinVel,
    Go1FlatCfgPPORewNoLinVel,
    Go1CfgRewNoActionRate,
    Go1CfgPPORewNoActionRate,
    Go1CfgRewNoAngVelXY,
    Go1CfgPPORewNoAngVelXY,
    Go1CfgRewNoCollision,
    Go1CfgPPORewNoCollision,
    Go1CfgRewNoDofAcc,
    Go1CfgPPORewNoDofAcc,
    Go1CfgRewNoAirTime,
    Go1CfgPPORewNoAirTime,
    Go1CfgRewNoVelZ,
    Go1CfgPPORewNoVelZ,
    Go1CfgRewNoTorque,
    Go1CfgPPORewNoTorque,
    Go1CfgRewNoAngVel,
    Go1CfgPPORewNoAngVel,
    Go1CfgRewNoLinVel,
    Go1CfgPPORewNoLinVel,
)

import os

from legged_gym.utils.task_registry import task_registry


task_registry.register("anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO())
task_registry.register("anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO())
task_registry.register("anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO())
task_registry.register("a1_rough", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
task_registry.register("a1_flat", LeggedRobot, A1FlatCfg(), A1FlatCfgPPO())
task_registry.register("cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO())
task_registry.register("go1", LeggedRobot, Go1Cfg(), Go1CfgPPO())
task_registry.register("go1_flat", LeggedRobot, Go1FlatCfg(), Go1FlatCfgPPO())

task_registry.register("pupper_flat", LeggedRobot, PupperFlatCfg(), PupperFlatCfgPPO())
task_registry.register("pupper_flat_high_leg_mass", LeggedRobot, PupperHighLegMassCfg(), PupperFlatHighLegMassCfgPPO())
task_registry.register("pupper_flat_a1_mass", LeggedRobot, PupperA1MassCfg(), PupperA1MassCfgPPO())
task_registry.register("pupper_flat_go1_mass", LeggedRobot, PupperGo1MassCfg(), PupperGo1MassCfgPPO())

task_registry.register("go1_flat_repeat1", LeggedRobot, Go1FlatCfg1(), Go1FlatCfgPPO1())
task_registry.register("go1_flat_repeat2", LeggedRobot, Go1FlatCfg2(), Go1FlatCfgPPO2())
task_registry.register("go1_flat_repeat3", LeggedRobot, Go1FlatCfg3(), Go1FlatCfgPPO3())
task_registry.register("go1_flat_repeat4", LeggedRobot, Go1FlatCfg4(), Go1FlatCfgPPO4())
task_registry.register("go1_flat_repeat5", LeggedRobot, Go1FlatCfg5(), Go1FlatCfgPPO5())
task_registry.register("go1_flat_repeat6", LeggedRobot, Go1FlatCfg6(), Go1FlatCfgPPO6())
task_registry.register("go1_flat_ctrl_vel", LeggedRobot, Go1FlatCfgV(), Go1FlatCfgPPOV())
task_registry.register("go1_flat_ctrl_torque", LeggedRobot, Go1FlatCfgT(), Go1FlatCfgPPOT())
task_registry.register("go1_flat_cmd_curriculum", LeggedRobot, Go1FlatCfgCmdCurr(), Go1FlatCfgPPOCmdCurr())
task_registry.register("go1_cmd_curriculum", LeggedRobot, Go1CfgCmdCurr(), Go1CfgPPOCmdCurr())
task_registry.register("go1_no_curriculum", LeggedRobot, Go1CfgNoCurr(), Go1CfgPPONoCurr())
task_registry.register("go1_terrain_curriculum", LeggedRobot, Go1CfgTerrainCurr(), Go1CfgPPOTerrainCurr())
task_registry.register("go1_both_curriculum", LeggedRobot, Go1CfgBothCurr(), Go1CfgPPOBothCurr())

task_registry.register("go1_flat_rew_no_action_rate", LeggedRobot, Go1FlatCfgRewNoActionRate(), Go1FlatCfgPPORewNoActionRate())
task_registry.register("go1_flat_rew_no_ang_vel_xy", LeggedRobot, Go1FlatCfgRewNoAngVelXY(), Go1FlatCfgPPORewNoAngVelXY())
task_registry.register("go1_flat_rew_no_collision", LeggedRobot, Go1FlatCfgRewNoCollision(), Go1FlatCfgPPORewNoCollision())
task_registry.register("go1_flat_rew_no_dof_acc", LeggedRobot, Go1FlatCfgRewNoDofAcc(), Go1FlatCfgPPORewNoDofAcc())
task_registry.register("go1_flat_rew_no_air_time", LeggedRobot, Go1FlatCfgRewNoAirTime(), Go1FlatCfgPPORewNoAirTime())
task_registry.register("go1_flat_rew_no_vel_z", LeggedRobot, Go1FlatCfgRewNoVelZ(), Go1FlatCfgPPORewNoVelZ())
task_registry.register("go1_flat_rew_no_torque", LeggedRobot, Go1FlatCfgRewNoTorque(), Go1FlatCfgPPORewNoTorque())
task_registry.register("go1_flat_rew_no_ang_vel", LeggedRobot, Go1FlatCfgRewNoAngVel(), Go1FlatCfgPPORewNoAngVel())
task_registry.register("go1_flat_rew_no_lin_vel", LeggedRobot, Go1FlatCfgRewNoLinVel(), Go1FlatCfgPPORewNoLinVel())

task_registry.register("go1_rew_no_action_rate", LeggedRobot, Go1CfgRewNoActionRate(), Go1CfgPPORewNoActionRate())
task_registry.register("go1_rew_no_ang_vel_xy", LeggedRobot, Go1CfgRewNoAngVelXY(), Go1CfgPPORewNoAngVelXY())
task_registry.register("go1_rew_no_collision", LeggedRobot, Go1CfgRewNoCollision(), Go1CfgPPORewNoCollision())
task_registry.register("go1_rew_no_dof_acc", LeggedRobot, Go1CfgRewNoDofAcc(), Go1CfgPPORewNoDofAcc())
task_registry.register("go1_rew_no_air_time", LeggedRobot, Go1CfgRewNoAirTime(), Go1CfgPPORewNoAirTime())
task_registry.register("go1_rew_no_vel_z", LeggedRobot, Go1CfgRewNoVelZ(), Go1CfgPPORewNoVelZ())
task_registry.register("go1_rew_no_torque", LeggedRobot, Go1CfgRewNoTorque(), Go1CfgPPORewNoTorque())
task_registry.register("go1_rew_no_ang_vel", LeggedRobot, Go1CfgRewNoAngVel(), Go1CfgPPORewNoAngVel())
task_registry.register("go1_rew_no_lin_vel", LeggedRobot, Go1CfgRewNoLinVel(), Go1CfgPPORewNoLinVel())

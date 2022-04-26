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

import os
from datetime import datetime
from typing import Tuple, Any, Type

import torch


from legged_gym import LEGGED_GYM_ROOT_DIR
from .helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    get_load_path,
    set_seed,
    parse_sim_params,
    flatten,
)
from legged_gym.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

from typing import TYPE_CHECKING

from rsl_rl.runners import OnPolicyRunner

if TYPE_CHECKING:
    from legged_gym.base.legged_robot import LeggedRobot

WANDB_PROJ = "legged-gym"


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(
        self,
        name: str,
        task_class: Type["LeggedRobot"],
        env_cfg: LeggedRobotCfg,
        train_cfg: LeggedRobotCfgPPO,
    ):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> Type["LeggedRobot"]:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(
        self, name, args=None, env_cfg=None, ppo_config=None, test=False
    ) -> Tuple["LeggedRobot", LeggedRobotCfg, Any, Any]:
        """Creates an environment either from a registered name or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name'

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        print(f"== Making env {name} based on config {type(env_cfg)}")

        if args is None:
            args = get_args()

        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, ppo_config = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, ppo_config = update_cfg_from_args(env_cfg, ppo_config, args)

        train_cfg_dict = class_to_dict(ppo_config)
        env_args_dict = class_to_dict(env_cfg)
        if not test:
            train_cfg_dict["env"] = env_args_dict
            import wandb

            wandb.init(
                project=WANDB_PROJ,
                config=flatten(train_cfg_dict),
                group=train_cfg_dict["runner"]["experiment_name"],
            )
        else:
            wandb = None

        if args.sweep:
            assert not test
            # FIXME: this snippet assumes 2 GPUs available. This is Flo-specific. =========== START
            # for gpu in range(2):
            #     print(gpu, torch.cuda.list_gpu_processes(torch.device(f"cuda:{gpu}")))

            # FIXME: why are they inverted?
            device1_free = "no processes are running" in torch.cuda.list_gpu_processes(torch.device("cuda:0"))
            device0_free = "no processes are running" in torch.cuda.list_gpu_processes(torch.device("cuda:1"))

            assert device0_free or device1_free
            if device0_free:
                sim_device = "cuda:0"
                rl_device = "cuda:0"
            else:
                sim_device = "cuda:1"
                rl_device = "cuda:1"
            print(f"== Using {sim_device}")
            # FIXME: =========== END

            # Access all hyperparameter values through wandb.config
            config = wandb.config

            for reward_prop in [
                "tracking_lin_vel_arm",
                "termination",
                "tracking_lin_vel",
                "tracking_ang_vel",
                "lin_vel_z",
                "ang_vel_xy",
                "orientation",
                "torques",
                "dof_vel",
                "dof_acc",
                "dof_pos_limits" "base_height",
                "feet_air_time",
                "collision",
                "feet_stumble",
                "action_rate",
            ]:
                prop_name = f"reward_{reward_prop}"  # it's called reward_termination in the sweep yaml file
                if prop_name in config:
                    setattr(env_cfg.rewards.scales, reward_prop, config[prop_name])

        else:
            sim_device = args.sim_device
            rl_device = args.rl_device

        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=sim_device,
            headless=args.headless,
            use_viewer=args.use_viewer,
        )
        return env, env_cfg, rl_device, wandb

    def make_alg_runner(
        self,
        env: "LeggedRobot",
        name=None,
        args=None,
        train_cfg: "LeggedRobotCfgPPO" = None,
        log_root="default",
        wandb=None,
        rl_device=None,
        obs_dim=None,
        act_dim=None,
    ) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """Creates the training algorithm  either from a registered name or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example).
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.
            obs_from_cfg (bool, optional): default: false. If true, use observation space of config, not environment.
                                        Useful for running different train/test envs.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file

        """
        print(f"== Making PPO trainer for env {name} based on config {type(train_cfg)}")
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%y%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
            )
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%y%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
            )

        print(f"supposed logdir: {log_dir}")

        train_cfg_dict = class_to_dict(train_cfg)
        # env_args_dict = class_to_dict(env.cfg)

        extra_args = {}
        if obs_dim is not None:
            # this is used when train env != test env
            extra_args["obs_override"] = obs_dim
        if act_dim is not None:
            extra_args["act_override"] = act_dim

        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=rl_device, wandb=wandb, **extra_args)
        # save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg


# make global task registry
task_registry = TaskRegistry()

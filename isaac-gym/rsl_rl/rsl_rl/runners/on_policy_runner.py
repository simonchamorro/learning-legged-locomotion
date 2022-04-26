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
import pathlib
import time
import os
from collections import deque
import statistics

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic
from legged_gym.utils import to_torch_device


class OnPolicyRunner:
    def __init__(
        self,
        env: VecEnv,
        train_cfg,
        log_dir=None,
        device="cpu",
        wandb=None,
        obs_override=None,
        act_override=None,
    ):

        self.wandb = wandb
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        obs_dim_actor = self.env.num_obs
        obs_dim_critic = num_critic_obs
        act_dim = self.env.num_actions
        if obs_override is not None:
            obs_dim_actor = obs_override
            obs_dim_critic = obs_override
        if act_override is not None:
            act_dim = act_override

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        print(
            f"== Creating ActorCritic '{actor_critic_class}' with actor_obs_dim {obs_dim_actor}, critic_obs_dim {obs_dim_critic}, action_dim {act_dim}."
        )
        actor_critic: ActorCritic = actor_critic_class(obs_dim_actor, obs_dim_critic, act_dim, **self.policy_cfg).to(
            self.device
        )
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.env.rollouts_per_env,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            None,
        )

        # Log
        self.log_dir = log_dir
        if log_dir is not None:
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=False)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        obs = to_torch_device(obs, self.device)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        print("== total iterations:", tot_iter)
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                from collections import defaultdict

                all_critic_obs = defaultdict(list)
                self.alg.storage.rollout_idx = 0
                for rollouts_idx in range(self.env.rollouts_per_env):
                    for i in range(self.num_steps_per_env):
                        actions = self.alg.act(obs, critic_obs)
                        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)

                        critic_obs = privileged_obs if privileged_obs is not None else obs
                        obs, critic_obs, rewards, dones = (
                            to_torch_device(obs, device=self.device),
                            to_torch_device(critic_obs, device=self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )
                        self.alg.process_env_step(rewards, dones, infos)

                        if self.log_dir is not None:
                            # Book keeping
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            cur_reward_sum += rewards
                            cur_episode_length += 1
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0
                    for k, v in critic_obs.items():
                        all_critic_obs[k].append(v)
                    self.alg.storage.increment_rollout_idx()

                processed_critic_obs = {}
                for k, v in all_critic_obs.items():
                    processed_critic_obs[k] = torch.cat(v)
                self.alg.compute_returns(processed_critic_obs)
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs * self.env.rollouts_per_env
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        wandb_out = {
            "Loss/value_function": locs["mean_value_loss"],
            "Loss/surrogate": locs["mean_surrogate_loss"],
            "Loss/learning_rate": self.alg.learning_rate,
            "Policy/mean_noise_std": mean_std.item(),
            "Perf/total_fps": fps,
            "Perf/collection time": locs["collection_time"],
            "Perf/learning_time": locs["learn_time"],
        }

        if len(locs["rewbuffer"]) > 0:
            wandb_out.update(
                {
                    "Train/mean_reward": statistics.mean(locs["rewbuffer"]),
                    "Train/mean_episode_length": statistics.mean(locs["lenbuffer"]),
                    # "Train/mean_reward/time": statistics.mean(locs["rewbuffer"]),
                    # "Train/mean_episode_length/time": statistics.mean(locs["lenbuffer"]),
                }
            )

        ep_string = f""
        if locs["ep_infos"]:
            correct_behavior = 0
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_out["Episode/" + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                if key in [
                    "rew_tracking_lin_vel",
                    "rew_tracking_lin_vel_arm",
                    "rew_tracking_ang_vel",
                ]:
                    correct_behavior += value
            wandb_out["Episode/correct_behavior"] = correct_behavior

        progress = (
            f" \033[1m Learning iteration {locs['it']}/"
            f"{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        )

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{progress.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{progress.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)
        if self.wandb is not None:
            self.wandb.log(wandb_out, step=locs["it"])
            
    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

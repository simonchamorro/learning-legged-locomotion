
import pickle
import wandb
from tqdm import tqdm


KEYS = ["Episode/correct_behavior",
        "Loss/value_function",
        "Loss/surrogate",
        "Train/mean_episode_length",
        "Train/mean_reward"]

RUN_ID = {
    "go1": "simonchamorro/legged-gym/35khcfo9",
    "go1_flat": "simonchamorro/legged-gym/3qteyvba",
    "go1_rew_no_lin_vel": "simonchamorro/legged-gym/22bxhpag",
    "go1_rew_no_ang_vel": "simonchamorro/legged-gym/nfd4pl68",
    "go1_rew_no_torque": "simonchamorro/legged-gym/18fr3vnx",
    "go1_rew_no_vel_z": "simonchamorro/legged-gym/ezch29v2",
    "go1_rew_no_air_time": "simonchamorro/legged-gym/310bra8y",
    "go1_rew_no_dof_acc": "simonchamorro/legged-gym/3jxxi62l",
    "go1_rew_no_collision": "simonchamorro/legged-gym/xh08aekn",
    "go1_rew_no_ang_vel_xy": "simonchamorro/legged-gym/yavg0p7x",
    "go1_rew_no_action_rate": "simonchamorro/legged-gym/2y6aw9ml",
    "go1_flat_rew_no_lin_vel": "simonchamorro/legged-gym/2yp6fq5m",
    "go1_flat_rew_no_ang_vel": "simonchamorro/legged-gym/17d8olw4",
    "go1_flat_rew_no_torque": "simonchamorro/legged-gym/12nyx7ht",
    "go1_flat_rew_no_vel_z": "simonchamorro/legged-gym/23kxu54v",
    "go1_flat_rew_no_air_time": "simonchamorro/legged-gym/29g005r5",
    "go1_flat_rew_no_dof_acc": "simonchamorro/legged-gym/1hf6fp4m",
    "go1_flat_rew_no_collision": "simonchamorro/legged-gym/21thuhc7",
    "go1_flat_rew_no_ang_vel_xy": "simonchamorro/legged-gym/p64dcyjz",
    "go1_flat_rew_no_action_rate": "simonchamorro/legged-gym/2v0v9mrl",
    "go1_cmd_terrain_curriculum": "simonchamorro/legged-gym/1ch76gb3",
    "go1_flat_fixed_init_state": "simonchamorro/legged-gym/1q5xyqy7",
    "go1_flat_torque1.2": "simonchamorro/legged-gym/setzjsmu",
    "go1_flat_torque0.8": "simonchamorro/legged-gym/2qmfr9dy",
    "go1_flat_torque0.6": "simonchamorro/legged-gym/1vp1t79t",
    "go1_flat_repeat1": "simonchamorro/legged-gym/rgjzefv9",
    "go1_flat_repeat2": "simonchamorro/legged-gym/3fyoyt5t",
    "go1_flat_repeat3": "simonchamorro/legged-gym/o6kv69nw",
    "go1_flat_repeat4": "simonchamorro/legged-gym/1lwhlnuv",
    "go1_flat_repeat5": "simonchamorro/legged-gym/14cad74f",
    "go1_flat_repeat6": "simonchamorro/legged-gym/28jjy3k3",
    "go1_flat_cmd_curriculum": "simonchamorro/legged-gym/3iqbey8p",
    "go1_cmd_curriculum": "simonchamorro/legged-gym/8o5he99v",
    "go1_no_curriculum": "simonchamorro/legged-gym/1aut13hf",
    "go1_terrain_curriculum": "simonchamorro/legged-gym/36sm8h9j"
}

api = wandb.Api()

all_runs = {}
for name, run_id in tqdm(RUN_ID.items()):
    run = api.run(run_id)
    run_history = run.scan_history()
    run_dict = {}
    for key in KEYS:
        run_dict[key] = [row[key] for row in run_history]

    all_runs[name] = run_dict

# Save to picle
with open("all_runs.pkl", "wb") as f:
    pickle.dump(all_runs, f)
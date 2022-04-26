
import pickle
import wandb
from tqdm import tqdm
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_data(api, run_id, label):
    run = api.run(run_id)
    run_history = run.scan_history()
    data = [row[label] for row in run_history]
    return data


def plot_from_wandb(runs, key, title, x_label, y_label, file_name):
    y_datas = []
    x_datas = []
    run_labels = []
    for label, run_id in runs.items():
        y_datas.append(load_data(api, run_id, key))
        x_datas.append(np.arange(0, len(y_datas[-1])))
        run_labels.append(label)

    build_plot(title, x_label, y_label, x_datas, y_datas, run_labels, file_name=file_name, eps=False, to_img=True) 
    

def smooth_data(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def build_plot(title, x_label, y_label, x_datas, y_datas, run_labels, to_img=False, colab=False, eps=True, file_name=''):
    """ Generate plots for the experiments.

    title: the title of the plot (str)

    x_label: the name for the x-axis label (str)
    y_label: the name for the y-axis label (str)

    x_datas: a list of list which contains the x values for the runs (list of list[float])
    y_datas: a list of list which contains the y values for the runs (list of list[float])

    run_labels: a list containing the name of each run (list[str])

    to_img: convert the plot into a png image (bool)
    file_name: name of the file that will be saved (str)
    colab: set to true if using on google colab (bool)
    """

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    assert len(x_datas) == len(y_datas), 'x_datas need to have the number of run than y_datas'
    assert (len(x_datas) == len(run_labels)) or  len(run_labels) == 0, 'you need to provide all the run labels or none of them !'

    figure(figsize=(12, 10), dpi=100)
    
    # Add title
    plt.title(title, fontsize=40, pad=30)

    # Add the grid
    plt.grid(True, linestyle=':', alpha=0.9)

    # Add x and y label
    plt.xlabel(x_label, fontsize=30)
    plt.ylabel(y_label, fontsize=30)

    # Plot data
    for idx, (x, y) in enumerate(zip(x_datas, y_datas)):

        if len(run_labels) == 0:
            label = ''
        else:
            label = run_labels[idx]
        y = smooth_data(y, 5)
        plt.plot(x[:-2], y[:-2], label=label, alpha=0.7)

    # Add the legend
    if len(run_labels) >= 1:
        plt.legend(loc='upper left')

    # Save plot
    if to_img:
        if eps:
            end_file = 'eps'
        else:
            end_file = 'png'
            filename = str(file_name)+"."+end_file
            plt.savefig(filename,bbox_inches='tight', format=end_file)
        
        if colab:
            from google.colab import files
            files.download(filename)
    else:
        return plt



if __name__ == '__main__':  
    api = wandb.Api()
    
    # Action repeat
    print("Action repeat")
    key = "Train/mean_reward"
    runs = {"action_repeat=1": "simonchamorro/legged-gym/rgjzefv9",
            "action_repeat=2": "simonchamorro/legged-gym/3fyoyt5t",
            "action_repeat=3": "simonchamorro/legged-gym/o6kv69nw",
            "action_repeat=4": "simonchamorro/legged-gym/1lwhlnuv",
            "action_repeat=5": "simonchamorro/legged-gym/14cad74f",
            "action_repeat=6": "simonchamorro/legged-gym/28jjy3k3"}
    
    title = "Action Repeat on Flat Terrain"
    x_label = "Iterations"
    y_label = "Reward"
    file_name = "action-repeat"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name)
    
    
    # Curriculum learning
    print("Curriculum learning")
    key = "Train/mean_reward"
    runs = {"both curriculums": "simonchamorro/legged-gym/1ch76gb3",
            "cmd curriculum": "simonchamorro/legged-gym/8o5he99v",
            "no curriculum": "simonchamorro/legged-gym/1aut13hf",
            "terrain curriculum": "simonchamorro/legged-gym/36sm8h9j"}
    
    title = "Curriculum Learning on Rough Terrain"
    x_label = "Iterations"
    y_label = "Reward"
    file_name = "curriculum"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name)
    
    
    # Torque Reward
    print("Torque Reward")
    key = "Train/mean_reward"
    runs = {"torque=1.2": "simonchamorro/legged-gym/setzjsmu",
            "torque=0.8": "simonchamorro/legged-gym/2qmfr9dy",
            "torque=0.6": "simonchamorro/legged-gym/1vp1t79t",
            "torque=1.0": "simonchamorro/legged-gym/3qteyvba",}
    
    title = "Torque Limits Reward"
    x_label = "Iterations"
    y_label = "Reward"
    file_name = "torque-reward"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name)
    
    
    # Torque Reward
    print("Torque Reward")
    key = "Episode/rew_lin_vel_z"
    runs = {"torque=1.2": "simonchamorro/legged-gym/setzjsmu",
            "torque=0.8": "simonchamorro/legged-gym/2qmfr9dy",
            "torque=0.6": "simonchamorro/legged-gym/1vp1t79t",
            "torque=1.0": "simonchamorro/legged-gym/3qteyvba",}
    
    title = "Torque Limits Velocity Z"
    x_label = "Iterations"
    y_label = "Velocity on Z Axis"
    file_name = "torque-vel-z"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name)  
    
    
    # Baseline
    print("Baseline")
    key = "Train/mean_reward"
    runs = {"go1_flat": "simonchamorro/legged-gym/3qteyvba",
            "a1_flat": "simonchamorro/legged-gym/3habeu8t",
            'pupper_flat': "simonchamorro/legged-gym/2dqswztn",
            'pupper_flat_go1_mass': "simonchamorro/legged-gym/3te980i2"}
    
    title = "Comparing to Baseline: A1 Robot"
    x_label = "Iterations"
    y_label = "Reward"
    file_name = "baseline"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name)   
    
    
    # Reward vs. Iterations
    print("Reward 1")
    key = "Train/mean_reward"
    runs = {"baseline": "simonchamorro/legged-gym/3qteyvba",
            "no_lin_vel": "simonchamorro/legged-gym/2yp6fq5m",
            "no_ang_vel": "simonchamorro/legged-gym/17d8olw4",
            "no_torque": "simonchamorro/legged-gym/12nyx7ht",
            "no_vel_z": "simonchamorro/legged-gym/23kxu54v",
            "no_air_time": "simonchamorro/legged-gym/29g005r5",
            "no_dof_acc": "simonchamorro/legged-gym/1hf6fp4m",
            "no_collision": "simonchamorro/legged-gym/21thuhc7",
            "no_ang_vel_xy": "simonchamorro/legged-gym/p64dcyjz",
            "no_action_rate": "simonchamorro/legged-gym/2v0v9mrl"}
    
    title = "Train Reward"
    x_label = "Iterations"
    y_label = "Reward"
    file_name = "reward-flat-reward"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name)  
    
    print("Reward 2")
    key = "Train/mean_reward"
    runs = {"baseline": "simonchamorro/legged-gym/35khcfo9",
            "no_lin_vel": "simonchamorro/legged-gym/22bxhpag",
            "no_ang_vel": "simonchamorro/legged-gym/nfd4pl68",
            "no_torque": "simonchamorro/legged-gym/18fr3vnx",
            "no_vel_z": "simonchamorro/legged-gym/ezch29v2",
            "no_air_time": "simonchamorro/legged-gym/310bra8y",
            "no_dof_acc": "simonchamorro/legged-gym/3jxxi62l",
            "no_collision": "simonchamorro/legged-gym/xh08aekn",
            "no_ang_vel_xy": "simonchamorro/legged-gym/yavg0p7x",
            "no_action_rate": "simonchamorro/legged-gym/2y6aw9ml",}
    
    title = "Train Reward on Rough Terrain"
    x_label = "Iterations"
    y_label = "Episode Length"
    file_name = "rewad-rough-reward"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name) 
    
    print("Reward 3")
    key = "Train/mean_episode_length"
    runs = {"baseline": "simonchamorro/legged-gym/3qteyvba",
            "no_lin_vel": "simonchamorro/legged-gym/2yp6fq5m",
            "no_ang_vel": "simonchamorro/legged-gym/17d8olw4",
            "no_torque": "simonchamorro/legged-gym/12nyx7ht",
            "no_vel_z": "simonchamorro/legged-gym/23kxu54v",
            "no_air_time": "simonchamorro/legged-gym/29g005r5",
            "no_dof_acc": "simonchamorro/legged-gym/1hf6fp4m",
            "no_collision": "simonchamorro/legged-gym/21thuhc7",
            "no_ang_vel_xy": "simonchamorro/legged-gym/p64dcyjz",
            "no_action_rate": "simonchamorro/legged-gym/2v0v9mrl"}
    
    title = "Mean Episode Length"
    x_label = "Iterations"
    y_label = "Reward"
    file_name = "reward-flat-length"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name)  
    
    print("Reward 4")
    key = "Train/mean_episode_length"
    runs = {"baseline": "simonchamorro/legged-gym/35khcfo9",
            "no_lin_vel": "simonchamorro/legged-gym/22bxhpag",
            "no_ang_vel": "simonchamorro/legged-gym/nfd4pl68",
            "no_torque": "simonchamorro/legged-gym/18fr3vnx",
            "no_vel_z": "simonchamorro/legged-gym/ezch29v2",
            "no_air_time": "simonchamorro/legged-gym/310bra8y",
            "no_dof_acc": "simonchamorro/legged-gym/3jxxi62l",
            "no_collision": "simonchamorro/legged-gym/xh08aekn",
            "no_ang_vel_xy": "simonchamorro/legged-gym/yavg0p7x",
            "no_action_rate": "simonchamorro/legged-gym/2y6aw9ml",}
    
    title = "Mean Episode Length on Rough Terrain"
    x_label = "Iterations"
    y_label = "Episode Length"
    file_name = "rewad-rough-length"
    plot_from_wandb(runs, key, title, x_label, y_label, file_name) 
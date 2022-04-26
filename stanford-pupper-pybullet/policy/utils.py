"""
Overview: Utils class to handle policy related stuff.
"""

# Import
import glob
import torch

from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO

def get_policy(path, obs_dim=48, act_dim=12, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128]):

    """
    Re-use policy learned from ISAAC Gym.

    Parameters:
    -----------
    path: relative path to the .pt weight file (str)
    obs_dim: size of the observations vector (int)
    act_dim: size of the actions vector (int)
    actor_hidden_dims: size of each layer of the actor neural network (list)
    critic_hidden_dims: size of each layer of the critic neural network (list)
    """
    obs_dim_actor = obs_dim
    obs_dim_critic = obs_dim

    print("Loading policy ...")

    device = torch.device('cpu')

    loaded_dict = torch.load(path, map_location=device)

    actor_critic = ActorCritic(obs_dim_actor, obs_dim_critic, act_dim, actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims).to(device)
    alg = PPO(actor_critic, device=device)

    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    current_learning_iteration = loaded_dict["iter"]
    actor_critic.eval()  # switch to evaluation mode (dropout for example)
    actor_critic.to(device)

    print("Policy loaded !")

    return actor_critic.act_inference#self.alg.actor_critic.act_inference

def extract_last_model_save(path):
    """
    Extract the last saved model containing 'model.pt' in its name
    from the given path
    """
    models = [file for file in glob.glob(path) if "model" in file]
    return models[-1]

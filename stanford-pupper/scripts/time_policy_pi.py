import numpy as np
import time
import glob
import os

import torch
import torch.nn as nn
import torch.functional as F



def main():
    """Main program
    """

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        
    print("Loading policy ...")
    policy = nn.Sequential(nn.Linear(48, 512),
                       nn.ELU(),
                       nn.Linear(512, 256),
                       nn.ELU(),
                       nn.Linear(256, 128),
                       nn.ELU(),
                       nn.Linear(128, 12))

    time_list = []
    num_it = 10000
    for i in range(num_it):
        obs = torch.randn(48).to(device)
        start_time = time.time()
        out = policy(obs)
        time_list.append(time.time() - start_time)

    print("average inference time: " + str(sum(time_list) / num_it))


if __name__ == "__main__":
    main()
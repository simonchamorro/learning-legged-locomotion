# Installation

1. Create a new python virtual env with python 3.8
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install other requirements 
    - `pip install -r requirements.txt`
4. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
5. Install rsl_rl (PPO implementation)
   - `cd isaac_gym/rsl_rl && pip install -e .` 
6. Install legged_gym
   - `cd isaac-gym/legged_gym && pip install -e .`

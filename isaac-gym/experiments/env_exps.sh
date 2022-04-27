
# Action repeat
python legged_gym/scripts/train.py --task=go1_flat_repeat1
python legged_gym/scripts/train.py --task=go1_flat_repeat2
python legged_gym/scripts/train.py --task=go1_flat_repeat3
python legged_gym/scripts/train.py --task=go1_flat_repeat4
python legged_gym/scripts/train.py --task=go1_flat_repeat5
python legged_gym/scripts/train.py --task=go1_flat_repeat6

# Control type
python legged_gym/scripts/train.py --task=go1_flat_ctrl_vel
python legged_gym/scripts/train.py --task=go1_flat_ctrl_torque

# Curriculum
python legged_gym/scripts/train.py --task=go1_flat_cmd_curriculum
python legged_gym/scripts/train.py --task=go1_cmd_curriculum
python legged_gym/scripts/train.py --task=go1_no_curriculum
python legged_gym/scripts/train.py --task=go1_terrain_curriculum
python legged_gym/scripts/train.py --task=go1_both_curriculum



# Reproduce results from paper
python legged_gym/legged_gym/scripts/train.py --task=a1_flat
python legged_gym/legged_gym/scripts/train.py --task=a1_rough
python legged_gym/legged_gym/scripts/train.py --task=anymal_c_flat
python legged_gym/legged_gym/scripts/train.py --task=cassie
python legged_gym/legged_gym/scripts/train.py --task=anymal_c_rough
python legged_gym/legged_gym/scripts/train.py --task=anymal_b

# Go1 flat
python legged_gym/legged_gym/scripts/train.py --task=go1_flat

# Pupper experiments
python legged_gym/legged_gym/scripts/train.py --task=pupper_flat_a1_mass
python legged_gym/legged_gym/scripts/train.py --task=pupper_flat_high_leg_mass
python legged_gym/legged_gym/scripts/train.py --task=pupper_flat



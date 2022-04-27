# learning-legged-locomotion
Learning Legged Locomotion project for the course Robot Learning - IFT6163

Learning an end-to-end locomotion policy for a quadruped robot in the real world is a difficult problem, mainly because of its complexity, the sample inefficiency of learning algorithms, and the ability to deal with uncertainty. 
However, Sim2Real methods try to tackle this problem by learning in a simulated environment, where data is easy to generate and where collisions do not cause any real damages. 
In this report,we aim to make a cheap and flimsy robot, i.e. the Stanford Pupper, learn to walk in the massively parallelizable simulator Isaac Gym using curriculum learning and a hand-crafted reward function and transfer this policy to the real robot. 
We use PPO to learn our locomotion policy in simulation. 
Finally, we transfer our learned policy to a real physical robot and qualitatively analyse its performance. 
Additionally, we also propose an Isaac Gym compatible model of the Stanford Pupper as well as the Unitree Go1. 
Our work also contains an ablation study of reward terms and a study of different environment parameters to understand the learning process better.

This repository is divided into three sections: Isaac-Gym related code, code for the Stanford Pupper in the PyBullet simulator, and code for the real Stanford Pupper. 
See [INSTALLATION.md](INSTALLATION.md) for installation instructions.

## Isaac Gym

## Stanford Pupper in PyBullet

## Stanford Pupper

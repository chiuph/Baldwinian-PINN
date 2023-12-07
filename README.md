# Baldwinian-PINN

[![python >3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/) 

## **Generalizable Neural Physics Solvers by Baldwinian Evolution**

__Jian Cheng Wong, Chin Chun Ooi, Abhishek Gupta, Pao-Hsiung Chiu, Joshua Shao Zheng Low, My Ha Dao, Yew-Soon Ong__



_Physics-informed neural networks (PINNs) are at the forefront of scientific machine learning, making possible the creation of machine intelligence that is cognizant of physical laws and able to accurately simulate them. In this paper, the potential of discovering PINNs that generalize over an entire family of physics tasks is studied, for the first time, through a biological lens of the Baldwin effect. Drawing inspiration from the neurodevelopment of precocial species that have evolved to learn, predict and react quickly to their environment, we envision PINNs that are pre-wired with connection strengths inducing strong biases towards efficient learning of physics. To this end, evolutionary selection pressure (guided by proficiency over a family of tasks) is coupled with lifetime learning (to specialize on a smaller subset of those tasks) to produce PINNs that demonstrate fast and physics-compliant prediction capabilities across a range of empirically challenging problem instances. The Baldwinian approach achieves an order of magnitude improvement in prediction accuracy at a fraction of the computation cost compared to state-of-the-art results with PINNs meta-learned by gradient descent. This paper marks a leap forward in the meta-learning of PINNs as generalizable physics solvers._

# Install
[![jax-0.3.23](https://img.shields.io/badge/jax-0.3.23-yellowgreen)](https://github.com/google/jax) [![evojax-0.2.15](https://img.shields.io/badge/evojax-0.2.15-orange)](https://github.com/google/evojax) [![evosax-0.0.9](https://img.shields.io/badge/evosax-0.0.9-red)](https://github.com/RobertTLange/evosax) [![flax-0.6.1](https://img.shields.io/badge/flax-0.6.1-lightgrey)](https://github.com/google/flax) [![optax-0.1.3](https://img.shields.io/badge/optax-0.1.3-blue)](https://github.com/google-deepmind/optax) [![numpy-1.24.4](https://img.shields.io/badge/numpy-1.24.4-green)](https://github.com/numpy/numpy) [![pandas-1.4.4](https://img.shields.io/badge/pandas-1.4.4-yellow)](https://github.com/pandas-dev/pandas) [![matplotlib-3.5.2](https://img.shields.io/badge/matplotlib-3.5.2-purple)](https://github.com/matplotlib/matplotlib)

# Usage
You can download this repo and run the demo tasks on your computing machine.

# Time cost
Typical install time should be under 30min for all the above packages.

Expected time required for training and run across all tasks in the provided scripts should be under 4 hours on a workstation with 2 GPUs (Intel Xeon W-2275 and 2 Nvidia GeForce RTX 3090). Individual tasks might take as little as 10 mins.

# Coypright

This tool is developed in Fluid Dynamics department, Institute of High Performance Computing (IHPC), A*STAR.

The copyright holder for this project is Fluid Dynamics department, Institute of High Performance Computing (IHPC), A*STAR.

All rights reserved.

# Citation

Jian Cheng Wong, Chin Chun Ooi, Abhishek Gupta, Pao-Hsiung Chiu, Joshua Shao Zheng Low, My Ha Dao, Yew-Soon Ong, _"Generalizable Neural Physics Solvers by Baldwinian Evolution"_. https://arxiv.org/abs/2312.03243

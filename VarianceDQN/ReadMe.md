Run the code with command:
python train.py GameName Method

Method must be DDQN/VDQN/TDDQN.

An example:
python train.py CrazyClimber VDQN

Tensorboard log file is in "./log/" and can be accessed with "tensorboard --logdir=./log/"
Another copy is saved in csv format in "./result/"

Dependencies:
Numpy
Pytorch
Gym
Opencv-python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import yaml

from easydict import EasyDict

config = EasyDict()
config.NUM_GPUS = 1
config.OUTPUT_DIR= ""

config.DATALOADER = EasyDict()
config.DATALOADER.WORKERS = 2
config.DATALOADER.SHUFFLE = True
config.DATALOADER.LIDAR_DATA_PATH = "/home/fatih/LidarLabelsCameraViewTest"
config.DATALOADER.CAMERA_DATA_PATH = "/home/fatih/SegmentedInputTest"

config.LIDAR_GENERATOR = EasyDict()
# Base Learning rate for optimizer
config.LIDAR_GENERATOR.BASE_LR = 0.0006
# Change learning rate in each step_size number of iterations by multiplying it with gamma
config.LIDAR_GENERATOR.STEP_SIZE = 5
config.LIDAR_GENERATOR.STEP_GAMMA = 0.1

config.LIDAR_DISCRIMINATOR = EasyDict()
config.LIDAR_DISCRIMINATOR.BASE_LR = 0.0001
config.LIDAR_DISCRIMINATOR.STEP_SIZE = 5
config.LIDAR_DISCRIMINATOR.STEP_GAMMA = 0.1

config.CAMERA_GENERATOR = EasyDict()
config.CAMERA_GENERATOR.BASE_LR = 0.0001
config.CAMERA_GENERATOR.STEP_SIZE = 5
config.CAMERA_GENERATOR.STEP_GAMMA = 0.1

config.CAMERA_DISCRIMINATOR = EasyDict()
config.CAMERA_DISCRIMINATOR.BASE_LR = 0.0001
config.CAMERA_DISCRIMINATOR.STEP_SIZE = 5
config.CAMERA_DISCRIMINATOR.STEP_GAMMA = 0.1

config.TRAIN = EasyDict()
config.TRAIN.BATCH_SIZE = 64
config.TRAIN.START_EPOCH = 0
config.TRAIN.MAX_EPOCH = 1000
config.TRAIN.LOAD_WEIGHTS = ""
config.TRAIN.SAVE_WEIGHTS = ""
config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION = "mean"
config.TRAIN.CYCLE_LOSS_REDUCTION = "mean"
config.TRAIN.EXAMPLE_SAVE_PATH = ""
config.TRAIN.GRAPH_SAVE_PATH = ""
config.TRAIN.SAVE_AT = 2
config.TRAIN.LAMBDA_GP = 10
config.TRAIN.BETA1 = 0.9
config.TRAIN.BETA2 = 0.999
config.CAMERA_TO_LIDAR = True


def fix_the_type(desired_type, given_type):
    if type(desired_type) == type(given_type):
        return given_type
    elif isinstance(desired_type, bool):
        return bool(given_type)
    elif isinstance(desired_type, int):
        return int(given_type)
    elif isinstance(desired_type, float):
        return float(given_type)


def update_config_secondaries(left, right):
    for i,k in right.items():
        new_k = fix_the_type(config[left][i],k)
        config[left][i] = new_k


def load_config(config_file):
    if config_file is not None:
        with open(config_file,"r") as f:
            my_config = EasyDict(yaml.load(f,  Loader=yaml.BaseLoader) )

            for i, k in my_config.items():

                if i in config:
                    if isinstance(k, EasyDict):
                        update_config_secondaries(i,k)
                    else:
                        new_k = fix_the_type(config[i],k)
                        config[i] = new_k
                else:
                    raise ValueError(i, " is not one of the core variables")



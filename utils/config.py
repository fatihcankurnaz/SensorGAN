from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import yaml

from easydict import EasyDict

config = EasyDict()
config.NUM_GPUS = 2
config.OUTPUT_DIR= ""

config.DATALOADER = EasyDict()
config.DATALOADER.WORKERS = 2
config.DATALOADER.SHUFFLE = True
config.DATALOADER.SENSOR1_PATH = ""
config.DATALOADER.SENSOR2_PATH = ""

config.SENSOR1_GENERATOR = EasyDict()
config.SENSOR1_GENERATOR.BASE_LR = 0.0006

config.SENSOR1_DISCRIMINATOR = EasyDict()
config.SENSOR1_DISCRIMINATOR.BASE_LR = 0.0001

config.SENSOR2_GENERATOR = EasyDict()
config.SENSOR2_GENERATOR.BASE_LR = 0.0001

config.SENSOR2_DISCRIMINATOR = EasyDict()
config.SENSOR2_DISCRIMINATOR.BASE_LR = 0.0001

config.TRAIN = EasyDict()
config.TRAIN.BATCH_SIZE = 64
config.TRAIN.START_EPOCH = 0
config.TRAIN.MAX_EPOCH = 1000
config.TRAIN.LOAD_WEIGHTS = ""
config.TRAIN.SAVE_WEIGHTS = ""
config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION = "mean"
config.TRAIN.CYCLE_LOSS_REDUCTION = "mean"


def update_config_secondaries(left, right):
    for i,k in right.items():
        print(i)
        config[left][i] = k


def load_config(config_file):
    if config_file is not None:
        with open(config_file,"r") as f:
            my_config = EasyDict(yaml.load(f))

            for i, k in my_config.items():

                if i in config:
                    if isinstance(k, EasyDict):
                        update_config_secondaries(i,k)
                    else:
                        config[i] = k
                else:
                    raise ValueError(i, " is not one of the config variables")



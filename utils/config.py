from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import yaml

import numpy as np
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
config.SENSOR1_GENERATOR.BASE_LR = 0.0001

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

def load_config(config_file):
    with open(config_file,"r") as f:
        
        
        my_config = yaml.load(file(f, 'r'))

        for i in my_config:
            if i in config:
                config.i = my_config[i]
            else:
                raise ValueError(i," is not one of the config variables")



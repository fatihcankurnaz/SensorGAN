from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import yaml
from easydict import EasyDict

config = EasyDict()
config.NUM_GPUS = [0,1]
config.OUTPUT_DIR = "./output"
config.MODEL = "baseline"


config.DATALOADER = EasyDict()
config.DATALOADER.WORKERS = 1
config.DATALOADER.SHUFFLE = True
config.DATALOADER.LIDAR_DATA_PATH = "/home/tiago/Documents/LidarLabelsCameraViewTest"
config.DATALOADER.CAMERA_DATA_PATH = "/home/tiago/Documents/SegmentedInputTest"
config.DATALOADER.PIX2PIX = False
config.DATALOADER.RGB_PATH = "/SPACE/DATA/KITTI_Data/KITTI_raw_data/kitti/2011_09_26"
config.DATALOADER.SEGMENTED_PATH = "/home/tiago/Documents/Inputs/CameraData"

config.LIDAR_GENERATOR = EasyDict()
# Base Learning rate for optimizer
config.LIDAR_GENERATOR.BASE_LR = 0.0006
# Change learning rate in each step_size number of iterations by multiplying it with gamma
config.LIDAR_GENERATOR.STEP_SIZE = 5
config.LIDAR_GENERATOR.STEP_GAMMA = 0.1
config.LIDAR_GENERATOR.PIXEL_LAMBDA = 0.2

config.LIDAR_DISCRIMINATOR = EasyDict()
config.LIDAR_DISCRIMINATOR.BASE_LR = 0.0001
config.LIDAR_DISCRIMINATOR.STEP_SIZE = 5
config.LIDAR_DISCRIMINATOR.STEP_GAMMA = 0.1

config.CAMERA_GENERATOR = EasyDict()
config.CAMERA_GENERATOR.BASE_LR = 0.0001
config.CAMERA_GENERATOR.STEP_SIZE = 5
config.CAMERA_GENERATOR.STEP_GAMMA = 0.1
config.CAMERA_GENERATOR.PIXEL_LAMBDA = 0.2
config.CAMERA_GENERATOR.NEW_LOSS_LAMBDA = 0.2

config.CAMERA_DISCRIMINATOR = EasyDict()
config.CAMERA_DISCRIMINATOR.BASE_LR = 0.0001
config.CAMERA_DISCRIMINATOR.STEP_SIZE = 5
config.CAMERA_DISCRIMINATOR.STEP_GAMMA = 0.1

config.TRAIN = EasyDict()
config.TRAIN.BATCH_SIZE = 15
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
config.TRAIN.CYCLE_LAMBDA = 10
config.TRAIN.OUTPUT_FREQ = 5
config.TRAIN.SAVE_WEIGHTS = "/saved_models"
config.TRAIN.GRAPH_SAVE_PATH = "/plots"
config.TRAIN.EXAMPLE_SAVE_PATH = "/generated"


config.TEST = EasyDict()
config.TEST.LOAD_LIDAR_TO_CAM_WEIGHTS = "/home/tiago/Documents/my_git/sensorgan/outputs/saved_models/4.pth"
config.TEST.LOAD_PIX2PIX_WEIGHTS = "/home/tiago/Documents/my_git/sensorgan/outputs/pipx_yedk/normalized_0.9true_input1d_nonnormal_55"
config.TEST.LOAD_BASELINE_WEIGHTS = "/home/tiago/Documents/my_git/sensorgan/outputs/baseline_models/40.pth"
config.TEST.SEGMENTED_LIDAR_ROOT = "/home/tiago/Documents/Inputs/2011_09_26_drive_0046_sync_lid"
config.TEST.SEGMENTED_CAMERA_ROOT = "/home/tiago/Documents/Inputs/2011_09_26_drive_0046_sync_cam"
config.TEST.CLOUD_ROOT = "/SPACE/DATA/KITTI_Data/KITTI_labeledPC_with_BBs/2011_09_26/2011_09_26_drive_0046_sync"
config.TEST.RGB_ROOT = "/SPACE/DATA/KITTI_Data/KITTI_raw_data/kitti/2011_09_26/2011_09_26_drive_0046_sync/image_02/data"
config.TEST.RESULT_SAVE_PATH = "/home/tiago/Documents/my_git/sensorgan/outputs/eval_result"
config.TEST.INPUT_DIR = "/home/tiago/Documents/Inputs/TEST"
config.TEST.FILES = {}
config.TEST.EVAL_EPOCH = 2


def fix_the_type(desired_type, given_type):
    if type(desired_type) == type(given_type):
        return given_type
    elif isinstance(desired_type, bool):
        if (given_type == True):
            return True
        else:
            return False

    elif isinstance(desired_type, int):
        return int(given_type)
    elif isinstance(desired_type, float):
        return float(given_type)


def update_config_secondaries(left, right):
    for i, k in right.items():
        new_k = fix_the_type(config[left][i], k)
        config[left][i] = new_k


def load_config(config_file):
    if config_file is not None:
        with open(config_file, "r") as f:
            my_config = EasyDict(yaml.load(f, Loader=yaml.BaseLoader))

            for i, k in my_config.items():

                if i in config:
                    if isinstance(k, EasyDict):
                        update_config_secondaries(i, k)
                    else:
                        new_k = fix_the_type(config[i], k)
                        config[i] = new_k
                else:
                    raise ValueError(i, " is not one of the core variables")
        return my_config

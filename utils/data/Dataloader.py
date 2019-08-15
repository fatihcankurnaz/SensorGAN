from torch.utils.data import DataLoader
from utils.data.Dataset import LidarAndCameraDataset
import numpy as np


def lidar_camera_dataloader(config):

    # lidar_path = config.DATALOADER.LIDAR_DATA_PATH
    # camera_path = config.DATALOADER.CAMERA_DATA_PATH

    dataloader = DataLoader(LidarAndCameraDataset(config), batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=config.DATALOADER.SHUFFLE, num_workers=config.DATALOADER.WORKERS, pin_memory=True)

    return dataloader

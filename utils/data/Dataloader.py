from torch.utils.data import DataLoader
from utils.data.Dataset import LidarAndCameraDataset
import numpy as np
from torchvision import transforms

def lidar_camera_dataloader(config, transforms_):

    # lidar_path = config.DATALOADER.LIDAR_DATA_PATH
    # camera_path = config.DATALOADER.CAMERA_DATA_PATH

    dataloader = DataLoader(LidarAndCameraDataset(config, transforms_), batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=config.DATALOADER.SHUFFLE, num_workers=config.DATALOADER.WORKERS, pin_memory=True)

    return dataloader

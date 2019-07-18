from torch.utils.data import Dataset
from os import listdir
from os.path import join
import numpy as np


class LidarAndCameraDataset(Dataset):
    def __init__(self, lidar_path, camera_path):
        self.lidar_path = lidar_path
        self.camera_path = camera_path
        self.dirs = [dirs for dirs in listdir(self.lidar_path)]
        self.lidar_dataset = []
        self.camera_dataset = []
        for dirs in self.dirs:
            for file in sorted(listdir(join(self.lidar_path, dirs))):
                current_dir = join(self.lidar_path, dirs)
                self.lidar_dataset.append(join(current_dir, file))

            for file in sorted(listdir(join(self.camera_path, dirs))):
                current_dir = join(self.camera_path, dirs)
                self.camera_dataset.append(join(current_dir, file))

    def __getitem__(self, idx):
        lidar_data = np.load(self.lidar_dataset[idx])["data"].reshape(5,375,1242)
        camera_data = np.load(self.camera_dataset[idx])["data"].reshape(5,375,1242)
        return {"lidar_data":lidar_data, "camera_data":camera_data}

    def __len__(self):
        return len(self.lidar_dataset)
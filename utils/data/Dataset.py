from torch.utils.data import Dataset
from scipy import misc
from os import listdir
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform
from torchvision import transforms


class LidarAndCameraDataset(Dataset):
    def __init__(self, config ,transforms_=None, transform_seg=None):
        self.isPix2Pix = config.DATALOADER.PIX2PIX
        self.transform = transforms.Compose(transforms_)
        self.transform_seg = transforms.Compose(transform_seg)
        self.multip = np.ones((5,375,1242))
        self.multip[0] = self.multip[0] * 0
        self.multip[1] = self.multip[1] * 1
        self.multip[2] = self.multip[2] * 2
        self.multip[3] = self.multip[3] * 3
        self.multip[4] = self.multip[4] * 4
        if self.isPix2Pix is True:
            self.segmented_path = config.DATALOADER.SEGMENTED_PATH
            self.rgb_path = config.DATALOADER.RGB_PATH
            self.dirs = [dirs for dirs in listdir(self.segmented_path)]
            self.rgb_dataset = []
            self.segmented_dataset = []
            for dirs in self.dirs:
                for file in sorted(listdir(join(self.segmented_path, dirs))):
                    current_segmented_dir = join(self.segmented_path, dirs)
                    current_rgb_dir = join(self.rgb_path,dirs+"/image_02/data")
                    current_rgb = join(current_rgb_dir, file.split(".")[0].split("segmented_")[1] + ".png")
                    if current_rgb != "/SPACE/DATA/KITTI_Data/KITTI_raw_data/kitti/2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000102.png":
                        self.segmented_dataset.append(join(current_segmented_dir, file))
                        self.rgb_dataset.append(current_rgb)
        else:

            self.lidar_path = config.DATALOADER.LIDAR_DATA_PATH
            self.camera_path = config.DATALOADER.CAMERA_DATA_PATH
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
        ##print(idx, self.lidar_dataset[idx])
        if self.isPix2Pix is True:

            try:
                rgb_data = Image.open(self.rgb_dataset[idx])
                rgb_data = self.transform(rgb_data)
                #rgb_data = transforms.ToTensor()(rgb_data)
                segmented_data = np.load(self.segmented_dataset[idx])["data"].reshape(5, 375, 1242)
                segmented_data = segmented_data * self.multip
                segmented_data = np.sum(segmented_data, axis=0).reshape(375,1242)
                segmented_data = Image.fromarray(segmented_data, 'L')
                segmented_data = self.transform_seg(segmented_data)

            except:
                print("lel", idx, self.rgb_dataset[idx])

            return {"rgb_data": rgb_data, "segmented_data": segmented_data}

        else:
            try:
                lidar_data = np.load(self.lidar_dataset[idx])["data"].reshape(5,375,1242)
                camera_data = np.load(self.camera_dataset[idx])["data"].reshape(5,375,1242)
            except:
                print("lel",idx,self.lidar_dataset[idx])
            return {"lidar_data": lidar_data, "camera_data": camera_data}




    def __len__(self):
        if self.isPix2Pix is True:
            return len(self.segmented_dataset)
        else:
            return len(self.lidar_dataset)
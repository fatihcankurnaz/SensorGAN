from os import listdir
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LidarAndCameraDataset(Dataset):
    def __init__(self, config, transforms_=None, transform_seg=None):
        if transform_seg is None:
            transform_seg = [transforms.ToTensor()]
        if transforms_ is None:
            transforms_ = [transforms.ToTensor()]
        self.model = config.MODEL
        self.transform = transforms.Compose(transforms_)
        self.transform_seg = transforms.Compose(transform_seg)
        self.multip = np.ones((5, 375, 1242))
        self.config = config

        if self.model == "pix2pix":
            print("pix2pix dataset loading")
            self.multip[0] = self.multip[0] * 0
            self.multip[1] = self.multip[1] * 1
            self.multip[2] = self.multip[2] * 2
            self.multip[3] = self.multip[3] * 3
            self.multip[4] = self.multip[4] * 4
            self.segmented_path = config.DATALOADER.SEGMENTED_PATH
            self.rgb_path = config.DATALOADER.RGB_PATH
            self.dirs = [dirs for dirs in listdir(self.segmented_path)]
            self.rgb_dataset = []
            self.segmented_dataset = []
            for dirs in self.dirs:
                for file in sorted(listdir(join(self.segmented_path, dirs))):
                    current_segmented_dir = join(self.segmented_path, dirs)
                    current_rgb_dir = join(self.rgb_path, dirs + "/image_02/data")
                    current_rgb = join(current_rgb_dir, file.split(".")[0].split("segmented_")[1] + ".png")
                    if current_rgb != "/SPACE/DATA/KITTI_Data/KITTI_raw_data/kitti/2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000102.png":
                        self.segmented_dataset.append(join(current_segmented_dir, file))
                        self.rgb_dataset.append(current_rgb)

        elif self.model == "baseline":
            print("baseline dataset loading")
            self.multip[0] = self.multip[0] * 0
            self.multip[1] = self.multip[1] * 1
            self.multip[2] = self.multip[2] * 1
            self.multip[3] = self.multip[3] * 1
            self.multip[4] = self.multip[4] * 1
            self.segmented_path = config.DATALOADER.SEGMENTED_PATH
            self.rgb_path = config.DATALOADER.RGB_PATH
            self.dirs = [dirs for dirs in listdir(self.segmented_path)]
            self.rgb_dataset = []
            self.segmented_dataset = []
            for dirs in self.dirs:
                for file in sorted(listdir(join(self.segmented_path, dirs))):
                    current_segmented_dir = join(self.segmented_path, dirs)
                    current_rgb_dir = join(self.rgb_path, dirs + "/image_02/data")
                    current_rgb = join(current_rgb_dir, file.split(".")[0].split("cameraView_")[1] + ".png")
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
        if self.model == "pix2pix":

            try:
                rgb_data = Image.open(self.rgb_dataset[idx])
                rgb_data = self.transform(rgb_data)
                # rgb_data = transforms.ToTensor()(rgb_data)
                segmented_data = np.load(self.segmented_dataset[idx])["data"].reshape(5, 375, 1242)
                segmented_data = segmented_data * self.multip
                segmented_data = np.sum(segmented_data, axis=0).reshape(1, 375, 1242)
                return {"y": rgb_data, "x": segmented_data}


            except:
                print("lel", idx, self.rgb_dataset[idx])



        elif self.model == "baseline":

            try:
                rgb_data = Image.open(self.rgb_dataset[idx])
                rgb_data = transforms.ToTensor()(rgb_data)
                segmented_data = np.load(self.segmented_dataset[idx])["data"].reshape(5, 375, 1242)
                segmented_data = segmented_data * self.multip
                segmented_data = np.sum(segmented_data, axis=0).reshape(1, 375, 1242)
                return {"y": rgb_data, "x": segmented_data}



            except:
                print("lel", idx, self.rgb_dataset[idx])



        else:
            try:
                lidar_data = np.load(self.lidar_dataset[idx])["data"].reshape(5, 375, 1242)
                camera_data = np.load(self.camera_dataset[idx])["data"].reshape(5, 375, 1242)
                return {"x": lidar_data, "y": camera_data}
            except:
                print("lel", idx, self.lidar_dataset[idx])


    def __len__(self):
        if self.model == "pix2pix" or self.model == "baseline":
            return len(self.segmented_dataset)
        else:
            return len(self.lidar_dataset)

    def get_test(self):

        test1 = self.config.TEST.FILES['1']
        test12 = self.config.TEST.FILES['12']

        test2 = self.config.TEST.FILES['2']
        test22 = self.config.TEST.FILES['22']



        if self.model == 'baseline' or self.model == 'pix2pix':
            test1 = Image.open(test1)
            test2 = Image.open(test2)
            test1 = self.transform(test1)
            test2 = self.transform(test2)
            test1 = test1.type(torch.float).cuda()
            test2 = test2.type(torch.float).cuda()

            test1 = test1.view(1, 3, 375, 1242)
            test2 = test2.view(1, 3, 375, 1242)
            test12 = np.sum(np.load(test12)["data"].reshape(5, 375, 1242) * self.multip, axis=0). \
                reshape(375, 1242)

            test22 = np.sum(np.load(test22)["data"].reshape(5, 375, 1242) * self.multip, axis=0). \
                reshape(375, 1242)
            test12 = self.transform_seg(test12).view(1, 1, 375, 1242).type(torch.float).cuda()
            test22 = self.transform_seg(test22).view(1, 1, 375, 1242).type(torch.float).cuda()
            patch = (1, 375 // 2 ** 4, 1242 // 2 ** 4)
            label_real = torch.cuda.FloatTensor(np.ones((self.config.TRAIN.BATCH_SIZE, *patch)))
            label_fake = torch.cuda.FloatTensor(np.zeros((self.config.TRAIN.BATCH_SIZE, *patch)))
        else:
            test1 = torch.from_numpy(np.load(test1)["data"].reshape(1, 5, 375, 1242)).type(torch.float).cuda()
            test2 = torch.from_numpy(np.load(test2)["data"].reshape(1, 5, 375, 1242)).type(torch.float).cuda()
            test12 = torch.from_numpy(np.load(test12)["data"].reshape(1, 5, 375, 1242)).type(torch.float).cuda()
            test22 = torch.from_numpy(np.load(test22)["data"].reshape(1, 5, 375, 1242)).type(torch.float).cuda()
            label_real = torch.cuda.FloatTensor(np.ones((self.config.TRAIN.BATCH_SIZE, 1, 23, 77)))
            label_fake = torch.cuda.FloatTensor(np.zeros((self.config.TRAIN.BATCH_SIZE, 1, 23, 77)))

        return test1,test12,test2,test22,label_real,label_fake

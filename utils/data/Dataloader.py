from torch.utils.data import DataLoader

from utils.data.Dataset import LidarAndCameraDataset


def lidar_camera_dataloader(config, transforms_=None, transform_seg=None):

    dataset = LidarAndCameraDataset(config, transforms_, transform_seg)
    dataloader = DataLoader(dataset,
                            batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=config.DATALOADER.SHUFFLE, num_workers=config.DATALOADER.WORKERS, pin_memory=True,
                            drop_last=True)

    return dataloader, dataset

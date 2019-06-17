from torch.utils.data import DataLoader
from data.dataset import CombinedDataset
import numpy as np


def DataLoader(config):
    sensor1 = np.load(config.DATALOADER.SENSOR1_PATH)
    sensor2 = np.load(config.DATALOADER.SENSOR2_PATH)

    dataloader = DataLoader(CombinedDataset(sensor1, sensor2), batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=config.DATALOADER.SHUFFLE, num_workers=config.DATALOADER.WORKERS)

    return dataloader

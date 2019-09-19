from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import optparse
import os
import time
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import utils.scripts.pointCloudScripts.PC2ImageConverter as PC2ImageConverter
from PIL import Image
from torchvision.utils import save_image
from utils.scripts.pointCloudScripts.runKITTIDataGeneratorForObjectDataset import processData
from models.Generator import Generator

parser = optparse.OptionParser()
colors = ['white', 'green', 'yellow', 'red', 'blue']
parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")


def turn_back_to_oneD(data):
    torch_version = torch.from_numpy(data).view(1, 5, -1, 1242)
    new_version = torch.max(torch_version, dim=1)[1].view(-1, 1242)

    return new_version.numpy()


def baseline_eval(config):
    PC2ImgConv = PC2ImageConverter.PC2ImgConverter(imgChannel=5, xRange=[0, 25], yRange=[-6, 12], zRange=[-10, 8],
                                                   xGridSize=0.1, yGridSize=0.15, zGridSize=0.3, maxImgHeight=128,
                                                   maxImgWidth=256, maxImgDepth=64)

    input_dir = config.TEST.INPUT_DIR






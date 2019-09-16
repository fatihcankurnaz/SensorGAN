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

from configs.config import *


torch.manual_seed(0)
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
    if len (config.NUM_GPUS) != 0:
        baseline_model = Generator(1, 3, config.NUM_GPUS).cuda()

        if len(config.NUM_GPUS) > 1:
            baseline_model = nn.DataParallel(baseline_model, config.NUM_GPUS)
    else:
        baseline_model = Generator(1, 3, config.NUM_GPUS)


    print("loading previous baseline model")
    baseline_checkpoint = torch.load(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.SAVE_WEIGHTS+'/training_sensor_gen_{}.pth'.format(str(config.TEST.EVAL_EPOCH)))
    baseline_model.load_state_dict(baseline_checkpoint)
    baseline_model.eval()
    print("done")
    multip = np.ones((5, 375, 1242))
    multip[0] = multip[0] * 0
    multip[1] = multip[1] * 1
    multip[2] = multip[2] * 1
    multip[3] = multip[3] * 1
    multip[4] = multip[4] * 1
    if not os.path.exists(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH):
        os.makedirs(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH)
    if not os.path.exists(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(config.TEST.EVAL_EPOCH)):
        os.makedirs(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(config.TEST.EVAL_EPOCH))
    image_count = 0
    for dir in sorted(listdir(input_dir)):
        segmented_lidar_root = join(input_dir, dir + "/lidar")
        rgb_root = join(config.TEST.RGB_ROOT, '')
        cloud_root = join(config.TEST.CLOUD_ROOT, '')
        for file in sorted(listdir(segmented_lidar_root)):
            print(file)

            segmented_lidar_path = join(segmented_lidar_root, file)
            segmented_lidar_numpy = np.load(segmented_lidar_path)["data"].reshape(5, 375, 1242) * multip
            segmented_1D = np.sum(segmented_lidar_numpy, axis=0).reshape(375, 1242)
            segmented_lidar_torch = torch.from_numpy(segmented_1D).type(torch.float).cuda(). \
                view(1, 1, 375, 1242)

            expected_rgb_path = join(rgb_root, file.split(".")[0].split("_")[1] + ".png")
            expected_rgb = Image.open(expected_rgb_path)

            cloud_path = join(cloud_root, "full_label_2011_09_26_0046_" + file.split(".")[0].split("_")[1] + ".npy")
            processData(cloud_path, "", PC2ImgConv, config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(config.TEST.EVAL_EPOCH) + "/cloud.png")
            cloud = Image.open(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(config.TEST.EVAL_EPOCH) + "/cloud.png")
            cloud = cloud.crop((0, 300, 1400, 800))

            baseline_model_start = time.time()
            generated_rgb = baseline_model(segmented_lidar_torch)
            baseline_model_end = time.time()

            generated_rgb = generated_rgb.detach().cpu()
            save_image(generated_rgb, config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(config.TEST.EVAL_EPOCH) + "/generated_rgb.png",
                       normalize=True)

            baseline_model_elapsed_time = baseline_model_end - baseline_model_start

            print("Baseline model took " + str(baseline_model_elapsed_time) + " ms ")

            generated_rgb = Image.open(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(config.TEST.EVAL_EPOCH) + "/generated_rgb.png")

            fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace=0.1, wspace=0.1)

            plt.subplot(2, 2, 1)
            plt.title("Given Lidar", fontdict={'fontsize': 15})
            plt.imshow(cloud)
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.title("Lidar", fontdict={'fontsize': 15})
            plt.imshow(segmented_1D)
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.title("Expected RGB", fontdict={'fontsize': 15})
            plt.imshow(expected_rgb)
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.title("Generated RGB", fontdict={'fontsize': 15})
            plt.imshow(generated_rgb)
            plt.axis("off")
            plt.savefig(config.OUTPUT_DIR + '/' + config.MODEL + config.TRAIN.EXAMPLE_SAVE_PATH + '/' + str(config.TEST.EVAL_EPOCH) + "/" + str(image_count) + "eval.png")

            plt.close()
            image_count += 1
            del segmented_lidar_torch


def main(opts):
    load_config(opts.config)

    baseline_eval(config)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)

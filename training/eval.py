from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import optparse

from utils.data.Dataloader import lidar_camera_dataloader
from utils.core.config import config

from utils.core.config import load_config
from utils.helpers.helpers import save_vanilla_model
from utils.helpers.helpers import display_two_images


from Generator import GeneratorLowParameter, Generator
from os import listdir
from os.path import join, isdir
import time

import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage import color
from utils.helpers.runKITTIDataGeneratorForObjectDataset import processData
import utils.helpers.PC2ImageConverter as PC2ImageConverter
from utils.helpers.visualizer import  Vis
torch.manual_seed(0)
parser = optparse.OptionParser()
colors = ['black', 'green', 'yellow', 'red', 'blue']
parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")


def turn_back_to_oneD(data):
    torch_version = torch.from_numpy(data).view(1, 5, -1, 1242)
    new_version = torch.max(torch_version, dim=1)[1].view(-1,1242)

    return new_version.numpy()


segmented_lidar_root = "/home/fatih/Inputs/2011_09_26_drive_0046_sync_lid"
segmented_camera_root = "/home/fatih/Inputs/2011_09_26_drive_0046_sync_cam"
cloud_root = "/SPACE/DATA/KITTI_Data/KITTI_labeledPC_with_BBs/2011_09_26/2011_09_26_drive_0046_sync"
rgb_root = "/SPACE/DATA/KITTI_Data/KITTI_raw_data/kitti/2011_09_26/2011_09_26_drive_0046_sync/image_02/data"

def eval(config, device):
    PC2ImgConv = PC2ImageConverter.PC2ImgConverter(imgChannel=5, xRange=[0, 25], yRange=[-6, 12], zRange=[-10, 8],
                                                   xGridSize=0.1, yGridSize=0.15, zGridSize=0.3, maxImgHeight=128,
                                                   maxImgWidth=256, maxImgDepth=64)






    real_lidar_seg_path = "/home/fatih/Inputs/test/46cameraView_0000000000.npz"
    real_camera_seg_path = "/home/fatih/Inputs/test/46segmented_0000000000.npz"
    real_camera_seg = turn_back_to_oneD(np.load(real_camera_seg_path)["data"])
    real_lidar_seg = turn_back_to_oneD(np.load(real_lidar_seg_path)["data"])




    real_image_gen = Generator(1, 3, config.NUM_GPUS).to(device)
    segment_gen = Generator(5, 5, config.NUM_GPUS).to(device)

    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        real_image_gen = nn.DataParallel(real_image_gen, list(range(config.NUM_GPUS)))
        segment_gen = nn.DataParallel(segment_gen, list(range(config.NUM_GPUS)))

    print("loading previous pix2pix model")
    real_image_checkpoint = torch.load(config.TEST.LOAD_PIX2PIX_WEIGHTS)
    real_image_gen.load_state_dict(real_image_checkpoint['sensor1_gen'])
    real_image_gen.eval()
    print("done")

    print("loading previous lidar_to_cam model")
    segment_checkpoint = torch.load(config.TEST.LOAD_LIDAR_TO_CAM_WEIGHTS)
    segment_gen.load_state_dict(segment_checkpoint['sensor1_gen'])
    segment_gen.eval()
    print("done")


    for file in sorted(listdir("/home/fatih/Inputs/2011_09_26_drive_0046_sync_lid")):
        print(file)
        segmented_lidar_path = join(segmented_lidar_root, file)
        segmented_lidar_numpy = np.load(segmented_lidar_path)["data"].reshape(5, 375, 1242)
        segmented_lidar = turn_back_to_oneD(segmented_lidar_numpy)
        segmented_lidar_torch = torch.from_numpy(segmented_lidar_numpy).to(device=device, dtype=torch.float).\
            view(1, 5, 375, 1242)

        expected_camera_segment_path = join(segmented_camera_root, "segmented_"+file.split("_")[1])
        expected_camera_segment = turn_back_to_oneD(np.load(expected_camera_segment_path)["data"].reshape(5, 375, 1242))

        expected_rgb_path = join(rgb_root, file.split(".")[0].split("_")[1]+ ".png")
        expected_rgb = Image.open(expected_rgb_path)


        cloud_path = join(cloud_root,"full_label_2011_09_26_0046_"+ file.split(".")[0].split("_")[1]+".npy")
        processData(cloud_path, "", PC2ImgConv, "/home/fatih/my_git/sensorgan/outputs/eval_result/cloud.png")
        cloud = Image.open("/home/fatih/my_git/sensorgan/outputs/eval_result/cloud.png")
        cloud = cloud.crop((0, 300, 1400, 800))
        segment_generator_time_start = time.time()
        generated_cam_segment = segment_gen(segmented_lidar_torch)
        segment_generator_time_end = time.time()
        generated_cam_segment = generated_cam_segment.detach().cpu().numpy()
        generated_cam_segment = turn_back_to_oneD(generated_cam_segment)
        generated_rgb_input = torch.from_numpy(generated_cam_segment).to(device=device,
                                                                         dtype=torch.float).view(1, 1, 375, 1242)
        rgb_generator_time_start = time.time()
        generated_rgb = real_image_gen(generated_rgb_input)
        rgb_generator_time_end = time.time()
        segment_generator_time_elapsed = segment_generator_time_end - segment_generator_time_start
        rgb_generator_time_elapsed = rgb_generator_time_end - rgb_generator_time_start
        print("segment generator took " + str(segment_generator_time_elapsed) + " ms ")
        print("rgb generator took " + str(rgb_generator_time_elapsed) + " ms ")
        print("total time " + str(rgb_generator_time_elapsed+ segment_generator_time_elapsed) + " ms ")
        generated_rgb = generated_rgb.detach().cpu()
        save_image(generated_rgb, "/home/fatih/my_git/sensorgan/outputs/eval_result/generated_rgb.png", normalize=True)



        generated_rgb = Image.open("/home/fatih/my_git/sensorgan/outputs/eval_result/generated_rgb.png")

        fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        plt.subplot(3, 2, 1)
        plt.title("Given Lidar",fontdict={'fontsize': 15 })
        plt.imshow(cloud)
        plt.axis("off")

        plt.subplot(3, 2, 2)
        plt.title("Lidar Segmentation",fontdict={'fontsize': 15 })
        plt.imshow(color.label2rgb(segmented_lidar, colors=colors))
        plt.axis("off")

        plt.subplot(3, 2, 3)
        plt.title("Expected Segmentation",fontdict={'fontsize': 15 })
        plt.imshow(color.label2rgb(expected_camera_segment, colors=colors))
        plt.axis("off")

        plt.subplot(3, 2, 4)
        plt.title("Expected RGB",fontdict={'fontsize': 15 })
        plt.imshow(expected_rgb)
        plt.axis("off")

        plt.subplot(3, 2, 5)
        plt.title("Generated Segmentation",fontdict={'fontsize': 15 })
        plt.imshow(color.label2rgb(generated_cam_segment, colors=colors))
        plt.axis("off")

        plt.subplot(3, 2, 6)
        plt.title("Generated RGB",fontdict={'fontsize': 15 })
        plt.imshow(generated_rgb)
        plt.axis("off")

        plt.savefig("/home/fatih/my_git/sensorgan/outputs/eval_result/"+file.split(".")[0].split("_")[1]+"eval.png")

        plt.close()







def main(opts):

    load_config(opts.config)


    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    eval(config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)


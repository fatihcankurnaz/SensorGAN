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


def eval(config, device):
    PC2ImgConv = PC2ImageConverter.PC2ImgConverter(imgChannel=5, xRange=[0, 25], yRange=[-6, 12], zRange=[-10, 8],
                                                   xGridSize=0.1, yGridSize=0.15, zGridSize=0.3, maxImgHeight=128,
                                                   maxImgWidth=256, maxImgDepth=64)


    cloud_path = "/SPACE/DATA/KITTI_Data/KITTI_labeledPC_with_BBs/2011_09_26/2011_09_26_drive_0046_sync/full_label_2011_09_26_0046_0000000000.npy"

    processData(cloud_path, "", PC2ImgConv, "/home/fatih/my_git/sensorgan/outputs/eval_result/cloud.png")

    real_lidar_seg_path = "/home/fatih/Inputs/test/46cameraView_0000000000.npz"
    real_camera_seg_path = "/home/fatih/Inputs/test/46segmented_0000000000.npz"
    real_camera_seg = turn_back_to_oneD(np.load(real_camera_seg_path)["data"])
    real_lidar_seg = turn_back_to_oneD(np.load(real_lidar_seg_path)["data"])

    cloud = Image.open("/home/fatih/my_git/sensorgan/outputs/eval_result/cloud.png")
    cloud = cloud.crop((0, 300, 1400, 800))
    print(cloud)
    print(cloud.size)

    real_image_gen = Generator(1, 3, config.NUM_GPUS).to(device)

    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        camera_gen = nn.DataParallel(real_image_gen, list(range(config.NUM_GPUS)))

    print("loading previous model")
    checkpoint = torch.load(config.TRAIN.LOAD_WEIGHTS)
    real_image_gen.load_state_dict(checkpoint['sensor1_gen'])
    real_image_gen.eval()
    print("done")





    for i in range(1, 100):

        test_segmented_path1 = "/home/fatih/my_git/sensorgan/outputs/examples/"+str(i)+"_generated_camera_1.npz"




        test_segmented1np = np.load(test_segmented_path1)["data"].reshape(5, 375, 1242)
        test1 = turn_back_to_oneD(test_segmented1np)

        test_segmented1 = torch.from_numpy(test1).to(device=device, dtype=torch.float).view(1,1,375,1242)

        output1 = real_image_gen(test_segmented1)


        output1 = output1.detach().cpu()

        save_image(output1, "/home/fatih/my_git/sensorgan/outputs/eval_result/output1.png", normalize=True)

        real_image = Image.open("/SPACE/DATA/KITTI_Data/KITTI_raw_data/kitti/2011_09_26/2011_09_26_drive_0046_sync/image_02/data/0000000000.png")

        output1 = Image.open("/home/fatih/my_git/sensorgan/outputs/eval_result/output1.png")

        fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        plt.subplot(3, 2, 1)
        plt.title("Given Lidar",fontdict={'fontsize': 15 })
        plt.imshow(cloud)
        plt.axis("off")

        plt.subplot(3, 2, 2)
        plt.title("Lidar Segmentation",fontdict={'fontsize': 15 })
        plt.imshow(color.label2rgb(real_lidar_seg, colors=colors))
        plt.axis("off")

        plt.subplot(3, 2, 3)
        plt.title("Expected Segmentation",fontdict={'fontsize': 15 })
        plt.imshow(color.label2rgb(real_camera_seg, colors=colors))
        plt.axis("off")

        plt.subplot(3, 2, 4)
        plt.title("Expected RGB",fontdict={'fontsize': 15 })
        plt.imshow(real_image)
        plt.axis("off")

        plt.subplot(3, 2, 5)
        plt.title("Generated Segmentation",fontdict={'fontsize': 15 })
        plt.imshow(color.label2rgb(turn_back_to_oneD(test_segmented1np),
                                   colors=colors))
        plt.axis("off")

        plt.subplot(3, 2, 6)
        plt.title("Generated RGB",fontdict={'fontsize': 15 })
        plt.imshow(output1)
        plt.axis("off")

        plt.savefig("/home/fatih/my_git/sensorgan/outputs/eval_result/"+str(i)+"eval.png")

        plt.close()







def main(opts):

    load_config(opts.config)


    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    eval(config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)


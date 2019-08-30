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


from utils.models.Generator import GeneratorLowParameter, Generator
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
colors = ['white', 'green', 'yellow', 'red', 'blue']
parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")


def turn_back_to_oneD(data):
    torch_version = torch.from_numpy(data).view(1, 5, -1, 1242)
    new_version = torch.max(torch_version, dim=1)[1].view(-1,1242)

    return new_version.numpy()




def baseline_eval(config, device):
    PC2ImgConv = PC2ImageConverter.PC2ImgConverter(imgChannel=5, xRange=[0, 25], yRange=[-6, 12], zRange=[-10, 8],
                                                   xGridSize=0.1, yGridSize=0.15, zGridSize=0.3, maxImgHeight=128,
                                                   maxImgWidth=256, maxImgDepth=64)

    input_dir = config.TEST.INPUT_DIR

    baseline_model = Generator(1, 3, config.NUM_GPUS).to(device)


    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        baseline_model = nn.DataParallel(baseline_model, list(range(config.NUM_GPUS)))

    print("loading previous baseline model")
    baseline_checkpoint = torch.load(config.TEST.LOAD_BASELINE_WEIGHTS)
    baseline_model.load_state_dict(baseline_checkpoint['sensor1_gen'])
    baseline_model.eval()
    print("done")
    multip = np.ones((5, 375, 1242))
    multip[0] = multip[0] * 0
    multip[1] = multip[1] * 1
    multip[2] = multip[2] * 1
    multip[3] = multip[3] * 1
    multip[4] = multip[4] * 1


    image_count = 0
    for dir in sorted(listdir(input_dir)):
        segmented_lidar_root = join(input_dir, dir+"/lidar")
        rgb_root = join(config.TEST.RGB_ROOT, dir + "/image_02/data")
        cloud_root = join(config.TEST.CLOUD_ROOT, dir)
        for file in sorted(listdir(segmented_lidar_root)):
            print(file)

            segmented_lidar_path = join(segmented_lidar_root, file)
            segmented_lidar_numpy = np.load(segmented_lidar_path)["data"].reshape(5, 375, 1242)*multip
            segmented_1D = np.sum(segmented_lidar_numpy, axis=0).reshape(375,1242)
            segmented_lidar_torch = torch.from_numpy(segmented_1D).to(device=device, dtype=torch.float).\
                view(1, 1, 375, 1242)

            expected_rgb_path = join(rgb_root, file.split(".")[0].split("_")[1]+ ".png")
            expected_rgb = Image.open(expected_rgb_path)


            cloud_path = join(cloud_root,"full_label_2011_09_26_0046_"+ file.split(".")[0].split("_")[1]+".npy")
            processData(cloud_path, "", PC2ImgConv, "/home/fatih/my_git/sensorgan/outputs/baseline_eval_result2/cloud.png")
            cloud = Image.open("/home/fatih/my_git/sensorgan/outputs/baseline_eval_result2/cloud.png")
            cloud = cloud.crop((0, 300, 1400, 800))

            baseline_model_start = time.time()
            generated_rgb = baseline_model(segmented_lidar_torch)
            baseline_model_end = time.time()

            generated_rgb = generated_rgb.detach().cpu()
            save_image(generated_rgb, "/home/fatih/my_git/sensorgan/outputs/baseline_eval_result2/generated_rgb.png",
                       normalize=True)



            baseline_model_elapsed_time = baseline_model_end - baseline_model_start

            print("Baseline model took " + str(baseline_model_elapsed_time) + " ms ")



            generated_rgb = Image.open("/home/fatih/my_git/sensorgan/outputs/baseline_eval_result2/generated_rgb.png")

            fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace=0.1, wspace=0.1)

            plt.subplot(2, 2, 1)
            plt.title("Given Lidar",fontdict={'fontsize': 15 })
            plt.imshow(cloud)
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.title("Lidar",fontdict={'fontsize': 15 })
            plt.imshow(segmented_1D)
            plt.axis("off")



            plt.subplot(2, 2, 3)
            plt.title("Expected RGB",fontdict={'fontsize': 15 })
            plt.imshow(expected_rgb)
            plt.axis("off")



            plt.subplot(2, 2, 4)
            plt.title("Generated RGB",fontdict={'fontsize': 15 })
            plt.imshow(generated_rgb)
            plt.axis("off")

            plt.savefig(config.TEST.RESULT_SAVE_PATH+"/"+str(image_count)+"eval.png")

            plt.close()
            image_count += 1
            del segmented_lidar_torch







def main(opts):

    load_config(opts.config)


    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    baseline_eval(config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)


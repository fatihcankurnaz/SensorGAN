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



    camera_gen = Generator(1, 3, config.NUM_GPUS).to(device)

    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        camera_gen = nn.DataParallel(camera_gen, list(range(config.NUM_GPUS)))

    print("loading previous model")
    checkpoint = torch.load(config.TRAIN.LOAD_WEIGHTS)
    camera_gen.load_state_dict(checkpoint['sensor1_gen'])
    camera_gen.eval()
    print("done")





    for i in range(1, 100):

        test_segmented_path1 = "/home/fatih/my_git/sensorgan/outputs/examples/"+str(i)+"_generated_camera_1.npz"
        test_segmented_path2 = "/home/fatih/my_git/sensorgan/outputs/examples/"+str(i)+"_generated_camera_2.npz"



        test_segmented1np = np.load(test_segmented_path1)["data"].reshape(5, 375, 1242)
        test1 = turn_back_to_oneD(test_segmented1np)

        test_segmented1 = torch.from_numpy(test1).to(device=device, dtype=torch.float).view(1,1,375,1242)


        test_segmented2np = np.load(test_segmented_path2)["data"].reshape(5, 375, 1242)
        test2 = turn_back_to_oneD(test_segmented2np)
        test_segmented2 = torch.from_numpy(test2).to(device=device, dtype=torch.float).view(1,1,375,1242)







        output1 = camera_gen(test_segmented1)
        output2 = camera_gen(test_segmented2)

        output1 = output1.detach().cpu()
        output2 = output2.detach().cpu()
        save_image(output1, "/home/fatih/my_git/sensorgan/outputs/eval_result/output1.png", normalize=True)
        save_image(output2, "/home/fatih/my_git/sensorgan/outputs/eval_result/output2.png", normalize=True)

        output1 = Image.open("/home/fatih/my_git/sensorgan/outputs/eval_result/output1.png")
        output2 = Image.open("/home/fatih/my_git/sensorgan/outputs/eval_result/output2.png")
        fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        plt.subplot(2, 2, 1)
        plt.title("Segmented 1")
        plt.imshow(color.label2rgb(turn_back_to_oneD(test_segmented1np),
                                   colors=colors))
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("Segmented 2")
        plt.imshow(color.label2rgb(turn_back_to_oneD(test_segmented2np),
                                   colors=colors))
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("RGB 1")
        plt.imshow(output1)
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("RGB 2")
        plt.imshow(output2)
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


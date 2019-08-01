import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import torch
from skimage import data, segmentation, color
from skimage.future import graph
import optparse

parser = optparse.OptionParser()

parser.add_option('-o', '--one', dest="num", action="store", type="int",
                  help="display only one output")
parser.add_option('-p', '--path', dest="path", action="store", type="string",
                  help="path to the files and no ending \"/\" ")
parser.add_option('-i', '--interval', dest="interval", action="store", type="string",
                  help="give an interval of numbers ex 1-14")
parser.add_option('-s', '--save', dest="save", action="store_true", default=False,
                  help="save the generated visual, if this is selected nothing will be shown")
parser.add_option('-v', '--verbose', dest="verbose", action="store_true", default=False,
                  help="print more information")
parser.add_option('-l', '--lidar_to_cam', dest="l_to_c", action="store_true", default=False,
                  help="whether output is lidar to cam or not")

#number = sys.argv[1]
#path = "/home/fatih/my_git/sensorgan/outputs/lidar_to_cam_examples/"

#
# If torch is not available
# def turn_back_to_oneD(data):
#
#     real_version = np.zeros((375, 1242))
#
#
#     for i in range(0, 374):
#         for j in range(0, 1241):
#             dim_val = -1
#             dim_type = -1
#             for dim in range(0, 5):
#                 if data[dim][i][j] > dim_val:
#                     dim_val = data[dim][i][j]
#                     dim_type = dim
#
#
#             real_version[i][j] = dim_type
#     return real_version

def turn_back_to_oneD(data, options):
    # applies argmax
    torch_version = torch.from_numpy(data).view(1, 5, -1, 1242)
    new_version = torch.max(torch_version, dim=1)[1].view(-1,1242)
    maxes = torch.max(torch_version, dim=1)[1].view(-1)
    count = 0
    # counts number of valid class points
    if options.verbose is True:
        for i in maxes:
            if i != 0:
                count+=1
        print(count)
    return new_version.numpy()


if __name__ == "__main__":
    options, args = parser.parse_args()
    print(options)
    path = options.path + "/"
    colors = ['black', 'green', 'yellow', 'red', 'blue']
    label = [0, 1, 2, 3, 4]

    if options.num is not None:
        start = options.num
        end = start+1
    elif options.interval is not None:
        start = int(options.interval.split("-")[0])
        end = int(options.interval.split("-")[1])
    else:
        start = 0
        end = 0
        print("Wrong input parameters")

    for number in range(start, end):
        print("Generating, ",number)
        generated = np.load(path + str(number) + "_generated_.npz")["data"]
        lidar = np.load(path + str(number) + "_lidar_.npz")["data"]
        camera = np.load(path + str(number) + "_camera_.npz")["data"]
        options, args = parser.parse_args()


        fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        if options.verbose is True:
            print("Lidar")
        if options.l_to_c is True:
            plt.subplot(3, 1, 1)
            plt.title("Given Lidar")
        else:
            plt.subplot(3, 1, 3)
            plt.title("Expected Lidar")
        plt.imshow(color.label2rgb(turn_back_to_oneD(lidar, options),
                                   colors=colors))
        plt.axis("off")


        if options.verbose is True:
            print("Camera")
        if options.l_to_c is True:
            plt.subplot(3, 1, 3)
            plt.title("Expected Camera")
        else:
            plt.subplot(3, 1, 1)
            plt.title("Given Camera")
        plt.imshow(color.label2rgb(turn_back_to_oneD(camera, options),
                                   colors=colors))
        plt.axis("off")


        if options.verbose is True:
            print("Generated")
        plt.subplot(3, 1, 2)
        plt.imshow(color.label2rgb(turn_back_to_oneD(generated, options),
                                   colors=colors))
        plt.axis("off")
        plt.title("Generated")
        if options.save is True:
            plt.savefig(path+str(number)+".png")
        else:
            plt.show()
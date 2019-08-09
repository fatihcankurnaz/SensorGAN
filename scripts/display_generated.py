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
parser.add_option('-r', '--reconstructed', dest="reconst", action="store_true", default=False,
                  help="reconstructed")



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
        if options.reconst is False:
            generated_lidar_1 = np.load(path + str(number) + "_generated_lidar_1.npz")["data"]
            generated_camera_1 = np.load(path + str(number) + "_generated_camera_1.npz")["data"]
            lidar1 = np.load(path + str(number) + "_lidar_1.npz")["data"]
            camera1 = np.load(path + str(number) + "_camera_1.npz")["data"]
            generated_lidar_2 = np.load(path + str(number) + "_generated_lidar_2.npz")["data"]
            generated_camera_2 = np.load(path + str(number) + "_generated_camera_2.npz")["data"]
            lidar2 = np.load(path + str(number) + "_lidar_2.npz")["data"]
            camera2 = np.load(path + str(number) + "_camera_2.npz")["data"]
        else:
            generated_lidar_1 = np.load(path + str(number) + "_reconstructed_lidar_1.npz")["data"]
            generated_camera_1 = np.load(path + str(number) + "_reconstructed_camera_1.npz")["data"]
            lidar1 = np.load(path + str(number) + "_lidar_1.npz")["data"]
            camera1 = np.load(path + str(number) + "_camera_1.npz")["data"]
            generated_lidar_2 = np.load(path + str(number) + "_reconstructed_lidar_2.npz")["data"]
            generated_camera_2 = np.load(path + str(number) + "_reconstructed_camera_2.npz")["data"]
            lidar2 = np.load(path + str(number) + "_lidar_2.npz")["data"]
            camera2 = np.load(path + str(number) + "_camera_2.npz")["data"]


        fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        if options.verbose is True:
            print("Lidar1")
        if options.l_to_c is True:
            plt.subplot(4, 2, 1)
            plt.title("Given Lidar1")
        else:
            plt.subplot(4, 2, 3)
            plt.title("Expected Lidar1")
        plt.imshow(color.label2rgb(turn_back_to_oneD(lidar1, options),
                                   colors=colors))
        plt.axis("off")


        if options.verbose is True:
            print("Camera1")
        if options.l_to_c is True:
            plt.subplot(4, 2, 3)
            plt.title("Expected Camera1")
        else:
            plt.subplot(4, 2, 1)
            plt.title("Given Camera1")
        plt.imshow(color.label2rgb(turn_back_to_oneD(camera1, options),
                                   colors=colors))
        plt.axis("off")


        if options.verbose is True:
            print("Generated Lidar 1")
        plt.subplot(4, 2, 5)
        plt.imshow(color.label2rgb(turn_back_to_oneD(generated_lidar_1, options),
                                   colors=colors))
        plt.axis("off")
        plt.title("Generated Lidar 1")

        if options.verbose is True:
            print("Generated Camera 1")
        plt.subplot(4, 2, 7)
        plt.imshow(color.label2rgb(turn_back_to_oneD(generated_camera_1, options),
                                   colors=colors))
        plt.axis("off")
        plt.title("Generated Camera 1")



        if options.verbose is True:
            print("Lidar2")
        if options.l_to_c is True:
            plt.subplot(4, 2, 2)
            plt.title("Given Lidar2")
        else:
            plt.subplot(4, 2, 4)
            plt.title("Expected Lidar2")
        plt.imshow(color.label2rgb(turn_back_to_oneD(lidar2, options),
                                   colors=colors))
        plt.axis("off")


        if options.verbose is True:
            print("Camera2")
        if options.l_to_c is True:
            plt.subplot(4, 2, 4)
            plt.title("Expected Camera2")
        else:
            plt.subplot(4, 2, 2)
            plt.title("Given Camera2")
        plt.imshow(color.label2rgb(turn_back_to_oneD(camera2, options),
                                   colors=colors))
        plt.axis("off")


        if options.verbose is True:
            print("Generated Lidar 2")
        plt.subplot(4, 2, 6)
        plt.imshow(color.label2rgb(turn_back_to_oneD(generated_lidar_2, options),
                                   colors=colors))
        plt.axis("off")
        plt.title("Generated Lidar 2")

        if options.verbose is True:
            print("Generated Camera 2")
        plt.subplot(4, 2, 8)
        plt.imshow(color.label2rgb(turn_back_to_oneD(generated_camera_2, options),
                                   colors=colors))
        plt.axis("off")
        plt.title("Generated Camera 2")

        if options.save is True:
            plt.savefig(path+str(number)+".png")
        else:
            plt.show()
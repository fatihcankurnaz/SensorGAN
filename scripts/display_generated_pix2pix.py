import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import torch
from skimage import data, segmentation, color
from skimage.future import graph
import optparse
from PIL import Image

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




def turn_back_to_oneD(data, options):
    # applies argmax
    torch_version = torch.from_numpy(data).view(1, 5, -1, 414)
    new_version = torch.max(torch_version, dim=1)[1].view(-1,414)
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
        generated_rgb_1 = Image.open(path + str(number) + "_generated_rgb_1.png")
        rgb1 = Image.open(path + str(number) + "_rgb_1.png")
        segmented1 = np.load(path + str(number) + "_segmented_1.npz")["data"][0]

        generated_rgb_2 = Image.open(path + str(number) + "_generated_rgb_2.png")
        rgb2 = Image.open(path + str(number) + "_rgb_2.png")
        segmented2 = np.load(path + str(number) + "_segmented_2.npz")["data"][0]



        fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        if options.verbose is True:
            print("Segmented 1")


        plt.subplot(3, 2, 1)
        plt.title("Given Segmented 1")
        plt.imshow(color.label2rgb(segmented1,
                                   colors=colors))
        plt.axis("off")


        if options.verbose is True:
            print("RGB1")

        plt.subplot(3, 2, 3)
        plt.title("Expected RGB 1")

        plt.imshow(rgb1)
        plt.axis("off")


        if options.verbose is True:
            print("Generated RGB 1")
        plt.subplot(3, 2, 5)
        plt.title("Generated RGB 1")
        plt.imshow(generated_rgb_1)
        plt.axis("off")


        if options.verbose is True:
            print("Segmented 2")

        plt.subplot(3, 2, 2)
        plt.title("Given Segmented 2")

        plt.imshow(color.label2rgb(segmented2,
                                   colors=colors))
        plt.axis("off")


        if options.verbose is True:
            print("RGB 2")

        plt.subplot(3, 2, 4)
        plt.title("Expected RGB 2")
        plt.imshow(rgb2)
        plt.axis("off")

        if options.verbose is True:
            print("Generated RGB 2")
        plt.subplot(3, 2, 6)
        plt.title("Generated RGB 2")
        plt.imshow(generated_rgb_2)
        plt.axis("off")


        if options.save is True:
            plt.savefig(path+"grouped/"+str(number)+".png")
            plt.close()
        else:
            plt.show()
            plt.close()
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from skimage import data, segmentation, color
from skimage.future import graph

#number = sys.argv[1]
path = "/home/fatih/my_git/sensorgan/outputs/lidar_examples/"

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

def turn_back_to_oneD(data):
    torch_version = torch.from_numpy(data).view(1, 5, -1, 1242)
    print("data", torch_version.shape)

    new_version = torch.max(torch_version, dim=1)[1].view(-1,1242)
    maxes = torch.max(torch_version, dim=1)[1].view(-1)
    print(maxes)
    count = 0
    for i in maxes:
        if i != 0:
            count+=1
    print(count)
    print("max, ", new_version.shape )
    return new_version.numpy()


colors = ['black','green','yellow','red','blue']
label = [0,1,2,3,4]
for i in range(0,99):
    number = str(i)
    generated = np.load(path+number+"_generated_.npz")["data"]
    expected_camera = np.load(path+number+"_expected_lidar_.npz")["data"]
    given_lidar = np.load(path+number+"_given_camera_.npz")["data"]

    fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.subplot(3, 1, 3)
    plt.imshow(color.label2rgb(turn_back_to_oneD(given_lidar),
                               colors=colors))
    plt.axis("off")
    plt.title("Expected Camera")
    plt.subplot(3, 1, 1)
    plt.imshow(color.label2rgb(turn_back_to_oneD(expected_camera),
                               colors=colors))
    plt.axis("off")
    plt.title("Given Lidar")
    plt.subplot(3, 1, 2)
    plt.imshow(color.label2rgb(turn_back_to_oneD(generated),
                               colors=colors))
    plt.axis("off")
    plt.title("Generated")
    plt.savefig(path+number+".png" )
    #plt.show()
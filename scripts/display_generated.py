import numpy as np
import sys
import matplotlib.pyplot as plt

number = sys.argv[1]
path = "/home/fatih/my_git/sensorgan/outputs/examples/"

def turn_back_to_oneD(data):

    real_version = np.zeros((375, 1242))
    for dim in range(0, 5):
        for i in range(0, 374):
            for j in range(0, 1241):
                if data[dim][i][j] > 0.1:
                    real_version[i][j] = dim
    return real_version


generated = np.load(path+number+"_generated_.npz")["data"]
expected_camera = np.load(path+number+"_expected_camera_.npz")["data"]
given_lidar = np.load(path+number+"_given_lidar_.npz")["data"]




fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=0.1, wspace=0.1)
plt.subplot(3, 1, 1)
plt.imshow(turn_back_to_oneD(given_lidar))
plt.axis("off")
plt.title("Given Lidar")
plt.subplot(3, 1, 3)
plt.imshow(turn_back_to_oneD(expected_camera))
plt.axis("off")
plt.title("Expected Camera")
plt.subplot(3, 1, 2)
plt.imshow(turn_back_to_oneD(generated))
plt.axis("off")
plt.title("Generated Camera")

plt.show()
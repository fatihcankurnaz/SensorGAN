from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from os import listdir
from os.path import join, isdir
import os


label_root_path = "/home/fatih/LidarLabelsCameraViewLast"

segmented_root_path = "/home/fatih/SegmentedInputLast"


def random_paths(image_number):

    rand = image_number

    count = 0
    for run in sorted(listdir(label_root_path) ):
        if isdir(join(label_root_path, run)):

            label_dir = join(label_root_path, run )
            for current_label in sorted(listdir(label_dir) ):

                count += 1
                if count == rand:


                    lidar_path = join(label_dir, current_label)
                    camera_path = join(segmented_root_path, run+"/segmented_"+current_label.split(".")[0].split("_")[-1]+".npz")


                    return lidar_path, camera_path

    return "", ""



def make_one_hot(data):
    new_version = np.zeros((5,375,1242))
    for i in range(0,375):
        for j in range(0,1242):
            if data[i][j] == 0:
                new_version[0][i][j] = 1

            if data[i][j] == 1:
                new_version[1][i][j] = 1

            if data[i][j] == 2:
                new_version[2][i][j] = 1

            if data[i][j] == 3:
                new_version[3][i][j] = 1

            if data[i][j] == 4:
                new_version[4][i][j] = 1
    return new_version


def main():

    for i in range(1,12920):

        paths = random_paths(i)
        lidar = paths[0]
        camera = paths[1]

        try:
            current_lidar = np.load(lidar)["data"]
            current_camera = np.load(camera)["data"]
        except:
            print("##############################")
            print("np load problem at: ", i, " -- ", lidar)
            print("##############################")
            continue

        # try:
        #     current_lidar = make_one_hot(current_lidar)
        #     current_camera = make_one_hot(current_camera)
        # except:
        #     print("##############################")
        #     print("fix function problem: ", i, " -- ", lidar)
        #     print("##############################")
        #     continue

        try:
            print(lidar)
            print(lidar.split(".")[0])
            np.save(lidar.split(".")[0]+".npy",current_lidar)
            np.save(camera.split(".")[0]+".npy",current_camera)
        except:
            print("##############################")
            print("save problem: ", i, " -- ", )
            print("##############################")

            continue
        print(i)








if __name__ == "__main__":

    main()
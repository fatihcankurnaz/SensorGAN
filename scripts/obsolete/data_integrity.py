from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from os import listdir
from os.path import join, isdir
import os
from shutil import copyfile



label_root_path = "/home/fatih/Inputs/LidarData"


def random_paths(image_number):

    rand = image_number

    count = 0
    for run in sorted(listdir(label_root_path) ):
        if isdir(join(label_root_path, run)):

            label_dir = join(label_root_path, run )
            new_label_dir = join(new_label_root_path, run)
            for current_label in sorted(listdir(label_dir) ):

                count += 1
                if count == rand:
                    lidar_path = join(label_dir, current_label)
                    return lidar_path

    return "", ""


def count_class(data):
    background_count = np.sum(data[0])
    road_count = np.sum(data[1])
    car_count = np.sum(data[2])
    pedesterian_count = np.sum(data[3])
    bicycle_count = np.sum(data[4])


    return background_count, road_count, car_count, pedesterian_count, bicycle_count

def main():
    background_count = 0
    road_count = 0
    car_count = 0
    ped_count = 0
    bicycle_count = 0
    for i in range(1,12920):

        paths = random_paths(i)
        lidar = paths[0]

        try:
            current_lidar = np.load(lidar)["data"].reshape(5,375,1242)
        except:
            print("##############################")
            print("np load problem at: ", i, " -- ", lidar)
            print("##############################")
            continue


        background_count += np.sum(current_lidar[0])
        road_count += np.sum(current_lidar[1])
        car_count += np.sum(current_lidar[2])
        ped_count += np.sum(current_lidar[3])
        bicycle_count += np.sum(current_lidar[4])









if __name__ == "__main__":

    main()
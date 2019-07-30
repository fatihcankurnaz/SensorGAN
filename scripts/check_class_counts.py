from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from os import listdir
from os.path import join, isdir
import os
from shutil import copyfile



label_root_path = "/home/fatih/Inputs/CameraData"


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

                    return lidar_path

    return ""


def main():
    background_count = 0
    road_count = 0
    car_count = 0
    ped_count = 0
    bicycle_count = 0
    for i in range(1,12920):

        paths = random_paths(i)
        lidar = paths

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

        if i%100 == 0:
            print(i)
            print("background points: ", background_count)
            print("road points: ", road_count)
            print("car points: ", car_count)
            print("human points: ", ped_count)
            print("bicycle points: ", bicycle_count)

    print("background points: ", background_count)
    print("road points: ", road_count)
    print("car points: ", car_count)
    print("human points: ", ped_count)
    print("bicycle points: ", bicycle_count)
    total_number_of_points = background_count + road_count + car_count + ped_count + bicycle_count
    print("total number points: ", total_number_of_points)
    print("Weights to be used")
    print("Background: ", float(total_number_of_points) / background_count)
    print("Road: ", float(total_number_of_points) / road_count)
    print("Car: ", float(total_number_of_points) / car_count)
    print("Human: ", float(total_number_of_points) / ped_count)
    print("Bicycle: ", float(total_number_of_points) / bicycle_count)








if __name__ == "__main__":

    main()
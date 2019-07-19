from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from os import listdir
from os.path import join, isdir
import os
from shutil import copyfile



label_root_path = "/home/fatih/LidarLabelsCameraViewReal"
segmented_root_path = "/home/fatih/SegmentedInputReal"
new_label_root_path = "/home/fatih/LidarLabelsCameraViewChecked"
new_segmented_root_path = "/home/fatih/SegmentedInputChecked"


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
                    new_lidar_path = join(new_label_dir, current_label)
                    camera_path = join(segmented_root_path, run+"/segmented_"+current_label.split(".")[0].split("_")[-1]+".npz")
                    new_camera_path = join(new_segmented_root_path, run+"/segmented_"+current_label.split(".")[0].split("_")[-1]+".npz")
                    new_camera_dir = join(new_segmented_root_path, run)

                    return lidar_path, camera_path, new_lidar_path, new_camera_path, new_label_dir, new_camera_dir

    return "", ""






def main():

    for i in range(1,12920):

        paths = random_paths(i)
        lidar = paths[0]
        camera = paths[1]
        new_lidar = paths[2]
        new_camera = paths[3]
        new_lidar_dir = paths[4]
        new_camera_dir = paths[5]

        if not os.path.exists(new_lidar_dir):
            print(new_lidar_dir)
            os.makedirs(new_lidar_dir)

        if not os.path.exists(new_camera_dir):
            print(new_camera_dir)
            os.makedirs(new_camera_dir)

        try:
            current_lidar = np.load(lidar)["data"].reshape(5,375,1242)
            current_camera = np.load(camera)["data"].reshape(5,375,1242)
        except:
            print("##############################")
            print("np load problem at: ", i, " -- ", lidar)
            print("##############################")
            continue

        copyfile(lidar, new_lidar)
        copyfile(camera, new_camera)


        print(i)








if __name__ == "__main__":

    main()
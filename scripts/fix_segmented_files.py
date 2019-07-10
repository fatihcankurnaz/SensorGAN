from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from os import listdir
from os.path import join, isdir
import os


label_root_path = "/home/fatih/LidarLabelsCameraView"
segmented_root_path = "/home/fatih/semantic_segmented_input"


def random_paths(image_number):

    rand = image_number

    count = 0
    for run in sorted(listdir(label_root_path) ):
        if isdir(join(label_root_path, run)):

            label_dir = join(label_root_path, run )
            for current_label in sorted(listdir(label_dir) ):

                count += 1
                if count == rand:


                    label_path = join(label_dir, current_label)
                    segment_path = join(segmented_root_path, run+"/segmented_"+current_label.split(".")[0].split("_")[-1]+".npy")
                    save_path = join("/home/fatih/SegmentedInput", run+"/segmented_"+current_label.split(".")[0].split("_")[-1]+".npz")
                    save_dir = join("/home/fatih/SegmentedInput", run)

                    return segment_path, save_path, save_dir

    return "", "", ""



def fix_my_segment(data):
    for i in range(0,375):
        for j in range(0,1242):
            if data[i][j] == 5:
                data[i][j] = 0
    return data


def main():

    for i in range(1,12920):

        paths = random_paths(i)
        segment_path = paths[0]
        save_path = paths[1]
        save_dir = paths[2]
        if not os.path.exists(save_dir):
            print(save_dir)
            os.makedirs(save_dir)
        try:
            current_segment = np.load(segment_path)
        except:
            print("##############################")
            print("np load problem at: ", i, " -- ", segment_path)
            print("##############################")
            continue

        try:
            current_segment = fix_my_segment(current_segment[0])
        except:
            print("##############################")
            print("fix function problem: ", i, " -- ", segment_path)
            print("##############################")
            continue

        try:
            np.savez_compressed(save_path, data=current_segment)
        except:
            print("##############################")
            print("save problem: ", i, " -- ", )
            print("##############################")

            continue








if __name__ == "__main__":

    main()
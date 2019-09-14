from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from os import listdir
from os.path import join, isdir
import os



label_root_path = "/SPACE/kan/Data/KITTI_labeledPC_with_BBs/2011_09_26"




def plot_full_label(velo_full):
    mylabelnums = {
        'road': 1,
        'car': 2,
        'person': 3,
        'cyclist': 4,
        'None': 0,
    }
    zeros = np.zeros((375, 1242))
    for i in velo_full:
        x, y, z, i, label, u, v = i
        if label != 'None':
            if label not in mylabelnums.keys(): print('error point', i, '\n error label', velo_full[int(i[2]), :])
            zeros[int(v)][int(u)] = mylabelnums[label]
        else:
            if label not in mylabelnums.keys(): print('error point', i, '\n error label', velo_full[int(i[2]), :])
            zeros[int(v)][int(u)] = mylabelnums[label]

    return zeros




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

                    save_path = join("/home/fatih/LidarLabelsCameraView", run+"/cameraView_"+current_label.split(".")[0].split("_")[-1]+".npz")
                    save_dir = join("/home/fatih/LidarLabelsCameraView", run)

                    return label_path, save_path, save_dir

    return "", "", ""



def main():

    for i in range(6500,12920):
        #print(i)
        paths = random_paths(i)

        _labeled_velo = paths[0]
        save_path = paths[1]
        save_dir = paths[2]
        #print(_labeled_velo)
        try:
            velo_full_labeled = np.load(_labeled_velo, allow_pickle=True)
        except:
            print("##############################")
            print("can not load: ", i, " -- ", _labeled_velo)
            print("##############################")

            continue
        try:
            label_image = plot_full_label(velo_full_labeled)
        except:
            print("##############################")
            print("broken file: ", i, " -- ", _labeled_velo)
            print("##############################")

            continue
        if not os.path.exists(save_dir):
            print(save_dir)
            os.makedirs(save_dir)

        np.savez_compressed(save_path, data= label_image)








if __name__ == "__main__":

    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from display_image import RGBImage


# 12919 images
from os import listdir
from os.path import join, isdir
image_root_path = "/SPACE/kan/Data/KITTI_raw_data/kitti/2011_09_26"
import cv2
from scipy import misc
import numpy as np
from PIL import Image



count = 0
my_d = RGBImage()
sum  = 0
for items1 in listdir(image_root_path):
    if isdir(join(image_root_path,items1)):

        left_image_directory = join(image_root_path, items1+"/image_02/data")
        for image_paths in listdir(left_image_directory):

            current_image =join(left_image_directory, image_paths)
            img = Image.open(current_image)
            arr = np.array(img)  # 640x480x4 array
            if count == 0:
                sum = np.mean(arr, axis=(0, 1))
            else:
                sum += np.mean(arr, axis=(0, 1))
            # print(sum)
            # print(arr[0][0].shape)
            # my_d.update_image(current_image)
            # my_d.display_image()
            # # print(join(left_image_directory,image_paths))
            count += 1
            print(count)




print(sum/count)
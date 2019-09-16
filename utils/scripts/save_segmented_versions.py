from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import pickle
import cv2
import os
from os import listdir
from os.path import join, isdir
import random
from runKITTIDataGeneratorForObjectDataset import processData
import PC2ImageConverter
import time
import scipy.misc
import cv2


from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.insert(0, '/home/fatih/other_git/PSPNet-tensorflow')

#import myscript
#
from model import PSPNet101, PSPNet50
from tools import *

image_root_path = "/SPACE/kan/Data/KITTI_raw_data/kitti/2011_09_26"
seg_save_root_path = "/home/fatih/semantic_segmented_input"



ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150,
                'model': PSPNet50,
                'weights_path': './model/pspnet50-ade20k/model.ckpt-0'}

cityscapes_param = {'crop_size': [1242, 375],
                    'num_classes': 19,
                    'model': PSPNet101,
                    'weights_path': "/home/fatih/other_git/PSPNet-tensorflow/model/pspnet101-cityscapes/model.ckpt-0"}

IMAGE_MEAN = np.array((94.5192887,  99.58598114, 94.95947993), dtype=np.float32)


param = cityscapes_param

"""
    Chooses a random image to display with bbs and labels
"""


def random_paths(image_number):

    rand = image_number#random.randint(1, 12919)
    count = 0
    for run in sorted(listdir(image_root_path) ):
        if isdir(join(image_root_path, run)):

            left_image_directory = join(image_root_path, run + "/image_02/data")
            for current_image in sorted(listdir(left_image_directory) ):

                left_image_path= join(left_image_directory, current_image)

                count += 1

                if count == rand:
                    save_dir = join(seg_save_root_path, run)
                    save_loc =  join(seg_save_root_path, run + "/segmented_"+ current_image )
                    np_save = join(seg_save_root_path, run+"/segmented_"+current_image.split(".")[0]+".npz")
                    return left_image_path, save_dir, save_loc, np_save

    return "", "", ""


def fixer(input):

    for i in range(0,375):
        for j in range(0,1242):
            if input[0][i][j] == 0:
                input[0][i][j] = 1

            elif input[0][i][j] == 1:
                input[0][i][j] = 0

            elif input[0][i][j] == 2:
                input[0][i][j] = 0

            elif input[0][i][j] == 3:
                input[0][i][j] = 0

            elif input[0][i][j] == 4:
                input[0][i][j] = 0

            elif input[0][i][j] == 5:
                input[0][i][j] = 0

            elif input[0][i][j] == 6:
                input[0][i][j] = 0

            elif input[0][i][j] == 7:
                input[0][i][j] = 0

            elif input[0][i][j] == 8:
                input[0][i][j] = 0

            elif input[0][i][j] == 9:
                input[0][i][j] = 0

            elif input[0][i][j] == 10:
                input[0][i][j] = 0

            elif input[0][i][j] == 11:
                input[0][i][j] = 3

            elif input[0][i][j] == 12:
                input[0][i][j] = 4

            elif input[0][i][j] == 13:
                input[0][i][j] = 2

            elif input[0][i][j] == 14:
                input[0][i][j] = 2

            elif input[0][i][j] == 15:
                input[0][i][j] = 2

            elif input[0][i][j] == 16:
                input[0][i][j] = 2

            elif input[0][i][j] == 17:
                input[0][i][j] = 4

            elif input[0][i][j] == 18:
                input[0][i][j] = 4

    return input


class PspInference:
    def __init__(self,image_path):
        self.img_path = image_path
        img_np, filename = load_img(self.img_path)
        img_shape = tf.shape(img_np)
        h, w = (tf.maximum(param['crop_size'][0], img_shape[0]), tf.maximum(param['crop_size'][1], img_shape[1]))
        img = preprocess(img_np, h, w)

        self.PSPNet = param['model']
        self.net = self.PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])

        self.raw_output = self.net.layers['conv6']

        # Predictions.
        self.raw_output_up = tf.image.resize_bilinear(self.raw_output, size=[h, w], align_corners=True)
        self.raw_output_up = tf.image.crop_to_bounding_box(self.raw_output_up, 0, 0, img_shape[0], img_shape[1])
        self.raw_output_up = tf.argmax(self.raw_output_up, dimension=3)

        self.pred = decode_labels(self.raw_output_up, img_shape, param['num_classes'])

        # Init tf Session
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)

        self.ckpt_path = param['weights_path']
        self.loader = tf.train.Saver(var_list=tf.global_variables())
        self.loader.restore(self.sess, self.ckpt_path)
        print("Restored model parameters from {}".format(self.ckpt_path))


    def infer(self):
        #start = time.time()
        one_d = fixer(self.sess.run(self.raw_output_up))
        preds = self.sess.run(self.pred)
        # given_count += 1
        #print(self.pred)
        #end = time.time()
        #print(end - start)
        res = preds[0]/255
        return res, one_d

    def update_image(self, image_path):
        self.img_path = image_path
        img_np, filename = load_img(self.img_path)
        img_shape = tf.shape(img_np)
        h, w = (tf.maximum(param['crop_size'][0], img_shape[0]), tf.maximum(param['crop_size'][1], img_shape[1]))
        img = preprocess(img_np, h, w)
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.net = self.PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])

        self.raw_output = self.net.layers['conv6']
        # Predictions.
        self.raw_output_up = tf.image.resize_bilinear(self.raw_output, size=[h, w], align_corners=True)
        self.raw_output_up = tf.image.crop_to_bounding_box(self.raw_output_up, 0, 0, img_shape[0], img_shape[1])
        self.raw_output_up = tf.argmax(self.raw_output_up, dimension=3)
        self.pred = decode_labels(self.raw_output_up, img_shape, param['num_classes'])

        # Init tf Session
        #self.core = tf.ConfigProto()
        #self.core.gpu_options.allow_growth = True
        #self.sess = tf.Session(core=self.core)
        #self.init = tf.global_variables_initializer()

        #self.sess.run(self.init)

        #self.ckpt_path = param['weights_path']
        #self.loader = tf.train.Saver(var_list=tf.global_variables())
        self.loader.restore(self.sess, self.ckpt_path)
        #print("Restored model parameters from {}".format(self.ckpt_path))

def main():
    print(sys.argv)
    first = True
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    my_psp = ""
    for i in range(start, end):
        print(i)
        paths = random_paths(i)
        print(paths[0],paths[2])
        left_image_path = paths[0]
        seg_save_dir = paths[1]
        seg_save_path = paths[2]
        np_save_path = paths[3]

        if not os.path.exists(seg_save_dir):
            os.makedirs(seg_save_dir)

        if first:
            start = time.time()
            my_psp = PspInference(left_image_path)
            segmented_images = my_psp.infer()
            end = time.time()
            print(end - start)


            plt.imsave(seg_save_path, segmented_images[0])
            np.savez_compressed(np_save_path,segmented_images[1])
            first = False
        else:
            start = time.time()
            my_psp.update_image(left_image_path)
            segmented_images = my_psp.infer()
            end = time.time()
            print(end - start)

            plt.imsave(seg_save_path, segmented_images[0])
            np.savez_compressed(np_save_path, segmented_images[1])














if __name__ == "__main__":

    main()
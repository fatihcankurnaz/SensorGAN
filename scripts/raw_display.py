from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import pickle
import cv2
from os import listdir
from os.path import join, isdir
import random
from runKITTIDataGeneratorForObjectDataset import processData
import PC2ImageConverter
import time



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
trans_root_path = "/SPACE/kan/Data/KITTI_labeledPC_with_BBs/TransformationMatrix"
bbs2d_root_path = "/SPACE/kan/Data/KITTI_labeledPC_with_BBs/BBs_2d/2011_09_26"
bbs3d_root_path = "/SPACE/kan/Data/KITTI_labeledPC_with_BBs/BBs_3d/2011_09_26"
label_root_path = "/SPACE/kan/Data/KITTI_labeledPC_with_BBs/2011_09_26"



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
def cart2hom( pts_3d,col =1):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending col, default is 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,col))))
    return pts_3d_hom


def Project3dConerTo2dImage(point,transformation_matrix):
    pts_2d = transformation_matrix.dot( cart2hom(point).T).T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    return pts_2d.T
#------------------------------------------


def loadBoundingBox(boundingBox):
    with open(boundingBox,'rb') as f:
        return pickle.load(f)


def plot_full_label(velo_full, image):
    mycolors = {
        'road': (128, 64, 128),
        'car': (119, 10, 32),
        'person': (219, 19, 60),
        'cyclist': (0, 0, 142),
        'None': (69, 69, 69),
    }

    for i in velo_full:
        x, y, z, i, label, u, v = i
        if label != 'None':
            if label not in mycolors.keys(): print('error point', i, '\n error label', velo_full[int(i[2]), :])
            cv2.circle(img=image, center=(int(u), int(v)), radius=2, thickness=-1, color=mycolors[label])
        else:
            if label not in mycolors.keys(): print('error point', i, '\n error label', velo_full[int(i[2]), :])
            cv2.circle(img=image, center=(int(u), int(v)), radius=2, thickness=-1, color=mycolors[label])

    return image


def GT_3d_image(image, boundingbox):
    type_c = {'car': (0, 255, 255), 'person': (0, 255, 0), 'cyclist': (255, 255, 0)}

    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])
    _image = np.zeros([375, 1242, 3])
    for BB in boundingbox:
        label, corner = BB
        print(label)
        print(corner)

        if label!='None':
            for k in line_order:
                cv2.line(image,
                         (int(corner[0][k[0]]), int(corner[1][k[0]])),
                         (int(corner[0][k[1]]), int(corner[1][k[1]])),
                         type_c[label], 2)

    plt.title("3D Tracklet display on image")
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    return image



"""
    Chooses a random image to display with bbs and labels
"""


def random_paths():

    rand = 200#random.randint(1, 12919)
    count = 0
    for run in sorted(listdir(image_root_path) ):
        if isdir(join(image_root_path, run)):

            left_image_directory = join(image_root_path, run + "/image_02/data")
            for current_image in sorted(listdir(left_image_directory) ):

                left_image_path= join(left_image_directory, current_image)

                count += 1

                if count == rand:
                    #print(count)
                    bb2_path = join(bbs2d_root_path,
                                    run+"/BB_2d_2011_09_26_"+run.split("_")[4]+"_"+
                                    current_image.split(".")[0]+".bin")

                    bb3_path = join(bbs3d_root_path,
                                    run + "/BB_3d_2011_09_26_" + run.split("_")[4] + "_" +
                                    current_image.split(".")[0] + ".bin")

                    trans_path = join(trans_root_path, "trans_2011_09_26_"+run.split("_")[4]+".npy")

                    label_path = join(label_root_path,
                                    run+ "/full_label_2011_09_26_"+run.split("_")[4]+"_"+
                                    current_image.split(".")[0]+".npy")

                    right_image_path = join(image_root_path,run+"/image_03/data")
                    right_image_path = join(right_image_path,current_image)


                    return trans_path, label_path, bb2_path, bb3_path, left_image_path, right_image_path

    return "", "", "", "", "", ""




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
        start = time.time()
        preds = self.sess.run(self.pred)
        # given_count += 1
        #print(self.pred)
        end = time.time()
        print(end - start)
        res = preds[0]/255
        return res

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

def main():
    paths = random_paths()

    for i in paths:
        print(i)
    trans = np.load(paths[0], allow_pickle=True)
    _labeled_velo = paths[1]
    _Boundingbox_2d = paths[2]
    _Boundingbox_3d = paths[3]
    _rgb_img = paths[4]

    left_image_path = paths[4]
    right_image_path = paths[5]
    velo_full_labeled = np.load(_labeled_velo, allow_pickle=True)
    boundingbox_2d = loadBoundingBox(_Boundingbox_2d)
    boundingbox_3d = loadBoundingBox(_Boundingbox_3d)

    rgb_img = cv2.imread(_rgb_img)
    velo_on_image = plot_full_label(velo_full_labeled, rgb_img)

    test3dbox = 0
    vis_fov_img= ""
    if test3dbox:
        BB_3d_to_2d = []
        for i in range(len(boundingbox_3d)):
            tmp = []
        label, bb3d = boundingbox_3d[i]
        tmp.append(label)
        tmp.append(Project3dConerTo2dImage(bb3d.T, transformation_matrix=trans))
        BB_3d_to_2d.append(tmp)
        image_velo_3dbox = GT_3d_image(velo_on_image, BB_3d_to_2d)
        vis_fov_img = 'image_velo_from3d.png'
    else:
        image_velo_3dbox = GT_3d_image(velo_on_image, boundingbox_2d)
        vis_fov_img = './output/image_velo_from2d.png'

        cv2.imwrite(vis_fov_img, image_velo_3dbox)

    PC2ImgConv = PC2ImageConverter.PC2ImgConverter(imgChannel=5, xRange=[0, 50], yRange=[-6, 12], zRange=[-10, 8],
                                                   xGridSize=0.2, yGridSize=0.3, zGridSize=0.3, maxImgHeight=64,
                                                   maxImgWidth=256, maxImgDepth=64)
    outputFileName = "./output/Cloud.png"
    myPsp = PspInference(left_image_path)
    processData(_labeled_velo, _Boundingbox_3d, PC2ImgConv, outputFileName)

    left_image = mpimg.imread(left_image_path)
    right_image = myPsp.infer()
    bird_eye_image = mpimg.imread(outputFileName)
    overlayed_image = mpimg.imread(vis_fov_img)
    fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
    # left=None, bottom=None, right=None, top=None,
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    plt.subplot(2, 2, 1)
    plt.imshow(left_image)
    plt.axis("off")
    plt.title("LEFT IMAGE")
    plt.subplot(2, 2, 2)
    plt.imshow(right_image)
    plt.axis("off")
    plt.title("SEMANTIC SEGMENTATION")
    plt.subplot(2, 2, 3)
    plt.imshow(bird_eye_image)
    plt.axis("off")
    plt.title("BIRD EYE VIEW")
    plt.subplot(2, 2, 4)
    plt.imshow(overlayed_image)
    plt.axis("off")
    plt.title("LABELED")
    plt.tight_layout()
    plt.savefig("./output/result.png")
    plt.close()







if __name__ == "__main__":

    main()
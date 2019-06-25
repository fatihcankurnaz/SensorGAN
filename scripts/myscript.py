import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mping
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from os import listdir
from os.path import join, isdir
from matplotlib.lines import Line2D


from model import PSPNet101, PSPNet50
from tools import *
import time


ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150, 
                'model': PSPNet50,
                'weights_path': './model/pspnet50-ade20k/model.ckpt-0'}

cityscapes_param = {'crop_size': [1242, 375],
                    'num_classes': 19,
                    'model': PSPNet101,
                    'weights_path': './model/pspnet101-cityscapes/model.ckpt-0'}

IMAGE_MEAN = np.array((94.5192887,  99.58598114, 94.95947993), dtype=np.float32)

image_root_path = "/SPACE/kan/Data/KITTI_raw_data/kitti/2011_09_26"

param = cityscapes_param 

image_path = '/home/fatih/Downloads/data_semantics/testing/image_2/000003_10.png'

given_count= 0


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
        with tf.variable_scope('', reuse=True):
            self.net = self.PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])

        self.raw_output = self.net.layers['conv6']
        # Predictions.
        self.raw_output_up = tf.image.resize_bilinear(self.raw_output, size=[h, w], align_corners=True)
        self.raw_output_up = tf.image.crop_to_bounding_box(self.raw_output_up, 0, 0, img_shape[0], img_shape[1])
        self.raw_output_up = tf.argmax(self.raw_output_up, dimension=3)
        self.pred = decode_labels(self.raw_output_up, img_shape, param['num_classes'])



def image_new(previous):
    skip =16
    for items in sorted(listdir(image_root_path) ):


        if isdir(join(image_root_path,items)):

            left_image_directory = join(image_root_path, items+"/image_02/data")

            for image_paths in sorted(listdir(left_image_directory) ):

                if skip > 0:
                    skip -= 1
                    break
                if skip == 0:
                    skip -= 1
                current_image = join(left_image_directory, image_paths)

                if previous == "":

                    return current_image

                else:


                    if previous >= current_image:
                        continue
                    else:
                        return current_image
            if(skip == -1) and previous != "":

                return None






my_img_path = ""
given_count = 1 ## TODO if given is zero it does not work
myPsp = ""
my_imgs = []
my_results = []

def new_inf(previous):

    global  my_img_path
    global given_count
    global myPsp
    my_img_path = image_new(previous)

    if my_img_path != None:
        if given_count == 1:
            myPsp = PspInference(my_img_path)
            given_count = 0
            return myPsp.infer()
        else:
            myPsp.update_image(my_img_path)
            return myPsp.infer()
    else:
        return None




labels = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32)]
                # 18 = bicycle

labels = np.array(labels)/255
labels = labels.tolist()

legend_elements = [Line2D([0], [0], color=labels[0], lw=4, label='Road'),
                    Line2D([0], [0], color=labels[1], lw=4, label='Sidewalk'),
                    Line2D([0], [0], color=labels[2], lw=4, label='Building'),
                    Line2D([0], [0], color=labels[3], lw=4, label='Wall'),
                    Line2D([0], [0], color=labels[4], lw=4, label='Fence'),
                    Line2D([0], [0], color=labels[5], lw=4, label='Pole'),
                    Line2D([0], [0], color=labels[6], lw=4, label='Traffic Light'),
                    Line2D([0], [0], color=labels[7], lw=4, label='Traffic Sign'),
                    Line2D([0], [0], color=labels[8], lw=4, label='Vegetation'),
                    Line2D([0], [0], color=labels[9], lw=4, label='Terrain'),
                    Line2D([0], [0], color=labels[10], lw=4, label='Sky'),
                    Line2D([0], [0], color=labels[11], lw=4, label='Person'),
                    Line2D([0], [0], color=labels[12], lw=4, label='Rider'),
                    Line2D([0], [0], color=labels[13], lw=4, label='Car'),
                    Line2D([0], [0], color=labels[14], lw=4, label='Truck'),
                    Line2D([0], [0], color=labels[15], lw=4, label='Bus'),
                    Line2D([0], [0], color=labels[16], lw=4, label='Train'),
                    Line2D([0], [0], color=labels[17], lw=4, label='Motocycle'),
                    Line2D([0], [0], color=labels[18], lw=4, label='Bicycle')
                   ]



fig, ax_list = plt.subplots(3, sharex=True, sharey=True)
fig.set_figheight(12)
fig.set_figwidth(25)
fig.subplots_adjust(hspace=0.01, wspace=0.01)
fig.tight_layout()
image1 = new_inf("")
image2 = mping.imread(my_img_path)

ax_list[0].axis('off')
ax_list[1].axis('off')
ax_list[2].axis("off")
plt.tight_layout()


first = ax_list[0].imshow(image1)
second = ax_list[1].imshow(image2)
ax_list[2].legend(handles= legend_elements,loc="upper center",ncol=10)
images = [first, second]

def update(i):


    images[0] = ax_list[0].imshow(new_inf(my_img_path))

    images[1] = ax_list[1].imshow(mping.imread(my_img_path))

    return images





ani = animation.FuncAnimation(fig, update, interval=0, blit=True,
                                repeat_delay=0)

plt.show()






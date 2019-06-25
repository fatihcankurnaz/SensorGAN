from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


class RGBImage(object):

    def __init__(self, image_path=None, display_time=0):
        self.image_path = image_path
        self.display_time = display_time

    def update_image(self, new_image_path):
        self.image_path = new_image_path

    def update_display_time(self, new_display_time):
        self.display_time = new_display_time

    def display_image(self):
        img = mpimg.imread(self.image_path)
        plt.imshow(img)
        plt.show()
        time.sleep(self.display_time)
        plt.close()